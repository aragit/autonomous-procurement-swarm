"""The Neuro-Symbolic bridge: structured (neuro) generation with symbolic auto-correction.

Flow:
    1. The backend produces a **structured** Pydantic object (schema-constrained).
    2. The object is flattened into a kernel payload and wrapped in a
       :class:`~mesh.neuro.types.NeuralProposal`.
    3. The :class:`~mesh.actors.base.SafetyKernelActor` performs the mandatory
       symbolic validation.
    4. If the kernel rejects the proposal (an "OPA/Rego-style" policy violation),
       the rejection reason is fed back into the LLM prompt and generation is
       retried -- up to ``max_retries`` -- before the bridge concedes and returns
       the last verdict so the actor can fall back to its deterministic path.

The bridge is deliberately decoupled from Ray: the kernel is injected as an
async callable ``validator``.  Inside a Ray actor the validator wraps
``await self.kernel.validate.remote(...)``; in tests it is a plain coroutine
stub.  This keeps the retry logic fully unit-testable without a cluster.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel

from mesh.neuro.backend import StructuredBackend
from mesh.neuro.types import NeuralProposal, SymbolicVerdict

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: A kernel validator is any async callable accepting a NeuralProposal and
#: returning a SymbolicVerdict.  In a Ray actor it wraps
#: ``await self.kernel.validate.remote(proposal)``.
KernelValidator = Callable[[NeuralProposal], Any]


class ProtocolViolation(RuntimeError):  # noqa: N818
    """Raised when the neuro bridge exhausts retries and the proposal is still
    rejected by the SafetyKernelActor.  Only raised when ``raise_on_exhaustion``
    is set; otherwise a :class:`NeuroResult` with ``exhausted=True`` is returned.
    """


@dataclass
class NeuroResult:
    """Outcome of a (possibly retried) structured-proposal attempt."""

    payload: dict[str, Any] | None
    verdict: SymbolicVerdict
    attempts: int
    model_instance: BaseModel | None = None
    exhausted: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


class NeuroSymbolicBridge:
    """Mediates between a structured LLM backend and the SafetyKernelActor.

    Parameters
    ----------
    backend:
        The structured LLM backend (e.g. :class:`OpenAICompatibleBackend`).
    validator:
        Async callable enforcing symbolic policy (``NeuralProposal`` ->
        ``SymbolicVerdict``).  Inject the kernel here.
    max_retries:
        Maximum number of LLM generation attempts.  The first attempt is always
        made, so ``max_retries=1`` means a single shot with no correction.
    raise_on_exhaustion:
        When ``True`` the bridge raises :class:`ProtocolViolation` after the last
        failed attempt instead of returning it.  Actors typically keep this
        ``False`` so they can fall back to deterministic logic.
    """

    _REJECTION_PROMPT = """\
Your previous output was REJECTED by the SafetyKernelActor (symbolic policy
validation) and was NOT written to the blackboard.

Rejection reason: {reason}

Policy violations:
{violations}

Please correct the payload so that ALL of the following hold:
- price > 0 and <= 1,000,000 (total_price <= 1,000,000)
- lead_time_days in [1, 365]
- payment_terms in [net_30, net_60, cod, letter_of_credit]
- material in [steel, aluminum, copper, plastic, lumber, rubber]
- unit_price * quantity <= budget (when budget is supplied)
- confidence >= 0.5

Return only a corrected response matching the original JSON schema. Do not
include any prose or explanations outside the structured fields."""

    def __init__(
        self,
        backend: StructuredBackend,
        validator: KernelValidator,
        max_retries: int = 3,
        raise_on_exhaustion: bool = False,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._backend = backend
        self._validator = validator
        self._max_retries = max_retries
        self._raise_on_exhaustion = raise_on_exhaustion

    @classmethod
    def _rejection_prompt(cls, reason: str, violations: list[str]) -> str:
        """Build the re-prompt message carrying a policy failure back to the LLM."""
        bullet = "\n".join(f"- {v}" for v in violations) if violations else f"- {reason}"
        return cls._REJECTION_PROMPT.format(reason=reason, violations=bullet)

    async def safe_propose(
        self,
        *,
        archetype: str,
        response_model: type[T],
        messages: Sequence[dict[str, str]],
        payload_builder: Callable[[T], dict[str, Any]],
        correlation_id: str,
        confidence: float = 1.0,
        max_retries: int | None = None,
    ) -> NeuroResult:
        """Generate, validate and (if needed) auto-correct a structured proposal.

        Parameters
        ----------
        archetype:
            The archetype string registered with the kernel ACL (e.g. "scout").
        response_model:
            The Pydantic schema the backend must conform to.
        messages:
            The base chat messages for the LLM.  Re-prompts are appended to a
            *copy*, so the caller's list is never mutated.
        payload_builder:
            Converts the structured model into the flat kernel payload dict.
        correlation_id:
            Carried into the NeuralProposal metadata.
        confidence:
            Confidence reported to the kernel (defaults to 1.0 for LLM output;
            callers may lower it to gate acceptance).
        max_retries:
            Override the bridge-level default for this call.
        """
        limit = max_retries if max_retries is not None else self._max_retries
        working_messages: list[dict[str, str]] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        history: list[dict[str, Any]] = []
        last_verdict: SymbolicVerdict | None = None
        last_payload: dict[str, Any] | None = None
        model_instance: T | None = None

        for attempt in range(1, limit + 1):
            logger.info(
                "neuro_generation_attempt",
                archetype=archetype,
                attempt=attempt,
                max_retries=limit,
            )

            # 1. Structured (neuro) generation.
            try:
                model_instance = await self._backend.generate_structured(
                    working_messages, response_model
                )
            except Exception as exc:  # schema parse error / backend failure
                logger.warning(
                    "neuro_generation_error",
                    archetype=archetype,
                    attempt=attempt,
                    error=str(exc),
                )
                history.append(
                    {
                        "attempt": attempt,
                        "approved": False,
                        "phase": "generation",
                        "error": str(exc),
                    }
                )
                last_verdict = SymbolicVerdict.rejected(
                    reason=f"GENERATION_ERROR: {exc}",
                    violations=[f"generation_error: {exc}"],
                )
                if attempt < limit:
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last output could not be parsed into the required "
                                "schema. Please correct it to strictly match the schema "
                                f"({response_model.__name__}) with no extra text."
                            ),
                        }
                    )
                continue

            # 2. Flatten to a kernel payload and validate symbolically.
            payload = payload_builder(model_instance)
            proposal = NeuralProposal(
                proposal_id=uuid.uuid4().hex,
                archetype=archetype,
                payload=payload,
                confidence=confidence,
                structured=True,
                metadata={
                    "response_model": response_model.__name__,
                    "correlation_id": correlation_id,
                    "attempt": attempt,
                },
            )
            verdict: SymbolicVerdict = await self._validator(proposal)
            last_verdict = verdict
            last_payload = payload
            history.append(
                {
                    "attempt": attempt,
                    "approved": verdict.approved,
                    "phase": "kernel_validation",
                    "reason": verdict.reason,
                    "violations": verdict.violations,
                }
            )

            if verdict.approved:
                logger.info(
                    "neuro_proposal_accepted",
                    archetype=archetype,
                    attempt=attempt,
                )
                return NeuroResult(
                    payload=verdict.clamped_payload or payload,
                    verdict=verdict,
                    attempts=attempt,
                    model_instance=model_instance,
                    exhausted=False,
                    history=history,
                )

            # 3. Rejected -> auto-correct: feed the failure back into the prompt.
            logger.warning(
                "neuro_proposal_rejected",
                archetype=archetype,
                attempt=attempt,
                reason=verdict.reason,
                violations=verdict.violations,
            )
            if attempt < limit:
                working_messages.append(
                    {
                        "role": "user",
                        "content": self._rejection_prompt(verdict.reason, verdict.violations),
                    }
                )

        # 4. Retries exhausted.
        logger.error(
            "neuro_exhausted_retries",
            archetype=archetype,
            attempts=limit,
            last_reason=last_verdict.reason if last_verdict else None,
        )
        result = NeuroResult(
            payload=last_payload,
            verdict=last_verdict or SymbolicVerdict.rejected(reason="NO_ATTEMPTS_MADE"),
            attempts=limit,
            model_instance=model_instance,
            exhausted=True,
            history=history,
        )
        if self._raise_on_exhaustion:
            raise ProtocolViolation(
                f"{archetype} proposal rejected after {limit} attempts: {result.verdict.reason}"
            )
        return result
