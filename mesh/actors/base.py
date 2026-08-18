"""Base classes for mesh archetype actors.

Provides the MeshActor base class with common functionality for all four
archetypes: blackboard interaction, kernel validation, and lifecycle management.

The shared :class:`~mesh.neuro.types.NeuralProposal` / :class:`~mesh.neuro.types.SymbolicVerdict`
contracts live in the ray-free :mod:`mesh.neuro.types` module so the
Neuro-Symbolic bridge can be exercised in pure-Python tests without a Ray
cluster.
"""

from __future__ import annotations

import contextlib
import uuid
from abc import ABC, abstractmethod
from typing import Any

import ray
import structlog
from ray.actor import ActorHandle

from mesh.channels import ChannelType, check_read_permission, check_write_permission
from mesh.neuro.kernel import symbolic_validate
from mesh.neuro.types import NeuralProposal, SymbolicVerdict

logger = structlog.get_logger(__name__)

__all__ = [
    "MeshActor",
    "SafetyKernelActor",
    "NeuralProposal",
    "SymbolicVerdict",
]


@ray.remote(max_restarts=0, max_task_retries=0)
class SafetyKernelActor:
    """Singleton safety kernel actor that validates all neural proposals.

    This is the non-overridable "root of trust" — every archetype MUST call
    kernel.validate.remote() before writing to the blackboard.
    """

    def __init__(self) -> None:
        self._validations = 0
        self._rejections = 0

    def validate(self, proposal: NeuralProposal) -> SymbolicVerdict:
        """Deterministic validation of a neural proposal.

        Delegates to the pure, ray-free :func:`mesh.neuro.kernel.symbolic_validate`
        so the policy rules are identical in cluster mode and in unit tests.

        Checks:
        1. Price bounds (must be positive, within reasonable range)
        2. Lead time bounds (1-365 days)
        3. Payment terms whitelist
        4. Material whitelist
        5. Budget compliance
        6. Confidence threshold
        """
        self._validations += 1
        verdict = symbolic_validate(proposal)
        if not verdict.approved:
            self._rejections += 1
        return verdict

    def stats(self) -> dict[str, int]:
        return {"validations": self._validations, "rejections": self._rejections}


class MeshActor(ABC):
    """Base class for all four archetype actors.

    Each archetype instance is a Ray actor that:
    1. Reads from allowed channels on the DistributedBlackboard
    2. Generates proposals (may use LLM/SLM for cognitive tasks)
    3. MANDATORY: Validates proposals through SafetyKernelActor
    4. Writes validated output to allowed channels

    The kernel is a named Ray actor registered as "safety_kernel".
    """

    def __init__(
        self,
        actor_id: str,
        archetype: str,
        blackboard: ActorHandle[Any],
        kernel: ActorHandle[Any] | None = None,
        cognitive_engine: str | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.archetype = archetype
        self.blackboard = blackboard
        self.kernel = kernel or ray.get_actor("safety_kernel")
        self.cognitive_engine = cognitive_engine
        self._step_count = 0
        self._errors: list[str] = []

    @abstractmethod
    async def perceive(self, blackboard: ActorHandle[Any]) -> dict[str, Any]:
        """Read allowed channels and return perception data."""
        ...

    @abstractmethod
    async def reason(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Generate a proposal from perception data."""
        ...

    @abstractmethod
    async def act(self, blackboard: ActorHandle[Any], proposal: dict[str, Any]) -> None:
        """Write validated proposal to allowed channels."""
        ...

    async def step(self) -> dict[str, Any]:
        """Execute one full perceive → reason → validate → act cycle."""
        self._step_count += 1
        logger.info(
            "actor_step_start",
            actor_id=self.actor_id,
            archetype=self.archetype,
            step=self._step_count,
        )

        try:
            # 1. Perceive: read what I'm allowed to see
            perception = await self.perceive(self.blackboard)

            # 2. Reason: generate proposal (may use LLM/SLM)
            proposal = await self.reason(perception)

            # 3. MANDATORY: Kernel validation before any write
            neural_proposal = NeuralProposal(
                proposal_id=uuid.uuid4().hex,
                archetype=self.archetype,
                payload=proposal,
                confidence=proposal.get("confidence", 1.0),
                structured=True,
            )

            verdict: SymbolicVerdict = await self.kernel.validate.remote(neural_proposal)

            if not verdict.approved:
                self._errors.append(verdict.reason)
                logger.warning(
                    "actor_proposal_rejected",
                    actor_id=self.actor_id,
                    archetype=self.archetype,
                    reason=verdict.reason,
                    proposal_id=neural_proposal.proposal_id,
                )
                return {"status": "rejected", "reason": verdict.reason}

            # 4. Act: write to blackboard (ACL-enforced by blackboard)
            final_payload = verdict.clamped_payload or proposal
            await self.act(self.blackboard, final_payload)

            logger.info(
                "actor_step_complete",
                actor_id=self.actor_id,
                archetype=self.archetype,
                step=self._step_count,
            )
            return {"status": "success"}

        except Exception as e:
            self._errors.append(str(e))
            logger.error(
                "actor_step_error",
                actor_id=self.actor_id,
                archetype=self.archetype,
                step=self._step_count,
                error=str(e),
            )
            # Deposit failure trace for observability
            with contextlib.suppress(Exception):
                await self.blackboard.write.remote(
                    self.archetype,
                    ChannelType.DISCOVERY if self.archetype == "scout" else ChannelType.SCORE,
                    {
                        "type": "failure",
                        "error": str(e),
                        "actor_id": self.actor_id,
                        "step": self._step_count,
                    },
                )
            return {"status": "error", "error": str(e)}

    def get_stats(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "archetype": self.archetype,
            "step_count": self._step_count,
            "errors": self._errors,
        }

    async def read_channel(self, channel: ChannelType, limit: int = 100) -> list[dict[str, Any]]:
        """Read from a channel with ACL enforcement."""
        if not check_read_permission(self.archetype, channel):
            raise PermissionError(
                f"Archetype '{self.archetype}' cannot read from '{channel.value}'"
            )
        return await self.blackboard.read.remote(self.archetype, channel, limit)  # type: ignore[no-any-return]

    async def read_channels(
        self, channels: list[ChannelType], limit: int = 100
    ) -> dict[ChannelType, list[dict[str, Any]]]:
        """Read from multiple channels atomically."""
        for ch in channels:
            if not check_read_permission(self.archetype, ch):
                raise PermissionError(f"Archetype '{self.archetype}' cannot read from '{ch.value}'")
        return await self.blackboard.read_multi.remote(self.archetype, channels, limit)  # type: ignore[no-any-return]

    async def write_channel(
        self,
        channel: ChannelType,
        payload: dict[str, Any],
        parent_ids: list[str] | None = None,
    ) -> str:
        """Write to a channel with ACL enforcement."""
        if not check_write_permission(self.archetype, channel):
            raise PermissionError(f"Archetype '{self.archetype}' cannot write to '{channel.value}'")
        return await self.blackboard.write.remote(self.archetype, channel, payload, parent_ids)  # type: ignore[no-any-return]
