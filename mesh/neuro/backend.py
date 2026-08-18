"""OpenAI-compatible structured LLM backends for the Neuro-Symbolic bridge.

The bridge must drive local open-weight backends (vLLM, Ollama, llama.cpp) that
serve a standard OpenAI-compatible ``/chat/completions`` endpoint.  Nothing here
is hard-wired to a closed-source provider: the base URL, model name and (optionally)
API key are fully configurable via :class:`LLMConfig`.

``instructor`` is used to coerce raw model output into a validated Pydantic
model when the backend advertises tool-calling support; for backends that do
not, we fall back to JSON-extraction parsing.  A :class:`MockNeuroBackend` is
provided so the retry loop can be unit-tested with no LLM and no network.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, Field, PrivateAttr

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMConfig(BaseModel):
    """Configuration for an OpenAI-compatible LLM backend.

    ``base_url`` points at the local open-weight server (e.g.
    ``http://localhost:8000/v1`` for vLLM, ``http://localhost:11434/v1`` for
    Ollama).  ``model_name`` selects the served model (e.g. ``gemma-2b-it``,
    ``Qwen/Qwen2.5-3B-Instruct``).  No credential or provider is hard-coded.
    """

    base_url: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    api_key: str | None = None
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout_s: float = Field(default=60.0, gt=0.0)


class StructuredBackend:
    """Abstract contract for a backend that produces validated Pydantic objects."""

    async def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        **extra: Any,
    ) -> T:
        raise NotImplementedError

    async def generate_text(
        self,
        messages: Sequence[dict[str, str]],
        **extra: Any,
    ) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any resources (HTTP clients, etc.)."""


class OpenAICompatibleBackend(StructuredBackend):
    """Structured backend backed by an OpenAI-compatible endpoint via ``instructor``.

    Supports vLLM, Ollama (OpenAI-compatible mode) and llama.cpp servers.  The
    client is constructed lazily on first use so that simply importing this
    module never touches the network or requires credentials.
    """

    _client: Any = PrivateAttr(default=None)
    _instructor_client: Any = PrivateAttr(default=None)

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _init_clients(self) -> None:
        if self._client is not None:
            return
        import instructor
        from openai import AsyncOpenAI

        openai_kwargs: dict[str, Any] = {
            "base_url": self.config.base_url,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_s,
        }
        if self.config.api_key is not None:
            openai_kwargs["api_key"] = self.config.api_key
        self._client = AsyncOpenAI(**openai_kwargs)
        # instructor patches the OpenAI client for structured output.  TOOLS mode
        # works with every OpenAI-compatible server that supports tool calling
        # (vLLM, Ollama >= 0.1.30, llama.cpp server).
        self._instructor_client = instructor.from_openai(self._client, mode=instructor.Mode.TOOLS)

    async def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        **extra: Any,
    ) -> T:
        self._init_clients()
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": list(messages),
            "response_model": response_model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        kwargs.update(extra)
        result = await self._instructor_client.chat.completions.create(**kwargs)
        return result  # type: ignore[no-any-return]

    async def generate_text(
        self,
        messages: Sequence[dict[str, str]],
        **extra: Any,
    ) -> str:
        self._init_clients()
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        kwargs.update(extra)
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def aclose(self) -> None:
        client = self._client
        if client is not None:
            await client.close()
            self._client = None
            self._instructor_client = None


# ─── Mock backend for offline / deterministic tests ──────────────────────────


class _ResponseSequence:
    """Holds an ordered sequence of canned responses (or a single one)."""

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses: list[Any] = list(responses)
        self._index = 0
        self.calls: list[Sequence[dict[str, str]]] = []

    def next(self) -> Any:
        if not self._responses:
            raise RuntimeError("MockNeuroBackend has no canned responses left")
        if self._index >= len(self._responses):
            self._index = len(self._responses) - 1
        val = self._responses[self._index]
        self._index += 1
        return val


class MockNeuroBackend(StructuredBackend):
    """Deterministic backend used in tests.

    Yields a pre-arranged response for each call.  Pass a single value to
    always return it; pass a sequence to simulate an LLM that first produces a
    policy-violating payload and then a corrected one (exercising the retry
    loop).
    """

    def __init__(self, responses: T | Sequence[T]) -> None:
        # Normalize a scalar into a one-element list without consuming the
        # single-value case (a Pydantic model instance is not a sequence).
        if isinstance(responses, BaseModel):
            self._seq = _ResponseSequence([responses])
        else:
            self._seq = _ResponseSequence(list(responses))
        self.config = LLMConfig(
            base_url="mock://test",
            model_name="mock-llm",
        )

    async def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        **extra: Any,
    ) -> T:
        self._seq.calls.append(list(messages))
        value = self._seq.next()
        # A canned BaseException instance is re-raised to simulate a backend /
        # schema-parse failure (e.g. the LLM emitted unparseable JSON).
        if isinstance(value, BaseException):
            raise value
        if not isinstance(value, response_model):
            raise RuntimeError(
                f"MockNeuroBackend expected {response_model.__name__}, got {type(value).__name__}"
            )
        return value

    async def generate_text(self, messages: Sequence[dict[str, str]], **extra: Any) -> str:
        self._seq.calls.append(list(messages))
        value = self._seq.next()
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)

    async def aclose(self) -> None:
        pass

    @property
    def call_count(self) -> int:
        return len(self._seq.calls)
