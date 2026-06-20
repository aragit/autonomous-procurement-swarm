"""
LLM Inference Engine for the Procurement Swarm.

Supports three backends:
1. MockLLM — deterministic responses, instant, no download (Active Blueprint default)
2. Transformers — real LLM via HuggingFace (CPU, downloads on first use)
3. vLLM — fast batched inference (CPU, requires build)

The engine exposes an OpenAI-compatible chat completion API.
"""

import os
import json
import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


class BaseLLMEngine(ABC):
    """Abstract base for LLM inference backends."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def shutdown(self):
        pass


class MockLLMEngine(BaseLLMEngine):
    """
    Deterministic mock LLM for Active Blueprint demonstration.
    Generates realistic negotiation responses without any model download.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.model_name = "mock-llm-blueprint"
        print("[LLM] Using MockLLM — deterministic, instant, no download")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()

        # Extract role from system prompt to determine agent type
        system_prompt = messages[0]["content"] if messages else ""
        role = "unknown"
        if "procurement agent" in system_prompt.lower():
            role = "buyer"
        elif "supplier agent" in system_prompt.lower():
            role = "seller"
        elif "market intelligence" in system_prompt.lower():
            role = "market"
        elif "arbiter" in system_prompt.lower():
            role = "arbiter"

        # Extract context values from prompt
        spot_price = self._extract_float(system_prompt, "spot price: $")
        geo_risk = self._extract_float(system_prompt, "geopolitical risk:")

        # Generate contextually appropriate response
        content = self._generate_response(role, spot_price, geo_risk)

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            tokens_in=len(str(messages)),
            tokens_out=len(content.split()),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _extract_float(self, text: str, key: str) -> float:
        """Extract float value after key in text."""
        try:
            idx = text.lower().find(key.lower())
            if idx == -1:
                return 0.0
            start = idx + len(key)
            # Find the number
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] in ".,"):
                end += 1
            return float(text[start:end].replace(",", ""))
        except (ValueError, IndexError):
            return 0.0

    def _generate_response(self, role: str, spot_price: float, geo_risk: float) -> str:
        """Generate deterministic but realistic negotiation response."""
        if role == "buyer":
            # Buyer counters at 85-95% of spot, or accepts if price is low
            if spot_price > 0 and self.rng.random() > 0.3:
                counter = round(spot_price * self.rng.uniform(0.85, 0.95), 2)
                return json.dumps({
                    "type": "counter",
                    "material": "steel",
                    "quantity": self.rng.choice([500, 1000, 2000]),
                    "counter_price": counter,
                    "justification": f"Market oversupply and {geo_risk:.0%} geopolitical risk warrants discount",
                    "deadline": "2026-06-30"
                })
            else:
                return json.dumps({
                    "type": "accept",
                    "material": "steel",
                    "quantity": 1000,
                    "final_price": round(spot_price * 0.92, 2) if spot_price > 0 else 400.0,
                    "delivery_date": "2026-07-15"
                })

        elif role == "seller":
            # Seller offers at 105-115% of spot, or counters higher
            if spot_price > 0 and self.rng.random() > 0.3:
                offer = round(spot_price * self.rng.uniform(1.05, 1.15), 2)
                return json.dumps({
                    "type": "counter",
                    "material": "steel",
                    "quantity": self.rng.choice([500, 1000, 2000]),
                    "counter_price": offer,
                    "justification": f"Premium quality and supply chain resilience amid {geo_risk:.0%} risk",
                    "deadline": "2026-06-25"
                })
            else:
                return json.dumps({
                    "type": "offer",
                    "material": "steel",
                    "quantity": 1000,
                    "unit_price": round(spot_price * 1.10, 2) if spot_price > 0 else 500.0,
                    "delivery_date": "2026-07-15",
                    "payment_terms": "net_30"
                })

        elif role == "market":
            trend = "rising" if self.rng.random() > 0.5 else "falling"
            return json.dumps({
                "type": "report",
                "material": "steel",
                "spot_price": round(spot_price, 2) if spot_price > 0 else 450.0,
                "trend": trend,
                "volatility": round(self.rng.uniform(0.1, 0.4), 2),
                "geo_risk": round(geo_risk, 2),
                "supply_disruption": geo_risk > 0.6,
                "recommendation": f"Consider hedging given {geo_risk:.0%} geopolitical risk"
            })

        else:  # arbiter or unknown
            return json.dumps({
                "type": "validate",
                "agent": "unknown",
                "message_valid": True,
                "errors": []
            })

    def shutdown(self):
        pass


class TransformersEngine(BaseLLMEngine):
    """
    Real LLM engine using HuggingFace Transformers.
    Downloads model on first use (~6GB for Qwen2.5-3B).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cpu",
    ):
        print(f"[LLM] Loading {model_name} via Transformers (CPU)...")
        print(f"[LLM] This will download ~6GB on first run. Go get coffee.")
        start = time.time()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        load_time = time.time() - start
        print(f"[LLM] Model loaded in {load_time:.1f}s")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )

        input_ids = model_inputs.input_ids
        tokens_in = input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        tokens_out = generated_ids.shape[1] - tokens_in
        response = self.tokenizer.decode(
            generated_ids[0][tokens_in:], skip_special_tokens=True
        )

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=response.strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def shutdown(self):
        del self.model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class VLLMEngine(BaseLLMEngine):
    """
    Primary LLM engine using vLLM (CPU).
    Faster batched inference, preferred when available.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        tensor_parallel_size: int = 1,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Use: VLLM_TARGET_DEVICE=cpu pip install vllm"
            )

        print(f"[LLM] Loading {model_name} via vLLM (CPU)...")
        start = time.time()

        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            device="cpu",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams

        load_time = time.time() - start
        print(f"[LLM] Model loaded in {load_time:.1f}s")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()

        from vllm import SamplingParams

        prompt = self._format_chat(messages)

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(prompt, sp)
        response = outputs[0].outputs[0].text

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=response.strip(),
            tokens_in=len(outputs[0].prompt_token_ids),
            tokens_out=len(outputs[0].outputs[0].token_ids),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Simple chat formatting for vLLM."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts) + "\nAssistant:"

    def shutdown(self):
        pass


class LLMEngineFactory:
    """Factory to select the best available LLM backend."""

    @staticmethod
    def create(
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        prefer_vllm: bool = True,
        use_mock: bool = False,
    ) -> BaseLLMEngine:
        if use_mock:
            return MockLLMEngine()

        if prefer_vllm:
            try:
                return VLLMEngine(model_name)
            except Exception as e:
                print(f"[LLM] vLLM failed ({e}), falling back to Transformers")
                return TransformersEngine(model_name)
        return TransformersEngine(model_name)
