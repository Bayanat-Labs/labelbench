from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .datasets import IADD_REGIONS, Example
from .models import Prediction
from .ollama import NON_ALPHANUMERIC, _coerce_confidence


class OpenAIResponsesDialectClassifier:
    """Strict-schema dialect classifier using OpenAI's Responses API."""

    labels = IADD_REGIONS

    def __init__(
        self,
        model: str = "gpt-5.4-mini-2026-03-17",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        confidence_mode: str = "verbalized",
    ) -> None:
        if confidence_mode not in {"constant", "verbalized"}:
            raise ValueError("confidence_mode must be constant or verbalized")
        self.model = model
        self.confidence_mode = confidence_mode
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required; copy .env.example and export the key"
            )
        self.base_url = base_url.rstrip("/")
        safe_name = NON_ALPHANUMERIC.sub("_", model).strip("_")
        suffix = "_verbal_confidence" if confidence_mode == "verbalized" else ""
        self.name = f"openai_{safe_name}_dialect_region_zero_shot{suffix}"
        self.model_metadata = {
            "provider": "openai",
            "requested_model": model,
            "api": "responses",
            "structured_outputs": True,
            "store": False,
            "confidence_mode": confidence_mode,
        }

    def fit(self, examples: Iterable[Example]) -> None:
        del examples

    @classmethod
    def parse_label(cls, content: str) -> str:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return "__invalid__"
        label = payload.get("label")
        return label if label in cls.labels else "__invalid__"

    @classmethod
    def parse_label_and_confidence(cls, content: str) -> tuple[str, float]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return "__invalid__", 0.0
        label = payload.get("label")
        parsed_label = label if label in cls.labels else "__invalid__"
        return parsed_label, _coerce_confidence(payload.get("confidence"))

    def build_payload(self, text: str) -> dict:
        system_prompt = (
            "Classify the Arabic text into exactly one broad regional variety. "
            "GLF = Gulf Arabic; EGY = Egyptian Arabic; LEV = Levantine Arabic "
            "(Palestine, Jordan, Syria, Lebanon); MGH = Maghrebi Arabic "
            "(Morocco, Algeria, Tunisia); IRQ = Iraqi Arabic. Use dialect "
            "vocabulary, morphology, and phrasing rather than topic. Return "
            "only the requested structured result."
        )
        properties = {
            "label": {
                "type": "string",
                "enum": list(self.labels),
            }
        }
        required = ["label"]
        if self.confidence_mode == "verbalized":
            system_prompt += (
                " Include confidence as your calibrated probability that the "
                "selected region label is exactly correct, from 0 to 1."
            )
            properties["confidence"] = {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "Calibrated probability that the selected label is exactly "
                    "correct, from 0 to 1."
                ),
            }
            required.append("confidence")
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "dialect_region_label",
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
            "reasoning": {"effort": "none"},
            "max_output_tokens": 32,
            "store": False,
        }

    @staticmethod
    def _output_text(response: dict) -> str:
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
        return ""

    def predict(self, text: str) -> Prediction:
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(self.build_payload(text)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as raw_response:
                response = json.load(raw_response)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI Responses API returned HTTP {error.code}: {detail}"
            ) from error
        content = self._output_text(response)
        if self.confidence_mode == "verbalized":
            label, confidence = self.parse_label_and_confidence(content)
        else:
            label = self.parse_label(content)
            confidence = 1.0 if label in self.labels else 0.0
        usage = response.get("usage") or {}
        self.model_metadata["resolved_model"] = response.get("model")
        return Prediction(
            label=label,
            confidence=confidence if label in self.labels else 0.0,
            raw_response=content,
            metadata={
                "confidence_mode": self.confidence_mode,
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "response_id": response.get("id"),
                "resolved_model": response.get("model"),
            },
        )
