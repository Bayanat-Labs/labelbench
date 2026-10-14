from __future__ import annotations

import json
import re
import urllib.request
from typing import Iterable

from .datasets import (
    ARATOX_LABELS,
    IADD_REGIONS,
    Example,
    MultiLabelExample,
)
from .models import MultiLabelPrediction, Prediction


NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9_.-]+")


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


class OllamaToxicityClassifier:
    """Deterministic zero-shot binary toxicity annotation through Ollama."""

    labels = ("toxic", "not_toxic")

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        seed: int = 42,
        confidence_mode: str = "constant",
    ) -> None:
        if confidence_mode not in {"constant", "verbalized"}:
            raise ValueError("confidence_mode must be constant or verbalized")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.confidence_mode = confidence_mode
        safe_name = NON_ALPHANUMERIC.sub("_", model).strip("_")
        suffix = "_verbal_confidence" if confidence_mode == "verbalized" else ""
        self.name = f"ollama_{safe_name}_zero_shot{suffix}"
        self.model_metadata = self._show_model()
        self.model_metadata["confidence_mode"] = confidence_mode

    def fit(self, examples: Iterable[Example]) -> None:
        del examples

    def _post(self, path: str, payload: dict, timeout: int = 120) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def _show_model(self) -> dict:
        response = self._post("/api/show", {"model": self.model}, timeout=30)
        details = response.get("details", {})
        tags_request = urllib.request.Request(f"{self.base_url}/api/tags")
        with urllib.request.urlopen(tags_request, timeout=30) as tags_response:
            tags = json.load(tags_response)
        tag = next(
            (
                item
                for item in tags.get("models", [])
                if item.get("name") == self.model or item.get("model") == self.model
            ),
            {},
        )
        return {
            "provider": "ollama_local",
            "requested_model": self.model,
            "digest": tag.get("digest"),
            "size_bytes": tag.get("size"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "modified_at": response.get("modified_at"),
        }

    @staticmethod
    def parse_label(content: str) -> str:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return "__invalid__"
        label = payload.get("label")
        return label if label in OllamaToxicityClassifier.labels else "__invalid__"

    @classmethod
    def parse_label_and_confidence(cls, content: str) -> tuple[str, float]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return "__invalid__", 0.0
        label = payload.get("label")
        parsed_label = label if label in cls.labels else "__invalid__"
        confidence = _coerce_confidence(payload.get("confidence"))
        return parsed_label, confidence

    def _schema(self) -> dict:
        properties = {"label": {"type": "string", "enum": list(self.labels)}}
        required = ["label"]
        if self.confidence_mode == "verbalized":
            properties["confidence"] = {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "Your calibrated probability that the label is exactly "
                    "correct, from 0 to 1."
                ),
            }
            required.append("confidence")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def predict(self, text: str) -> Prediction:
        system_prompt = (
            "You are labeling Arabic social-media text for toxicity. "
            "Choose toxic when the text contains an insult, curse, hatred, "
            "racism, sexual abuse, threat or violence, or an attack on a "
            "person's appearance. Choose not_toxic otherwise. Judge the text "
            "itself, including Arabic dialect and slang."
        )
        if self.confidence_mode == "verbalized":
            system_prompt += (
                " Also return confidence as your calibrated probability that "
                "the selected label is exactly correct, using a number from 0 "
                "to 1. Do not use 1.0 unless the decision is effectively "
                "certain."
            )
        system_prompt += " Return only the required JSON object."
        response = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "format": self._schema(),
                "options": {
                    "temperature": 0,
                    "seed": self.seed,
                    "num_predict": 20,
                },
            },
        )
        content = response.get("message", {}).get("content", "")
        if self.confidence_mode == "verbalized":
            label, confidence = self.parse_label_and_confidence(content)
        else:
            label = self.parse_label(content)
            confidence = 1.0 if label in self.labels else 0.0
        return Prediction(
            label=label,
            confidence=confidence if label in self.labels else 0.0,
            raw_response=content,
            metadata={
                "confidence_mode": self.confidence_mode,
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_duration_ns": response.get("total_duration", 0),
            },
        )


class OllamaMultiLabelToxicityClassifier(OllamaToxicityClassifier):
    labels = ARATOX_LABELS

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        seed: int = 42,
        confidence_mode: str = "constant",
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            seed=seed,
            confidence_mode=confidence_mode,
        )
        safe_name = NON_ALPHANUMERIC.sub("_", model).strip("_")
        suffix = "_verbal_confidence" if confidence_mode == "verbalized" else ""
        self.name = f"ollama_{safe_name}_multilabel_zero_shot{suffix}"

    def fit(self, examples: Iterable[MultiLabelExample]) -> None:
        del examples

    @staticmethod
    def parse_labels(content: str) -> tuple[str, ...]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return ("__invalid__",)
        labels = payload.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or any(label not in ARATOX_LABELS for label in labels)
            or len(labels) != len(set(labels))
        ):
            return ("__invalid__",)
        selected = set(labels)
        if "not" in selected and len(selected) > 1:
            return ("__invalid__",)
        return tuple(label for label in ARATOX_LABELS if label in selected)

    @classmethod
    def parse_labels_and_confidence(cls, content: str) -> tuple[tuple[str, ...], float]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return ("__invalid__",), 0.0
        labels = cls.parse_labels(content)
        return labels, _coerce_confidence(payload.get("confidence"))

    def _schema(self) -> dict:
        properties = {
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": list(ARATOX_LABELS)},
                "minItems": 1,
                "uniqueItems": True,
            }
        }
        required = ["labels"]
        if self.confidence_mode == "verbalized":
            properties["confidence"] = {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "Your calibrated probability that the complete label set "
                    "is exactly correct, from 0 to 1."
                ),
            }
            required.append("confidence")
        return {"type": "object", "properties": properties, "required": required}

    def predict(self, text: str) -> MultiLabelPrediction:
        system_prompt = (
            "You are a native-aware Arabic toxicity data annotator. Assign all "
            "labels that apply. Labels: appearance (attacks on looks/body), "
            "cussing (swearing or direct insults), hatred (hate or dehumanizing "
            "attacks), racial (race/ethnicity/nationality attacks), sexual "
            "(sexual insults or sexual abuse), violence (threats or advocacy of "
            "physical harm), not (none of the six toxic categories). Multiple "
            "toxic labels may apply. Use not only when no toxic label applies. "
            "Interpret Arabic dialect and slang."
        )
        if self.confidence_mode == "verbalized":
            system_prompt += (
                " Also return confidence as your calibrated probability that "
                "the entire selected label set is exactly correct, using a "
                "number from 0 to 1. Do not use 1.0 unless the complete label "
                "set is effectively certain."
            )
        system_prompt += " Return only the required JSON."
        response = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "format": self._schema(),
                "options": {
                    "temperature": 0,
                    "seed": self.seed,
                    "num_predict": 40,
                },
            },
        )
        content = response.get("message", {}).get("content", "")
        if self.confidence_mode == "verbalized":
            labels, confidence = self.parse_labels_and_confidence(content)
        else:
            labels = self.parse_labels(content)
            confidence = 1.0 if "__invalid__" not in labels else 0.0
        valid = "__invalid__" not in labels
        scores = {
            label: float(label in labels) for label in ARATOX_LABELS
        }
        return MultiLabelPrediction(
            labels=labels,
            confidence=confidence if valid else 0.0,
            scores=scores,
            raw_response=content,
            metadata={
                "confidence_mode": self.confidence_mode,
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_duration_ns": response.get("total_duration", 0),
            },
        )


class OllamaDialectRegionClassifier(OllamaToxicityClassifier):
    labels = IADD_REGIONS

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        seed: int = 42,
        confidence_mode: str = "constant",
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            seed=seed,
            confidence_mode=confidence_mode,
        )
        safe_name = NON_ALPHANUMERIC.sub("_", model).strip("_")
        suffix = "_verbal_confidence" if confidence_mode == "verbalized" else ""
        self.name = f"ollama_{safe_name}_dialect_region_zero_shot{suffix}"

    @staticmethod
    def parse_label(content: str) -> str:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return "__invalid__"
        label = payload.get("label")
        return label if label in IADD_REGIONS else "__invalid__"

    def predict(self, text: str) -> Prediction:
        system_prompt = (
            "Classify the Arabic text into exactly one broad regional variety. "
            "GLF = Gulf Arabic; EGY = Egyptian Arabic; LEV = Levantine Arabic "
            "(Palestine, Jordan, Syria, Lebanon); MGH = Maghrebi Arabic "
            "(Morocco, Algeria, Tunisia); IRQ = Iraqi Arabic. Use dialect "
            "vocabulary, morphology, and phrasing. Do not choose based on the "
            "topic alone."
        )
        if self.confidence_mode == "verbalized":
            system_prompt += (
                " Also return confidence as your calibrated probability that "
                "the selected region label is exactly correct, using a number "
                "from 0 to 1. Do not use 1.0 unless the decision is effectively "
                "certain."
            )
        system_prompt += " Return only the required JSON object."
        response = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "format": self._schema(),
                "options": {
                    "temperature": 0,
                    "seed": self.seed,
                    "num_predict": 20,
                },
            },
        )
        content = response.get("message", {}).get("content", "")
        if self.confidence_mode == "verbalized":
            label, confidence = self.parse_label_and_confidence(content)
        else:
            label = self.parse_label(content)
            confidence = 1.0 if label in IADD_REGIONS else 0.0
        return Prediction(
            label=label,
            confidence=confidence if label in IADD_REGIONS else 0.0,
            raw_response=content,
            metadata={
                "confidence_mode": self.confidence_mode,
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_duration_ns": response.get("total_duration", 0),
            },
        )
