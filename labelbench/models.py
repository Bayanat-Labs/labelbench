from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from .datasets import ARATOX_TOXIC_LABELS, Example, MultiLabelExample


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    raw_response: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MultiLabelPrediction:
    labels: tuple[str, ...]
    confidence: float
    scores: dict[str, float]
    raw_response: str | None = None
    metadata: dict[str, Any] | None = None


class Classifier(Protocol):
    name: str

    def fit(self, examples: Iterable[Example]) -> None: ...

    def predict(self, text: str) -> Prediction: ...


def normalize_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ـ", "")
    )
    return WHITESPACE.sub(" ", text.strip().lower())


def char_ngrams(text: str, minimum: int = 2, maximum: int = 5) -> Counter[str]:
    normalized = f" {normalize_arabic(text)} "
    grams: Counter[str] = Counter()
    for size in range(minimum, maximum + 1):
        for index in range(max(0, len(normalized) - size + 1)):
            grams[normalized[index : index + size]] += 1
    return grams


class MajorityClassifier:
    name = "majority"

    def __init__(self) -> None:
        self.label = ""
        self.confidence = 0.0

    def fit(self, examples: Iterable[Example]) -> None:
        counts = Counter(example.label for example in examples)
        if not counts:
            raise ValueError("cannot fit majority classifier on no examples")
        self.label, frequency = counts.most_common(1)[0]
        self.confidence = frequency / sum(counts.values())

    def predict(self, text: str) -> Prediction:
        del text
        if not self.label:
            raise RuntimeError("classifier must be fit before prediction")
        return Prediction(self.label, self.confidence)


class CharacterNgramNB:
    name = "char_ngram_nb"

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.class_docs: Counter[str] = Counter()
        self.class_feature_totals: Counter[str] = Counter()
        self.feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()
        self.total_docs = 0

    def fit(self, examples: Iterable[Example]) -> None:
        for example in examples:
            features = char_ngrams(example.text)
            self.class_docs[example.label] += 1
            self.feature_counts[example.label].update(features)
            self.class_feature_totals[example.label] += sum(features.values())
            self.vocabulary.update(features)
            self.total_docs += 1
        if not self.total_docs:
            raise ValueError("cannot fit n-gram classifier on no examples")

    def _scores(self, text: str) -> tuple[list[tuple[str, float]], int]:
        if not self.total_docs:
            raise RuntimeError("classifier must be fit before prediction")
        features = char_ngrams(text)
        vocabulary_size = max(1, len(self.vocabulary))
        scores: list[tuple[str, float]] = []
        number_of_classes = len(self.class_docs)
        for label, document_count in self.class_docs.items():
            score = math.log(
                (document_count + self.alpha)
                / (self.total_docs + self.alpha * number_of_classes)
            )
            denominator = (
                self.class_feature_totals[label] + self.alpha * vocabulary_size
            )
            label_features = self.feature_counts[label]
            for feature, frequency in features.items():
                probability = (label_features.get(feature, 0) + self.alpha) / denominator
                score += frequency * math.log(probability)
            scores.append((label, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores, max(1, sum(features.values()))

    def predict_proba(self, text: str) -> dict[str, float]:
        scores, _ = self._scores(text)
        maximum = scores[0][1]
        exponentials = {
            label: math.exp(score - maximum) for label, score in scores
        }
        denominator = sum(exponentials.values())
        return {
            label: value / denominator for label, value in exponentials.items()
        }

    def predict(self, text: str) -> Prediction:
        scores, feature_mass = self._scores(text)
        best_label, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else best_score
        normalized_margin = (best_score - second_score) / feature_mass
        confidence = 1 / (1 + math.exp(-normalized_margin))
        return Prediction(best_label, confidence)


class OneVsRestCharacterNgram:
    def __init__(
        self,
        labels: tuple[str, ...] = ARATOX_TOXIC_LABELS,
        threshold: float = 0.5,
        tune_thresholds: bool = False,
        validation_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.name = (
            "char_ngram_nb_ovr_tuned"
            if tune_thresholds
            else "char_ngram_nb_ovr"
        )
        self.labels = labels
        self.thresholds = {label: threshold for label in labels}
        self.classifiers = {label: CharacterNgramNB() for label in labels}
        self.tune_thresholds = tune_thresholds
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.model_metadata: dict[str, Any] = {
            "threshold_tuning": tune_thresholds,
            "thresholds": dict(self.thresholds),
        }

    def fit(self, examples: Iterable[MultiLabelExample]) -> None:
        materialized = list(examples)
        if not materialized:
            raise ValueError("cannot fit multi-label classifier on no examples")
        if self.tune_thresholds:
            self._tune(materialized)
        for label, classifier in self.classifiers.items():
            binary_examples = [
                Example(
                    id=example.id,
                    text=example.text,
                    label="positive" if label in example.labels else "negative",
                    metadata=example.metadata,
                )
                for example in materialized
            ]
            classifier.fit(binary_examples)
        self.model_metadata["thresholds"] = dict(self.thresholds)

    def _fit_binary_classifiers(
        self,
        examples: list[MultiLabelExample],
    ) -> dict[str, CharacterNgramNB]:
        classifiers = {label: CharacterNgramNB() for label in self.labels}
        for label, classifier in classifiers.items():
            classifier.fit(
                [
                    Example(
                        id=example.id,
                        text=example.text,
                        label=(
                            "positive" if label in example.labels else "negative"
                        ),
                        metadata=example.metadata,
                    )
                    for example in examples
                ]
            )
        return classifiers

    def _tune(self, examples: list[MultiLabelExample]) -> None:
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        indices = list(range(len(examples)))
        random.Random(self.seed).shuffle(indices)
        validation_size = max(1, round(len(indices) * self.validation_fraction))
        validation_indices = set(indices[:validation_size])
        train = [
            example
            for index, example in enumerate(examples)
            if index not in validation_indices
        ]
        validation = [
            example
            for index, example in enumerate(examples)
            if index in validation_indices
        ]
        temporary = self._fit_binary_classifiers(train)
        probabilities = [
            {
                label: temporary[label]
                .predict_proba(example.text)
                .get("positive", 0.0)
                for label in self.labels
            }
            for example in validation
        ]

        def exact_accuracy(thresholds: dict[str, float]) -> float:
            correct = 0
            for example, scores in zip(validation, probabilities):
                selected = {
                    label
                    for label in self.labels
                    if scores[label] >= thresholds[label]
                }
                predicted = selected or {"not"}
                correct += predicted == set(example.labels)
            return correct / len(validation)

        candidates = [value / 100 for value in range(5, 100, 5)]
        for _ in range(4):
            changed = False
            for label in self.labels:
                best_threshold = self.thresholds[label]
                best_score = exact_accuracy(self.thresholds)
                for candidate in candidates:
                    trial = dict(self.thresholds)
                    trial[label] = candidate
                    score = exact_accuracy(trial)
                    if score > best_score + 1e-12 or (
                        abs(score - best_score) <= 1e-12
                        and candidate > best_threshold
                    ):
                        best_score = score
                        best_threshold = candidate
                if best_threshold != self.thresholds[label]:
                    self.thresholds[label] = best_threshold
                    changed = True
            if not changed:
                break
        self.model_metadata.update(
            {
                "validation_fraction": self.validation_fraction,
                "validation_examples": len(validation),
                "validation_subset_accuracy": exact_accuracy(self.thresholds),
                "thresholds": dict(self.thresholds),
            }
        )

    def predict(self, text: str) -> MultiLabelPrediction:
        scores = {
            label: classifier.predict_proba(text).get("positive", 0.0)
            for label, classifier in self.classifiers.items()
        }
        selected = tuple(
            label
            for label in self.labels
            if scores[label] >= self.thresholds[label]
        )
        labels = selected or ("not",)
        decision_distances = [
            abs(scores[label] - self.thresholds[label]) * 2 for label in self.labels
        ]
        confidence = min(1.0, min(decision_distances)) if decision_distances else 0.0
        return MultiLabelPrediction(
            labels=labels,
            confidence=confidence,
            scores={**scores, "not": 1.0 - max(scores.values(), default=0.0)},
        )
