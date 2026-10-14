from __future__ import annotations

from dataclasses import dataclass

from .metrics import (
    ScoredPrediction,
    bootstrap_accuracy_interval,
    safe_automation_coverage,
)


@dataclass(frozen=True)
class ScoredMultiLabelPrediction:
    gold: frozenset[str]
    predicted: frozenset[str]
    confidence: float

    @property
    def correct(self) -> bool:
        return self.gold == self.predicted


def subset_accuracy(rows: list[ScoredMultiLabelPrediction]) -> float:
    if not rows:
        return 0.0
    return sum(row.correct for row in rows) / len(rows)


def _counts(
    rows: list[ScoredMultiLabelPrediction],
    label: str | None = None,
) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for row in rows:
        labels = (
            {label}
            if label is not None
            else row.gold | row.predicted
        )
        for current in labels:
            in_gold = current in row.gold
            in_prediction = current in row.predicted
            true_positive += int(in_gold and in_prediction)
            false_positive += int(not in_gold and in_prediction)
            false_negative += int(in_gold and not in_prediction)
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def multilabel_report(
    rows: list[ScoredMultiLabelPrediction],
    labels: tuple[str, ...],
    seed: int = 42,
) -> dict:
    micro_counts = _counts(rows)
    per_label = {}
    for label in labels:
        true_positive, false_positive, false_negative = _counts(rows, label)
        per_label[label] = {
            "f1": _f1(true_positive, false_positive, false_negative),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    macro_f1 = (
        sum(item["f1"] for item in per_label.values()) / len(per_label)
        if per_label
        else 0.0
    )
    hamming_errors = sum(
        len(row.gold ^ row.predicted) for row in rows
    )
    interval_rows = [
        ScoredPrediction(
            gold="correct",
            predicted="correct" if row.correct else "incorrect",
            confidence=row.confidence,
        )
        for row in rows
    ]
    interval = bootstrap_accuracy_interval(interval_rows, seed=seed)
    return {
        "subset_accuracy": subset_accuracy(rows),
        "subset_accuracy_bootstrap_95": {
            "low": interval[0],
            "high": interval[1],
        },
        "micro_f1": _f1(*micro_counts),
        "macro_f1": macro_f1,
        "hamming_loss": (
            hamming_errors / (len(rows) * len(labels))
            if rows and labels
            else 0.0
        ),
        "mean_gold_cardinality": (
            sum(len(row.gold) for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_predicted_cardinality": (
            sum(len(row.predicted) for row in rows) / len(rows) if rows else 0.0
        ),
        "sac_95_exact_set": safe_automation_coverage(interval_rows, 0.95),
        "sac_98_exact_set": safe_automation_coverage(interval_rows, 0.98),
        "per_label": per_label,
    }
