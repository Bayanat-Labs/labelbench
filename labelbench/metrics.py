from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class ScoredPrediction:
    gold: str
    predicted: str
    confidence: float

    @property
    def correct(self) -> bool:
        return self.gold == self.predicted


def accuracy(rows: list[ScoredPrediction]) -> float:
    if not rows:
        return 0.0
    return sum(row.correct for row in rows) / len(rows)


def macro_f1(rows: list[ScoredPrediction]) -> float:
    per_label = classification_report(rows)
    if not per_label:
        return 0.0
    return sum(item["f1"] for item in per_label) / len(per_label)


def classification_report(
    rows: list[ScoredPrediction],
) -> list[dict[str, float | int | str]]:
    labels = sorted({row.gold for row in rows} | {row.predicted for row in rows})
    report: list[dict[str, float | int | str]] = []
    for label in labels:
        true_positive = sum(
            row.gold == label and row.predicted == label for row in rows
        )
        false_positive = sum(
            row.gold != label and row.predicted == label for row in rows
        )
        false_negative = sum(
            row.gold == label and row.predicted != label for row in rows
        )
        support = sum(row.gold == label for row in rows)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return report


def confusion_matrix(rows: list[ScoredPrediction]) -> dict[str, object]:
    labels = sorted({row.gold for row in rows} | {row.predicted for row in rows})
    counts = Counter((row.gold, row.predicted) for row in rows)
    return {
        "labels": labels,
        "matrix": [
            [counts[(gold, predicted)] for predicted in labels]
            for gold in labels
        ],
    }


def brier_score(rows: list[ScoredPrediction]) -> float:
    if not rows:
        return 0.0
    return sum(
        (row.confidence - float(row.correct)) ** 2 for row in rows
    ) / len(rows)


def expected_calibration_error(
    rows: list[ScoredPrediction],
    bins: int = 10,
) -> float:
    if not rows:
        return 0.0
    if bins <= 0:
        raise ValueError("bins must be positive")
    total_error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= row.confidence < upper
            or (bin_index == bins - 1 and row.confidence == 1.0)
        ]
        if not selected:
            continue
        mean_confidence = sum(row.confidence for row in selected) / len(selected)
        mean_accuracy = sum(row.correct for row in selected) / len(selected)
        total_error += (
            len(selected) / len(rows) * abs(mean_accuracy - mean_confidence)
        )
    return total_error


def bootstrap_accuracy_interval(
    rows: list[ScoredPrediction],
    iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    generator = random.Random(seed)
    values = []
    count = len(rows)
    correctness = [int(row.correct) for row in rows]
    for _ in range(iterations):
        values.append(
            sum(correctness[generator.randrange(count)] for _ in range(count)) / count
        )
    values.sort()
    low_index = max(0, int((alpha / 2) * iterations))
    high_index = min(iterations - 1, int((1 - alpha / 2) * iterations) - 1)
    return values[low_index], values[high_index]


def wilson_lower_bound(successes: int, total: int, z: float = 1.9599639845) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return max(0.0, (centre - margin) / denominator)


def safe_automation_coverage(
    rows: list[ScoredPrediction], target: float
) -> dict[str, float | int]:
    if not 0 < target <= 1:
        raise ValueError("target must be between 0 and 1")
    if not rows:
        return {
            "target": target,
            "accepted": 0,
            "total": 0,
            "coverage": 0.0,
            "empirical_accuracy": 0.0,
            "wilson_lower_bound": 0.0,
            "confidence_threshold": 1.0,
        }

    ranked = sorted(rows, key=lambda row: row.confidence, reverse=True)
    successes = 0
    accepted = 0
    best: dict[str, float | int] | None = None
    for confidence, group in groupby(ranked, key=lambda row: row.confidence):
        tied_rows = list(group)
        successes += sum(int(row.correct) for row in tied_rows)
        accepted += len(tied_rows)
        lower = wilson_lower_bound(successes, accepted)
        if lower >= target:
            best = {
                "target": target,
                "accepted": accepted,
                "total": len(rows),
                "coverage": accepted / len(rows),
                "empirical_accuracy": successes / accepted,
                "wilson_lower_bound": lower,
                "confidence_threshold": confidence,
            }
    if best is not None:
        return best
    return {
        "target": target,
        "accepted": 0,
        "total": len(rows),
        "coverage": 0.0,
        "empirical_accuracy": 0.0,
        "wilson_lower_bound": 0.0,
        "confidence_threshold": 1.0,
    }


def calibrated_safe_automation_coverage(
    calibration_rows: list[ScoredPrediction],
    evaluation_rows: list[ScoredPrediction],
    target: float,
) -> dict[str, object]:
    """Select the automation threshold on calibration rows, then evaluate it.

    This avoids the optimistic bias that occurs when a confidence threshold is
    selected and reported on the same examples. Tied confidence values are never
    split, matching :func:`safe_automation_coverage`.
    """
    calibration = safe_automation_coverage(calibration_rows, target)
    threshold = calibration["confidence_threshold"]
    if calibration["accepted"] == 0:
        selected: list[ScoredPrediction] = []
    else:
        selected = [
            row for row in evaluation_rows if row.confidence >= threshold
        ]
    successes = sum(int(row.correct) for row in selected)
    accepted = len(selected)
    return {
        "target": target,
        "selection": "calibration_split",
        "calibration": calibration,
        "evaluation": {
            "accepted": accepted,
            "total": len(evaluation_rows),
            "coverage": accepted / len(evaluation_rows) if evaluation_rows else 0.0,
            "empirical_accuracy": successes / accepted if accepted else 0.0,
            "wilson_lower_bound": wilson_lower_bound(successes, accepted),
            "confidence_threshold": threshold,
        },
    }


def top_confusions(
    rows: list[ScoredPrediction], limit: int = 10
) -> list[dict[str, int | str]]:
    counts = Counter(
        (row.gold, row.predicted)
        for row in rows
        if row.gold != row.predicted
    )
    return [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in counts.most_common(limit)
    ]
