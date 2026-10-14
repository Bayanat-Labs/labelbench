from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .datasets import Example, sha256_file
from .metrics import (
    ScoredPrediction,
    accuracy,
    brier_score,
    bootstrap_accuracy_interval,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    safe_automation_coverage,
    top_confusions,
)
from .models import Classifier


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_classifier(
    classifier: Classifier,
    train: list[Example],
    test: list[Example],
    output_dir: Path,
    train_file: Path | None,
    test_file: Path,
    dataset_id: str,
    task: str,
    seed: int = 42,
    resume: bool = True,
) -> dict:
    model_dir = output_dir / classifier.name
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_dir / "predictions.jsonl"
    fit_started = time.perf_counter()
    classifier.fit(train)
    fit_seconds = time.perf_counter() - fit_started

    existing_rows: dict[str, dict] = {}
    if predictions_path.exists():
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite existing predictions: {predictions_path}"
            )
        with predictions_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid cached prediction at line {line_number}: {error}"
                    ) from error
                if row["id"] in existing_rows:
                    raise ValueError(f"duplicate cached prediction id: {row['id']}")
                existing_rows[row["id"]] = row

    expected = {example.id: example for example in test}
    unknown_ids = sorted(set(existing_rows) - set(expected))
    if unknown_ids:
        raise ValueError(
            f"prediction cache contains {len(unknown_ids)} unknown ids; "
            f"first: {unknown_ids[0]}"
        )
    for example_id, row in existing_rows.items():
        if row["gold"] != expected[example_id].label:
            raise ValueError(f"cached gold label mismatch for {example_id}")

    inference_started = time.perf_counter()
    file_mode = "a" if predictions_path.exists() else "w"
    new_predictions = 0
    scored: list[ScoredPrediction] = []
    with predictions_path.open(file_mode, encoding="utf-8") as handle:
        for example in test:
            if example.id in existing_rows:
                continue
            prediction = classifier.predict(example.text)
            scored_prediction = ScoredPrediction(
                gold=example.label,
                predicted=prediction.label,
                confidence=prediction.confidence,
            )
            scored.append(scored_prediction)
            row = {
                "id": example.id,
                "gold": example.label,
                "predicted": prediction.label,
                "confidence": prediction.confidence,
                "correct": scored_prediction.correct,
                "raw_response": prediction.raw_response,
                "metadata": prediction.metadata,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            new_predictions += 1
    inference_seconds = time.perf_counter() - inference_started

    prediction_rows = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            prediction_rows.append(json.loads(line))
    by_id = {row["id"]: row for row in prediction_rows}
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        raise ValueError(
            f"incomplete prediction cache: {len(missing)} ids missing; "
            f"first: {missing[0] if missing else 'n/a'}"
        )
    scored = [
        ScoredPrediction(
            gold=by_id[example.id]["gold"],
            predicted=by_id[example.id]["predicted"],
            confidence=by_id[example.id]["confidence"],
        )
        for example in test
    ]
    usage_totals: dict[str, float] = {}
    for row in prediction_rows:
        for key, value in (row.get("metadata") or {}).items():
            if isinstance(value, (int, float)):
                usage_totals[key] = usage_totals.get(key, 0) + value

    interval = bootstrap_accuracy_interval(scored, seed=seed)
    report = {
        "schema_version": "0.1",
        "benchmark": "bayanat_labelbench",
        "dataset": dataset_id,
        "task": task,
        "model": classifier.name,
        "model_metadata": getattr(classifier, "model_metadata", None),
        "metrics": {
            "accuracy": accuracy(scored),
            "accuracy_bootstrap_95": {"low": interval[0], "high": interval[1]},
            "macro_f1": macro_f1(scored),
            "per_label": classification_report(scored),
            "confusion_matrix": confusion_matrix(scored),
            "brier_score": brier_score(scored),
            "expected_calibration_error_10": expected_calibration_error(scored),
            "sac_95": safe_automation_coverage(scored, 0.95),
            "sac_98": safe_automation_coverage(scored, 0.98),
            "top_confusions": top_confusions(scored),
        },
        "timing": {
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "new_predictions": new_predictions,
            "resumed_predictions": len(existing_rows),
            "examples_per_second": new_predictions / inference_seconds
            if inference_seconds and new_predictions
            else None,
        },
        "usage": usage_totals,
        "artifacts": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "normalized_train_sha256": (
                sha256_file(train_file) if train_file is not None else None
            ),
            "normalized_test_sha256": sha256_file(test_file),
        },
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "train_examples": len(train),
            "test_examples": len(test),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(),
        },
    }
    report_path = model_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
