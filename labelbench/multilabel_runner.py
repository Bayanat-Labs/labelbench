from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .datasets import MultiLabelExample, sha256_file
from .models import MultiLabelPrediction
from .multilabel_metrics import ScoredMultiLabelPrediction, multilabel_report


class MultiLabelClassifier(Protocol):
    name: str

    def fit(self, examples: list[MultiLabelExample]) -> None: ...

    def predict(self, text: str) -> MultiLabelPrediction: ...


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_multilabel_classifier(
    classifier: MultiLabelClassifier,
    train: list[MultiLabelExample],
    test: list[MultiLabelExample],
    labels: tuple[str, ...],
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
            raise FileExistsError(f"predictions already exist: {predictions_path}")
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in existing_rows:
                    raise ValueError(f"duplicate cached prediction id: {row['id']}")
                existing_rows[row["id"]] = row

    expected = {example.id: example for example in test}
    if set(existing_rows) - set(expected):
        raise ValueError("prediction cache contains ids outside the test manifest")
    for example_id, row in existing_rows.items():
        if set(row["gold"]) != set(expected[example_id].labels):
            raise ValueError(f"cached gold label mismatch for {example_id}")

    inference_started = time.perf_counter()
    file_mode = "a" if predictions_path.exists() else "w"
    new_predictions = 0
    with predictions_path.open(file_mode, encoding="utf-8") as handle:
        for example in test:
            if example.id in existing_rows:
                continue
            prediction = classifier.predict(example.text)
            row = {
                "id": example.id,
                "gold": list(example.labels),
                "predicted": list(prediction.labels),
                "confidence": prediction.confidence,
                "correct": set(example.labels) == set(prediction.labels),
                "scores": prediction.scores,
                "raw_response": prediction.raw_response,
                "metadata": prediction.metadata,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            new_predictions += 1
    inference_seconds = time.perf_counter() - inference_started

    rows_by_id = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_id[row["id"]] = row
    if set(rows_by_id) != set(expected):
        raise ValueError("incomplete multi-label prediction cache")

    scored = [
        ScoredMultiLabelPrediction(
            gold=frozenset(rows_by_id[example.id]["gold"]),
            predicted=frozenset(rows_by_id[example.id]["predicted"]),
            confidence=rows_by_id[example.id]["confidence"],
        )
        for example in test
    ]
    usage_totals: dict[str, float] = {}
    for row in rows_by_id.values():
        for key, value in (row.get("metadata") or {}).items():
            if isinstance(value, (int, float)):
                usage_totals[key] = usage_totals.get(key, 0) + value

    report = {
        "schema_version": "0.1",
        "benchmark": "bayanat_labelbench",
        "dataset": dataset_id,
        "task": task,
        "labels": list(labels),
        "model": classifier.name,
        "model_metadata": getattr(classifier, "model_metadata", None),
        "metrics": multilabel_report(scored, labels=labels, seed=seed),
        "timing": {
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "new_predictions": new_predictions,
            "resumed_predictions": len(existing_rows),
            "examples_per_second": (
                new_predictions / inference_seconds
                if inference_seconds and new_predictions
                else None
            ),
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
    (model_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
