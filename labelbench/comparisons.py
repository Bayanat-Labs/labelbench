from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate id at {path}:{line_number}")
            rows[row["id"]] = row
    return rows


def _mcnemar_exact_p(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    smaller = min(only_a, only_b)
    one_tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2 * one_tail)


def compare_prediction_files(
    a_path: Path,
    b_path: Path,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    a_rows = _read(a_path)
    b_rows = _read(b_path)
    if set(a_rows) != set(b_rows):
        only_a = sorted(set(a_rows) - set(b_rows))
        only_b = sorted(set(b_rows) - set(a_rows))
        raise ValueError(
            f"prediction IDs differ: {len(only_a)} only in A, "
            f"{len(only_b)} only in B"
        )
    ids = sorted(a_rows)
    paired: list[tuple[int, int]] = []
    for item_id in ids:
        a_row, b_row = a_rows[item_id], b_rows[item_id]
        if a_row["gold"] != b_row["gold"]:
            raise ValueError(f"gold labels differ for {item_id}")
        paired.append((int(a_row["correct"]), int(b_row["correct"])))
    both = sum(a and b for a, b in paired)
    only_a = sum(a and not b for a, b in paired)
    only_b = sum(b and not a for a, b in paired)
    neither = len(paired) - both - only_a - only_b
    differences = [a - b for a, b in paired]
    observed = sum(differences) / len(differences) if differences else 0.0
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(iterations):
        bootstrapped.append(
            sum(differences[generator.randrange(len(differences))]
                for _ in differences)
            / len(differences)
            if differences
            else 0.0
        )
    bootstrapped.sort()
    low = bootstrapped[int(0.025 * iterations)]
    high = bootstrapped[min(iterations - 1, int(0.975 * iterations))]
    return {
        "schema_version": "0.1",
        "paired_examples": len(paired),
        "a": {
            "predictions": str(a_path),
            "sha256": _sha256(a_path),
            "accuracy": (both + only_a) / len(paired) if paired else 0.0,
        },
        "b": {
            "predictions": str(b_path),
            "sha256": _sha256(b_path),
            "accuracy": (both + only_b) / len(paired) if paired else 0.0,
        },
        "accuracy_difference_a_minus_b": observed,
        "paired_bootstrap_95": {"low": low, "high": high},
        "contingency": {
            "both_correct": both,
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "neither_correct": neither,
        },
        "mcnemar_exact_two_sided_p": _mcnemar_exact_p(only_a, only_b),
        "bootstrap_iterations": iterations,
        "seed": seed,
    }
