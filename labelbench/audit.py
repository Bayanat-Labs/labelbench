from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .datasets import Example, read_examples
from .models import normalize_arabic


def _index(
    examples: list[Example],
    normalizer,
) -> dict[str, list[Example]]:
    index: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        index[normalizer(example.text)].append(example)
    return index


def audit_normalized_splits(train_path: Path, test_path: Path) -> dict:
    train = list(read_examples(train_path))
    test = list(read_examples(test_path))
    train_raw = _index(train, lambda text: text)
    test_raw = _index(test, lambda text: text)
    train_normalized = _index(train, normalize_arabic)
    test_normalized = _index(test, normalize_arabic)

    exact_overlap_rows = sum(
        len(test_rows) for text, test_rows in test_raw.items() if text in train_raw
    )
    normalized_overlap_rows = sum(
        len(test_rows)
        for text, test_rows in test_normalized.items()
        if text in train_normalized
    )
    cross_split_conflicts = sum(
        len(test_rows)
        for text, test_rows in test_normalized.items()
        if text in train_normalized
        and {
            example.label
            for example in train_normalized[text] + test_rows
        }
        and len(
            {
                example.label
                for example in train_normalized[text] + test_rows
            }
        )
        > 1
    )

    def split_report(
        examples: list[Example],
        raw_index: dict[str, list[Example]],
        normalized_index: dict[str, list[Example]],
    ) -> dict:
        conflicting_rows = sum(
            len(rows)
            for rows in normalized_index.values()
            if len({example.label for example in rows}) > 1
        )
        return {
            "rows": len(examples),
            "labels": dict(sorted(Counter(x.label for x in examples).items())),
            "blank_text_rows": sum(not normalize_arabic(x.text) for x in examples),
            "exact_duplicate_rows": len(examples) - len(raw_index),
            "normalized_duplicate_rows": len(examples) - len(normalized_index),
            "normalized_conflicting_label_rows": conflicting_rows,
        }

    return {
        "schema_version": "0.1",
        "train": split_report(train, train_raw, train_normalized),
        "test": split_report(test, test_raw, test_normalized),
        "cross_split": {
            "exact_overlap_test_rows": exact_overlap_rows,
            "exact_overlap_rate": exact_overlap_rows / len(test) if test else 0.0,
            "normalized_overlap_test_rows": normalized_overlap_rows,
            "normalized_overlap_rate": normalized_overlap_rows / len(test)
            if test
            else 0.0,
            "normalized_conflicting_label_test_rows": cross_split_conflicts,
        },
    }
