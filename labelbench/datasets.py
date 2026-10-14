from __future__ import annotations

import hashlib
import csv
import json
import random
import urllib.request
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


MASSIVE_ARCHIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)
MASSIVE_ARCHIVE_SHA256 = (
    "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
)
MASSIVE_AR_SA_MEMBER = "1.1/data/ar-SA.jsonl"
ARATOX_REVISION = "438f12260480318b976b15b23212ed3a661c088f"
ARATOX_FILES = {
    "train": {
        "sha256": "0a10c60fa15b15dd0a73855d8a46d0beed4db3a3fe1daebf5594c052f35d4e3e",
        "url": (
            "https://huggingface.co/datasets/Alshargi/Aratox/resolve/"
            f"{ARATOX_REVISION}/aratox_train.csv"
        ),
    },
    "test": {
        "sha256": "27dc4b80ffe79f46a4d2abe3aad362c3273827c68dfcc581ab4341ecd7636a7c",
        "url": (
            "https://huggingface.co/datasets/Alshargi/Aratox/resolve/"
            f"{ARATOX_REVISION}/aratox_test.csv"
        ),
    },
}
ARATOX_TOXIC_LABELS = (
    "appearance",
    "cussing",
    "hatred",
    "racial",
    "sexual",
    "violence",
)
ARATOX_LABELS = ARATOX_TOXIC_LABELS + ("not",)
IADD_REVISION = "340b5a6135f70878b95481edcaae4e3afb2067ac"
IADD_URL = (
    "https://raw.githubusercontent.com/JihadZa/IADD/"
    f"{IADD_REVISION}/IADD.json"
)
IADD_SHA256 = "a28c106d1825debabb6b16979c6c6ca66abdfd0d7802fd899b5ec4d7aee0f734"
IADD_REGIONS = ("GLF", "EGY", "LEV", "MGH", "IRQ")


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MultiLabelExample:
    id: str
    text: str
    labels: tuple[str, ...]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "labels": list(self.labels),
            "metadata": self.metadata,
        }


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Bayanat-Labs-LabelBench/0.1 "
                "(+https://github.com/Bayanat-Labs/labelbench; hello@bayanatlabs.com)"
            )
        },
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    temporary.replace(destination)


def _ensure_massive_ar_sa(cache_dir: Path) -> tuple[Path, Path]:
    data_dir = cache_dir.parent
    raw_dir = data_dir / "raw"
    archive_path = raw_dir / "amazon-massive-dataset-1.1.tar.gz"
    source_dir = raw_dir / "massive-1.1"
    source_path = source_dir / "ar-SA.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        _download(MASSIVE_ARCHIVE_URL, archive_path)
    archive_hash = sha256_file(archive_path)
    if archive_hash != MASSIVE_ARCHIVE_SHA256:
        raise ValueError(
            "MASSIVE archive checksum mismatch: "
            f"expected {MASSIVE_ARCHIVE_SHA256}, received {archive_hash}"
        )

    if not source_path.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        temporary = source_path.with_suffix(".jsonl.tmp")
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember(MASSIVE_AR_SA_MEMBER)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member is not a file: {MASSIVE_AR_SA_MEMBER}")
            with temporary.open("wb") as handle:
                while chunk := extracted.read(1024 * 1024):
                    handle.write(chunk)
        temporary.replace(source_path)
    return source_path, archive_path


def fetch_massive_ar_sa(
    split: str,
    limit: int,
    cache_dir: Path,
    page_size: int = 100,
) -> tuple[list[Example], Path]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported MASSIVE split: {split}")
    if limit <= 0:
        raise ValueError("limit must be positive")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"massive_ar_sa_{split}_{limit}.jsonl"
    if cache_path.exists():
        return list(read_examples(cache_path)), cache_path

    source_path, _ = _ensure_massive_ar_sa(cache_dir)
    examples: list[Example] = []
    del page_size
    with source_path.open(encoding="utf-8") as handle:
        for source_row_index, line in enumerate(handle):
            row = json.loads(line)
            if row["partition"] != split:
                continue
            examples.append(
                Example(
                    id=f"massive_ar_sa:{split}:{row['id']}",
                    text=row["utt"],
                    label=row["intent"],
                    metadata={
                        "dataset": "massive_ar_sa",
                        "split": split,
                        "locale": row["locale"],
                        "scenario": row["scenario"],
                        "source_row_index": source_row_index,
                    },
                )
            )
            if len(examples) == limit:
                break

    if len(examples) < limit:
        raise ValueError(
            f"requested {limit} MASSIVE {split} examples but received {len(examples)}"
        )
    write_examples(cache_path, examples)
    return examples, cache_path


def fetch_aratox_binary(
    split: str,
    limit: int | None,
    cache_dir: Path,
) -> tuple[list[Example], Path]:
    if split not in ARATOX_FILES:
        raise ValueError(f"unsupported AraTox split: {split}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or omitted")

    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_name = str(limit) if limit is not None else "all"
    cache_path = cache_dir / f"aratox_binary_{split}_{limit_name}.jsonl"
    if cache_path.exists():
        return list(read_examples(cache_path)), cache_path

    raw_dir = cache_dir.parent / "raw" / "aratox"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"aratox_{split}.csv"
    file_spec = ARATOX_FILES[split]
    if not raw_path.exists():
        _download(file_spec["url"], raw_path)
    actual_hash = sha256_file(raw_path)
    if actual_hash != file_spec["sha256"]:
        raise ValueError(
            f"AraTox {split} checksum mismatch: "
            f"expected {file_spec['sha256']}, received {actual_hash}"
        )

    examples: list[Example] = []
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        for source_row_index, raw_row in enumerate(csv.DictReader(handle)):
            row = {key.lower(): value for key, value in raw_row.items()}
            active_labels = [
                label for label in ARATOX_TOXIC_LABELS if int(row[label]) == 1
            ]
            binary_label = "toxic" if active_labels else "not_toxic"
            examples.append(
                Example(
                    id=f"aratox:{split}:{source_row_index}",
                    text=row["text"],
                    label=binary_label,
                    metadata={
                        "dataset": "aratox",
                        "split": split,
                        "source_row_index": source_row_index,
                        "active_toxicity_labels": active_labels,
                        "released_not_label": int(row["not"]),
                    },
                )
            )
            if limit is not None and len(examples) == limit:
                break
    if limit is not None and len(examples) < limit:
        raise ValueError(
            f"requested {limit} AraTox {split} examples but received {len(examples)}"
        )
    write_examples(cache_path, examples)
    return examples, cache_path


def fetch_aratox_multilabel(
    split: str,
    limit: int | None,
    cache_dir: Path,
) -> tuple[list[MultiLabelExample], Path]:
    if split not in ARATOX_FILES:
        raise ValueError(f"unsupported AraTox split: {split}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or omitted")

    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_name = str(limit) if limit is not None else "all"
    cache_path = cache_dir / f"aratox_multilabel_{split}_{limit_name}.jsonl"
    if cache_path.exists():
        return list(read_multilabel_examples(cache_path)), cache_path

    raw_dir = cache_dir.parent / "raw" / "aratox"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"aratox_{split}.csv"
    file_spec = ARATOX_FILES[split]
    if not raw_path.exists():
        _download(file_spec["url"], raw_path)
    actual_hash = sha256_file(raw_path)
    if actual_hash != file_spec["sha256"]:
        raise ValueError(
            f"AraTox {split} checksum mismatch: "
            f"expected {file_spec['sha256']}, received {actual_hash}"
        )

    examples: list[MultiLabelExample] = []
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        for source_row_index, raw_row in enumerate(csv.DictReader(handle)):
            row = {key.lower(): value for key, value in raw_row.items()}
            labels = tuple(label for label in ARATOX_LABELS if int(row[label]) == 1)
            has_toxic = any(label in ARATOX_TOXIC_LABELS for label in labels)
            if not labels or (has_toxic and "not" in labels):
                raise ValueError(
                    f"invalid AraTox label combination at {split}:{source_row_index}"
                )
            examples.append(
                MultiLabelExample(
                    id=f"aratox:{split}:{source_row_index}",
                    text=row["text"],
                    labels=labels,
                    metadata={
                        "dataset": "aratox",
                        "split": split,
                        "source_row_index": source_row_index,
                        "label_cardinality": len(labels),
                    },
                )
            )
            if limit is not None and len(examples) == limit:
                break
    if limit is not None and len(examples) < limit:
        raise ValueError(
            f"requested {limit} AraTox {split} examples but received {len(examples)}"
        )
    write_multilabel_examples(cache_path, examples)
    return examples, cache_path


def fetch_iadd_dart_region_eval(
    cache_dir: Path,
    per_region: int = 200,
    seed: int = 42,
) -> tuple[list[Example], Path]:
    if per_region <= 0:
        raise ValueError("per_region must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"iadd_dart_region_eval_{per_region}per_seed{seed}.jsonl"
    )
    if cache_path.exists():
        return list(read_examples(cache_path)), cache_path

    raw_dir = cache_dir.parent / "raw" / "iadd"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "IADD.json"
    if not raw_path.exists():
        _download(IADD_URL, raw_path)
    actual_hash = sha256_file(raw_path)
    if actual_hash != IADD_SHA256:
        raise ValueError(
            f"IADD checksum mismatch: expected {IADD_SHA256}, received {actual_hash}"
        )

    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    normalized_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["DataSource"] != "DART" or row["Region"] not in IADD_REGIONS:
            continue
        normalized = _basic_arabic_normalize(row["Sentence"])
        if not normalized:
            continue
        normalized_groups.setdefault(normalized, []).append(row)

    clean_by_region: dict[str, list[dict[str, Any]]] = {
        region: [] for region in IADD_REGIONS
    }
    for normalized, group in normalized_groups.items():
        regions = {row["Region"] for row in group}
        if len(regions) != 1:
            continue
        row = group[0]
        row = {**row, "_normalized": normalized}
        clean_by_region[row["Region"]].append(row)

    generator = random.Random(seed)
    examples: list[Example] = []
    for region in IADD_REGIONS:
        pool = clean_by_region[region]
        if len(pool) < per_region:
            raise ValueError(
                f"IADD DART has only {len(pool)} clean {region} rows; "
                f"requested {per_region}"
            )
        selected = generator.sample(pool, per_region)
        selected.sort(key=lambda row: row["_normalized"])
        for index, row in enumerate(selected):
            text_hash = hashlib.sha256(row["_normalized"].encode("utf-8")).hexdigest()
            examples.append(
                Example(
                    id=f"iadd_dart:{region}:{text_hash[:16]}",
                    text=row["Sentence"],
                    label=region,
                    metadata={
                        "dataset": "iadd_dart_region",
                        "source": "DART",
                        "region": region,
                        "country": row["Country"],
                        "within_region_index": index,
                        "text_sha256": text_hash,
                        "redistribution": "aggregate_only",
                    },
                )
            )
    examples.sort(key=lambda example: example.id)
    write_examples(cache_path, examples)
    return examples, cache_path


def fetch_iadd_dart_region_train(
    cache_dir: Path,
    per_region: int = 100,
    eval_per_region: int = 100,
    seed: int = 42,
) -> tuple[list[Example], Path]:
    """Build a balanced diagnostic train set disjoint from the fixed eval set."""
    if per_region <= 0:
        raise ValueError("per_region must be positive")
    evaluation, _ = fetch_iadd_dart_region_eval(
        cache_dir=cache_dir,
        per_region=eval_per_region,
        seed=seed,
    )
    held_out_ids = {example.id for example in evaluation}
    cache_path = (
        cache_dir
        / f"iadd_dart_region_train_{per_region}per_eval"
        f"{eval_per_region}per_seed{seed}.jsonl"
    )
    if cache_path.exists():
        examples = list(read_examples(cache_path))
        if held_out_ids & {example.id for example in examples}:
            raise ValueError("cached IADD train/eval manifests overlap")
        return examples, cache_path

    raw_path = cache_dir.parent / "raw" / "iadd" / "IADD.json"
    actual_hash = sha256_file(raw_path)
    if actual_hash != IADD_SHA256:
        raise ValueError(
            f"IADD checksum mismatch: expected {IADD_SHA256}, received {actual_hash}"
        )
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    normalized_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["DataSource"] != "DART" or row["Region"] not in IADD_REGIONS:
            continue
        normalized = _basic_arabic_normalize(row["Sentence"])
        if normalized:
            normalized_groups.setdefault(normalized, []).append(row)

    clean_by_region: dict[str, list[dict[str, Any]]] = {
        region: [] for region in IADD_REGIONS
    }
    for normalized, group in normalized_groups.items():
        if len({row["Region"] for row in group}) != 1:
            continue
        row = {**group[0], "_normalized": normalized}
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        example_id = f"iadd_dart:{row['Region']}:{text_hash[:16]}"
        if example_id not in held_out_ids:
            clean_by_region[row["Region"]].append(row)

    generator = random.Random(seed + 1)
    examples: list[Example] = []
    for region in IADD_REGIONS:
        pool = clean_by_region[region]
        if len(pool) < per_region:
            raise ValueError(
                f"IADD DART has only {len(pool)} clean non-eval {region} rows; "
                f"requested {per_region}"
            )
        for row in generator.sample(pool, per_region):
            text_hash = hashlib.sha256(
                row["_normalized"].encode("utf-8")
            ).hexdigest()
            examples.append(
                Example(
                    id=f"iadd_dart:{region}:{text_hash[:16]}",
                    text=row["Sentence"],
                    label=region,
                    metadata={
                        "dataset": "iadd_dart_region",
                        "source": "DART",
                        "region": region,
                        "country": row["Country"],
                        "text_sha256": text_hash,
                        "redistribution": "aggregate_only",
                    },
                )
            )
    examples.sort(key=lambda example: example.id)
    if held_out_ids & {example.id for example in examples}:
        raise AssertionError("IADD train/eval manifests overlap")
    write_examples(cache_path, examples)
    return examples, cache_path


def write_examples(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_multilabel_examples(
    path: Path,
    examples: list[MultiLabelExample],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def deterministic_sample(
    examples: list[Example],
    limit: int,
    seed: int,
) -> list[Example]:
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    if limit > len(examples):
        raise ValueError(
            f"cannot sample {limit} examples from a population of {len(examples)}"
        )
    generator = random.Random(seed)
    selected_indices = sorted(generator.sample(range(len(examples)), limit))
    return [examples[index] for index in selected_indices]


def _basic_arabic_normalize(text: str) -> str:
    diacritics = {
        chr(code)
        for start, end in ((0x0610, 0x061A), (0x064B, 0x065F), (0x06D6, 0x06ED))
        for code in range(start, end + 1)
    }
    normalized = "".join(
        character for character in text if character not in diacritics
    )
    normalized = (
        normalized.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ـ", "")
    )
    return " ".join(normalized.strip().lower().split())


def read_examples(path: Path) -> Iterator[Example]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield Example(
                id=row["id"],
                text=row["text"],
                label=row["label"],
                metadata=row.get("metadata", {}),
            )


def read_multilabel_examples(path: Path) -> Iterator[MultiLabelExample]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield MultiLabelExample(
                id=row["id"],
                text=row["text"],
                labels=tuple(row["labels"]),
                metadata=row.get("metadata", {}),
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
