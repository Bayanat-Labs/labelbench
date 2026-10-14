from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


TASK_TITLES = {
    "binary_toxicity": "Binary toxicity",
    "toxicity_multilabel": "Multi-label toxicity",
    "intent_classification": "Intent classification",
    "dialect_region_identification": "Dialect region identification",
}


def _record(report: dict, source: Path) -> dict:
    metrics = report["metrics"]
    multilabel = "subset_accuracy" in metrics
    primary_name = "subset_accuracy" if multilabel else "accuracy"
    interval_name = (
        "subset_accuracy_bootstrap_95"
        if multilabel
        else "accuracy_bootstrap_95"
    )
    sac_name = "sac_95_exact_set" if multilabel else "sac_95"
    return {
        "dataset": report["dataset"],
        "task": report["task"],
        "task_title": TASK_TITLES.get(report["task"], report["task"]),
        "model": report["model"],
        "test_examples": report["run"]["test_examples"],
        "primary_metric": primary_name,
        "primary_score": metrics[primary_name],
        "ci_95_low": metrics[interval_name]["low"],
        "ci_95_high": metrics[interval_name]["high"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics.get("micro_f1"),
        "sac_95_coverage": metrics[sac_name]["coverage"],
        "test_manifest_sha256": report["artifacts"]["normalized_test_sha256"],
        "predictions_sha256": report["artifacts"]["predictions_sha256"],
        "model_metadata": report.get("model_metadata"),
        "report_source": str(source),
    }


def _write_csv(records: list[dict], output: Path) -> None:
    fields = [
        "dataset",
        "task",
        "model",
        "test_examples",
        "primary_metric",
        "primary_score",
        "ci_95_low",
        "ci_95_high",
        "macro_f1",
        "micro_f1",
        "sac_95_coverage",
        "test_manifest_sha256",
        "predictions_sha256",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _write_markdown(records: list[dict], output: Path) -> None:
    lines = [
        "# LabelBench by Bayanat Labs: results",
        "",
        "Scores are computed on fixed, checksummed manifests. "
        "SAC95 is the fraction that can be accepted while the 95% Wilson "
        "lower confidence bound remains at or above 95% accuracy. "
        "For tied-confidence structured outputs, zero SAC is a routing "
        "diagnostic rather than proof that selective automation is impossible.",
        "",
        "| Task | Model | n | Primary score (95% bootstrap CI) | Macro-F1 | SAC95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        score = 100 * record["primary_score"]
        low = 100 * record["ci_95_low"]
        high = 100 * record["ci_95_high"]
        lines.append(
            f"| {record['task_title']} | `{record['model']}` | "
            f"{record['test_examples']} | {score:.1f}% ({low:.1f}–{high:.1f}) | "
            f"{100 * record['macro_f1']:.1f}% | "
            f"{100 * record['sac_95_coverage']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "> Local-model results are a reproducible launch baseline, not a "
            "claim about untested frontier systems. Evaluation-only source "
            "texts are never redistributed.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _score_chart(records: list[dict], output: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["task_title"], []).append(record)
    row_height = 42
    heading_height = 50
    height = 70 + sum(
        heading_height + row_height * len(rows) for rows in grouped.values()
    )
    width = 1100
    plot_left = 420
    plot_width = 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#17171c}"
        ".title{font-size:24px;font-weight:700}.task{font-size:17px;font-weight:700}"
        ".model{font-size:13px}.score{font-size:13px;font-weight:700}"
        ".axis{stroke:#d9d9e2;stroke-width:1}.bar{fill:#5b4df2}"
        ".ci{stroke:#17171c;stroke-width:2}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text class="title" x="30" y="38">LabelBench by Bayanat Labs: held-out performance</text>',
    ]
    y = 78
    for task, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["primary_score"], reverse=True)
        parts.append(f'<text class="task" x="30" y="{y}">{escape(task)}</text>')
        y += 26
        for tick in (0, 25, 50, 75, 100):
            x = plot_left + plot_width * tick / 100
            parts.append(
                f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" '
                f'y1="{y - 15}" y2="{y + row_height * len(rows) - 8}"/>'
            )
        for record in rows:
            centre = y + 11
            score = 100 * record["primary_score"]
            low = 100 * record["ci_95_low"]
            high = 100 * record["ci_95_high"]
            parts.append(
                f'<text class="model" x="30" y="{centre + 5}">'
                f'{escape(record["model"])}</text>'
            )
            parts.append(
                f'<rect class="bar" x="{plot_left}" y="{centre - 9}" '
                f'width="{plot_width * score / 100:.1f}" height="18" rx="4"/>'
            )
            x_low = plot_left + plot_width * low / 100
            x_high = plot_left + plot_width * high / 100
            parts.append(
                f'<line class="ci" x1="{x_low:.1f}" x2="{x_high:.1f}" '
                f'y1="{centre}" y2="{centre}"/>'
            )
            parts.append(
                f'<text class="score" x="{plot_left + plot_width + 18}" '
                f'y="{centre + 5}">{score:.1f}%</text>'
            )
            y += row_height
        y += heading_height - 26
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _confusion_chart(report: dict, output: Path) -> None:
    confusion = report["metrics"].get("confusion_matrix")
    if not confusion:
        return
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    cell = 92
    left = 150
    top = 110
    size = len(labels)
    width = left + size * cell + 40
    height = top + size * cell + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#17171c}"
        ".title{font-size:21px;font-weight:700}.label{font-size:13px;font-weight:700}"
        ".value{font-size:15px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text class="title" x="25" y="34">Dialect-region confusion matrix</text>',
        '<text x="25" y="60" font-size="13">Rows: gold · columns: predicted · cells: row percentage</text>',
    ]
    for index, label in enumerate(labels):
        x = left + index * cell + cell / 2
        y = top + index * cell + cell / 2
        parts.append(
            f'<text class="label" text-anchor="middle" x="{x}" y="{top - 18}">'
            f"{escape(label)}</text>"
        )
        parts.append(
            f'<text class="label" text-anchor="end" x="{left - 18}" '
            f'y="{y + 5}">{escape(label)}</text>'
        )
    for row_index, row in enumerate(matrix):
        total = max(1, sum(row))
        for column_index, value in enumerate(row):
            fraction = value / total
            alpha = 0.08 + 0.82 * fraction
            x = left + column_index * cell
            y = top + row_index * cell
            text_color = "#fff" if fraction >= 0.55 else "#17171c"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" '
                f'rx="7" fill="#5b4df2" fill-opacity="{alpha:.3f}"/>'
            )
            parts.append(
                f'<text class="value" text-anchor="middle" x="{x + cell / 2}" '
                f'y="{y + cell / 2 + 5}" fill="{text_color}" '
                f'style="fill:{text_color}">{100 * fraction:.0f}%</text>'
            )
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _with_holm_adjusted_pvalues(comparisons: list[dict]) -> list[dict]:
    indexed = [
        (index, comparison.get("mcnemar_exact_two_sided_p"))
        for index, comparison in enumerate(comparisons)
        if comparison.get("mcnemar_exact_two_sided_p") is not None
    ]
    if not indexed:
        return comparisons
    ordered = sorted(indexed, key=lambda item: item[1])
    m = len(ordered)
    adjusted_by_index: dict[int, float] = {}
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running = max(running, adjusted)
        adjusted_by_index[index] = running
    enriched = []
    for index, comparison in enumerate(comparisons):
        row = dict(comparison)
        if index in adjusted_by_index:
            row["mcnemar_holm_adjusted_p"] = adjusted_by_index[index]
        enriched.append(row)
    return enriched


def build_release(
    report_paths: list[Path],
    output_dir: Path,
    release_id: str,
    generated_at: str | None = None,
    comparison_paths: list[Path] | None = None,
) -> dict:
    if not report_paths:
        raise ValueError("at least one report is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    records = [_record(report, path) for report, path in zip(reports, report_paths)]
    records.sort(key=lambda item: (item["task"], -item["primary_score"], item["model"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.1",
        "benchmark": "bayanat_labelbench",
        "release_id": release_id,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "result_count": len(records),
        "results": records,
        "paired_comparisons": _with_holm_adjusted_pvalues(
            [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (comparison_paths or [])
            ]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(records, output_dir / "leaderboard.csv")
    _write_markdown(records, output_dir / "README.md")
    _score_chart(records, output_dir / "task-performance.svg")
    for report in reports:
        if report["task"] == "dialect_region_identification":
            _confusion_chart(report, output_dir / "dialect-confusion.svg")
            break
    return payload
