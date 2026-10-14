from __future__ import annotations

import html
import json
import random
from collections import Counter
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterable

from .datasets import MultiLabelExample


TOXICITY_DEFINITIONS = {
    "appearance": "Attack on appearance, body, age, or physical traits",
    "cussing": "Swearing, profanity, or a direct insult",
    "hatred": "Hate, dehumanization, or severe group/person hostility",
    "racial": "Attack based on race, ethnicity, nationality, or origin",
    "sexual": "Sexual insult, sexual degradation, or sexual abuse",
    "violence": "Threat, encouragement, or celebration of physical harm",
    "not": "None of the toxic categories apply",
}


def create_multilabel_review_tasks(
    examples: Iterable[MultiLabelExample],
    output_path: Path,
    seed: int = 42,
) -> list[dict]:
    tasks = [
        {
            "id": example.id,
            "task_type": "multilabel",
            "text": example.text,
            "labels": [
                {"id": label, "definition": definition}
                for label, definition in TOXICITY_DEFINITIONS.items()
            ],
            "exclusive_label": "not",
        }
        for example in examples
    ]
    random.Random(seed).shuffle(tasks)
    manifest = {
        "schema_version": "0.1",
        "benchmark": "bayanat_labelbench",
        "task": "aratox_multilabel_readjudication",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instructions": (
            "Label every category supported by the text. Select 'not' only "
            "when none of the six toxic categories applies."
        ),
        "tasks": tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tasks


def default_review_template() -> str:
    """Return the blinded review template that ships inside the package."""
    return (
        resources.files("labelbench")
        .joinpath("templates")
        .joinpath("review_template.html")
        .read_text(encoding="utf-8")
    )


def build_offline_review_bundle(
    task_manifest_path: Path,
    template_path: Path | None,
    output_path: Path,
) -> None:
    """Embed a task manifest in the review template.

    Pass ``template_path=None`` to use the template that ships with the package.
    """
    manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    if template_path is None:
        template = default_review_template()
    else:
        template = template_path.read_text(encoding="utf-8")
    embedded = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    rendered = template.replace("__TASK_MANIFEST_JSON__", embedded)
    if rendered == template:
        raise ValueError("review template placeholder was not found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def validate_submission(
    submission: dict,
    task_manifest: dict,
) -> list[str]:
    errors: list[str] = []
    if submission.get("schema_version") != "0.1":
        errors.append("unsupported submission schema")
    reviewer = submission.get("reviewer") or {}
    if not reviewer.get("id"):
        errors.append("reviewer id is required")
    tasks_by_id = {task["id"]: task for task in task_manifest["tasks"]}
    judgments = submission.get("judgments")
    if not isinstance(judgments, list):
        return errors + ["judgments must be a list"]
    seen = set()
    for judgment in judgments:
        task_id = judgment.get("task_id")
        if task_id in seen:
            errors.append(f"duplicate judgment for {task_id}")
            continue
        seen.add(task_id)
        task = tasks_by_id.get(task_id)
        if task is None:
            errors.append(f"unknown task id {task_id}")
            continue
        allowed = {label["id"] for label in task["labels"]}
        selected = judgment.get("labels")
        if (
            not isinstance(selected, list)
            or not selected
            or any(label not in allowed for label in selected)
        ):
            errors.append(f"invalid labels for {task_id}")
            continue
        if len(selected) != len(set(selected)):
            errors.append(f"duplicate labels for {task_id}")
        exclusive = task.get("exclusive_label")
        if exclusive in selected and len(selected) > 1:
            errors.append(f"exclusive label combined with others for {task_id}")
        if judgment.get("confidence") not in {1, 2, 3}:
            errors.append(f"invalid confidence for {task_id}")
    return errors


def aggregate_submissions(
    task_manifest_path: Path,
    submission_paths: list[Path],
    output_path: Path,
) -> dict:
    manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    by_task: dict[str, list[dict]] = {
        task["id"]: [] for task in manifest["tasks"]
    }
    reviewers = []
    for path in submission_paths:
        submission = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_submission(submission, manifest)
        if errors:
            raise ValueError(f"{path}: " + "; ".join(errors))
        reviewer_id = submission["reviewer"]["id"]
        if reviewer_id in reviewers:
            raise ValueError(f"duplicate reviewer id: {reviewer_id}")
        reviewers.append(reviewer_id)
        for judgment in submission["judgments"]:
            by_task[judgment["task_id"]].append(
                {
                    **judgment,
                    "reviewer_id": reviewer_id,
                }
            )

    results = []
    for task_id, judgments in by_task.items():
        votes = Counter(
            tuple(sorted(judgment["labels"])) for judgment in judgments
        )
        winning_labels: tuple[str, ...] | None = None
        winning_votes = 0
        if votes:
            winning_labels, winning_votes = votes.most_common(1)[0]
        majority_required = len(judgments) // 2 + 1 if judgments else 1
        consensus = (
            list(winning_labels)
            if winning_labels is not None and winning_votes >= majority_required
            else None
        )
        results.append(
            {
                "task_id": task_id,
                "judgment_count": len(judgments),
                "consensus_labels": consensus,
                "consensus_votes": winning_votes,
                "needs_adjudication": consensus is None,
                "vote_distribution": [
                    {"labels": list(labels), "count": count}
                    for labels, count in votes.most_common()
                ],
            }
        )

    aggregate = {
        "schema_version": "0.1",
        "benchmark": "bayanat_labelbench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewers": reviewers,
        "tasks": results,
        "summary": {
            "task_count": len(results),
            "fully_reviewed": sum(
                len(by_task[result["task_id"]]) == len(reviewers)
                for result in results
            ),
            "consensus_count": sum(
                result["consensus_labels"] is not None for result in results
            ),
            "needs_adjudication": sum(
                result["needs_adjudication"] for result in results
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate
