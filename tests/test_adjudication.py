import json
import tempfile
import unittest
from pathlib import Path

from labelbench.adjudication import (
    aggregate_submissions,
    build_offline_review_bundle,
    create_multilabel_review_tasks,
    validate_submission,
)
from labelbench.datasets import MultiLabelExample


class AdjudicationTests(unittest.TestCase):
    def test_bundle_validation_and_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "tasks.json"
            template_path = root / "template.html"
            bundle_path = root / "bundle.html"
            output_path = root / "aggregate.json"
            template_path.write_text(
                "<script>__TASK_MANIFEST_JSON__</script>",
                encoding="utf-8",
            )
            examples = [
                MultiLabelExample("task-1", "نص", ("cussing",), {}),
            ]
            create_multilabel_review_tasks(examples, manifest_path)
            build_offline_review_bundle(
                manifest_path,
                template_path,
                bundle_path,
            )
            self.assertIn("task-1", bundle_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            submissions = []
            for index, labels in enumerate(
                (["cussing"], ["cussing"], ["not"]),
                start=1,
            ):
                payload = {
                    "schema_version": "0.1",
                    "reviewer": {"id": f"r{index}", "qualification": "Arabic"},
                    "judgments": [
                        {
                            "task_id": "task-1",
                            "labels": labels,
                            "confidence": 3,
                            "flags": [],
                            "notes": "",
                        }
                    ],
                }
                self.assertEqual(validate_submission(payload, manifest), [])
                path = root / f"r{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                submissions.append(path)
            aggregate = aggregate_submissions(
                manifest_path,
                submissions,
                output_path,
            )
            self.assertEqual(
                aggregate["tasks"][0]["consensus_labels"],
                ["cussing"],
            )
            self.assertFalse(aggregate["tasks"][0]["needs_adjudication"])


if __name__ == "__main__":
    unittest.main()
