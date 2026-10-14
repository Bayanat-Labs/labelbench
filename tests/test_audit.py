import json
import tempfile
import unittest
from pathlib import Path

from labelbench.audit import audit_normalized_splits


class AuditTests(unittest.TestCase):
    def test_detects_normalized_overlap_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            train.write_text(
                json.dumps(
                    {"id": "1", "text": "أهلاً", "label": "greet", "metadata": {}},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            test.write_text(
                json.dumps(
                    {"id": "2", "text": "اهلا", "label": "other", "metadata": {}},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit_normalized_splits(train, test)
            self.assertEqual(
                report["cross_split"]["normalized_overlap_test_rows"], 1
            )
            self.assertEqual(
                report["cross_split"]["normalized_conflicting_label_test_rows"], 1
            )


if __name__ == "__main__":
    unittest.main()
