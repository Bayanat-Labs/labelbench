import json
import tempfile
import unittest
from pathlib import Path

from labelbench.comparisons import compare_prediction_files


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison_and_mcnemar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "a.jsonl", root / "b.jsonl"
            a_rows = [
                {"id": "1", "gold": "x", "correct": True},
                {"id": "2", "gold": "x", "correct": True},
                {"id": "3", "gold": "x", "correct": False},
                {"id": "4", "gold": "x", "correct": False},
            ]
            b_rows = [
                {"id": "1", "gold": "x", "correct": True},
                {"id": "2", "gold": "x", "correct": False},
                {"id": "3", "gold": "x", "correct": True},
                {"id": "4", "gold": "x", "correct": False},
            ]
            for path, rows in ((a, a_rows), (b, b_rows)):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            report = compare_prediction_files(a, b, iterations=100, seed=1)
            self.assertEqual(report["accuracy_difference_a_minus_b"], 0.0)
            self.assertEqual(
                report["contingency"],
                {
                    "both_correct": 1,
                    "only_a_correct": 1,
                    "only_b_correct": 1,
                    "neither_correct": 1,
                },
            )
            self.assertEqual(report["mcnemar_exact_two_sided_p"], 1.0)

    def test_mismatched_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "a.jsonl", root / "b.jsonl"
            a.write_text('{"id":"1","gold":"x","correct":true}\n')
            b.write_text('{"id":"2","gold":"x","correct":true}\n')
            with self.assertRaises(ValueError):
                compare_prediction_files(a, b, iterations=10)


if __name__ == "__main__":
    unittest.main()
