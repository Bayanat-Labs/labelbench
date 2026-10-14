import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from labelbench import datasets
from labelbench.datasets import Example, deterministic_sample


class DatasetTests(unittest.TestCase):
    def test_deterministic_sample_is_stable_and_not_a_prefix(self):
        examples = [
            Example(str(index), str(index), "x", {}) for index in range(100)
        ]
        first = deterministic_sample(examples, 10, seed=42)
        second = deterministic_sample(examples, 10, seed=42)
        self.assertEqual([x.id for x in first], [x.id for x in second])
        self.assertNotEqual([x.id for x in first], [str(x) for x in range(10)])

    def test_sample_rejects_oversized_request(self):
        examples = [Example("1", "one", "x", {})]
        with self.assertRaises(ValueError):
            deterministic_sample(examples, 2, seed=42)

    def test_iadd_supervised_manifest_is_disjoint_from_eval(self):
        rows = []
        for region in datasets.IADD_REGIONS:
            for index in range(4):
                rows.append(
                    {
                        "DataSource": "DART",
                        "Region": region,
                        "Country": region,
                        "Sentence": f"{region} نص {index}",
                    }
                )
        with TemporaryDirectory() as temporary:
            cache = Path(temporary) / "normalized"
            raw = Path(temporary) / "raw" / "iadd"
            raw.mkdir(parents=True)
            raw_path = raw / "IADD.json"
            raw_path.write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )
            digest = datasets.sha256_file(raw_path)
            with patch.object(datasets, "IADD_SHA256", digest):
                evaluation, _ = datasets.fetch_iadd_dart_region_eval(
                    cache, per_region=2, seed=4
                )
                train, _ = datasets.fetch_iadd_dart_region_train(
                    cache, per_region=2, eval_per_region=2, seed=4
                )
            self.assertEqual(len(evaluation), 10)
            self.assertEqual(len(train), 10)
            self.assertFalse(
                {example.id for example in evaluation}
                & {example.id for example in train}
            )


if __name__ == "__main__":
    unittest.main()
