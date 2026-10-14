import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_dataset_ids_are_unique_and_approved_sources_are_resolved(self):
        payload = json.loads((ROOT / "registry" / "datasets.json").read_text())
        datasets = payload["datasets"]
        ids = [dataset["id"] for dataset in datasets]
        self.assertEqual(len(ids), len(set(ids)))
        for dataset in datasets:
            if dataset["status"].startswith(("approved", "pilot")):
                self.assertNotEqual(dataset["canonical_source"], "TO_VERIFY")
                self.assertNotEqual(dataset["license"]["data"], "TO_VERIFY")
                self.assertTrue(dataset["license"]["verified_from"])

    def test_model_ids_are_unique(self):
        payload = json.loads((ROOT / "registry" / "models.json").read_text())
        ids = [model["id"] for model in payload["models"]]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
