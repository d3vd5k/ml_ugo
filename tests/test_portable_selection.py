"""Consumer-visible range/exclusion behavior of the embedded notebook runtime."""
from pathlib import Path
import unittest

from reports.synthetic_six_methods import notebook_runtime


class PortableSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = notebook_runtime(Path(__file__).resolve().parents[1] / "UGO_30_Datasets.ipynb")
        cls.benchmark = cls.runtime.__enter__()
        cls.catalogue = [{"number": number, "openml_id": 1000 + number,
                          "name": f"dataset-{number}", "imbalance_type": "low"}
                         for number in range(1, 33)]

    @classmethod
    def tearDownClass(cls):
        cls.runtime.__exit__(None, None, None)

    def test_default_active_selection_keeps_original_ids_and_appended_pair(self):
        config = self.benchmark._configuration({"DEVICE": "cpu"})
        selected = self.benchmark.select_catalogue(self.catalogue, config["DATASET_RANGE"],
            excluded_numbers=config["EXCLUDED_DATASETS"])
        self.assertEqual([row["number"] for row in selected],
                         [number for number in range(1, 33) if number not in (10, 26)])

    def test_partial_range_skips_hole_without_renumbering(self):
        selected = self.benchmark.select_catalogue(self.catalogue, "9-12", excluded_numbers=[10, 26])
        self.assertEqual([row["number"] for row in selected], [9, 11, 12])

    def test_appended_pair_ignores_exclusions_outside_requested_range(self):
        selected = self.benchmark.select_catalogue(self.catalogue, "31-32", excluded_numbers=[10, 26])
        self.assertEqual([row["number"] for row in selected], [31, 32])

    def test_excluded_only_range_fails_and_explicit_opt_in_restores_id(self):
        with self.assertRaisesRegex(ValueError, "No datasets remain"):
            self.benchmark.select_catalogue(self.catalogue, "10-10", excluded_numbers=[10, 26])
        selected = self.benchmark.select_catalogue(self.catalogue, "10-10", excluded_numbers=[])
        self.assertEqual([row["number"] for row in selected], [10])

    def test_invalid_exclusions_do_not_silently_change_selection(self):
        for exclusions in (None, [True], [1.0], [0], [-1], [10, 10], [33]):
            with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                self.benchmark.select_catalogue(self.catalogue, "1-32", excluded_numbers=exclusions)


if __name__ == "__main__":
    unittest.main()
