import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from benchmarks.protocols import (_archive, _csv, _dataset, _member, load_dataset, load_protocol,
                                  make_splits, validate_dataset, validate_protocol)

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def protocol(self, name="shoppers"):
        return load_protocol(ROOT / "configs" / "benchmarks" / f"{name}_controlled.yaml")

    def dataset(self, n=200, name="shoppers"):
        return {"name": name, "X": pd.DataFrame({"number": np.arange(n)}),
                "y": np.tile(np.array([0, 0, 0, 1], dtype=np.int64), n // 4),
                "row_ids": np.array([f"source:{i}" for i in range(n)], dtype=object), "metadata": {}}

    def test_outer_holdout_is_fixed_across_training_seeds(self):
        data, protocol = self.dataset(), self.protocol()
        first = make_splits(data, protocol, 0)[0]
        second = make_splits(data, protocol, 1)[0]
        np.testing.assert_array_equal(first["test_indices"], second["test_indices"])
        self.assertFalse(np.array_equal(first["validation_indices"], second["validation_indices"]))
        for partition in (first, second):
            self.assertFalse(set(partition["test_indices"]) & set(partition["fit_indices"]))
            self.assertFalse(set(partition["test_indices"]) & set(partition["validation_indices"]))
            self.assertEqual(set(partition["train_indices"]),
                             set(partition["fit_indices"]) | set(partition["validation_indices"]))
            self.assertEqual(np.mean(data["y"][partition["test_indices"]]), .25)
            for key in partition:
                if key.endswith("indices"):
                    self.assertEqual(partition[key].dtype, np.int64)

    def test_official_test_is_never_used_for_validation(self):
        data, protocol = self.dataset(name="aps"), self.protocol("aps")
        data["official_train_indices"] = np.arange(160)
        data["official_test_indices"] = np.arange(160, 200)
        partition = make_splits(data, protocol, 9)[0]
        np.testing.assert_array_equal(partition["test_indices"], np.arange(160, 200))
        self.assertTrue(np.all(partition["fit_indices"] < 160))
        self.assertTrue(np.all(partition["validation_indices"] < 160))

    def test_baf_uses_month_five_not_random_validation(self):
        data, protocol = self.dataset(32, name="baf_base"), self.protocol("baf_base")
        data["time_values"] = np.repeat(np.arange(8), 4)
        data["official_train_indices"] = np.arange(24)
        data["official_test_indices"] = np.arange(24, 32)
        for seed in (0, 91):
            partition = make_splits(data, protocol, seed)[0]
            np.testing.assert_array_equal(partition["fit_indices"], np.arange(20))
            np.testing.assert_array_equal(partition["validation_indices"], np.arange(20, 24))
            np.testing.assert_array_equal(partition["test_indices"], np.arange(24, 32))
        data["time_values"] += 1
        with self.assertRaisesRegex(ValueError, "zero-based"):
            make_splits(data, protocol, 0)

    def test_dataset_identity_and_target_contracts_fail_closed(self):
        for change in (lambda data: data["row_ids"].__setitem__(0, None),
                       lambda data: data["row_ids"].__setitem__(0, data["row_ids"][1]),
                       lambda data: data["row_ids"].__setitem__(0, ""),
                       lambda data: data.__setitem__("y", data["y"].astype(np.int32)),
                       lambda data: data.__setitem__("y", data["y"] + 1)):
            data = self.dataset()
            change(data)
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_invalid_official_indices_cannot_cross_boundary(self):
        for train, test in ((np.arange(160), np.arange(159, 200)),
                            (np.r_[np.arange(159), 0], np.arange(160, 200)),
                            (np.arange(159), np.arange(160, 200)),
                            (np.arange(160), np.arange(160, 201))):
            data = self.dataset(name="aps")
            data["official_train_indices"], data["official_test_indices"] = train, test
            with self.assertRaises(ValueError):
                make_splits(data, self.protocol("aps"), 0)

    def test_unresolved_author_protocols_refuse_execution(self):
        for name in ("shoppers", "aps", "baf_base", "satimage"):
            protocol = load_protocol(ROOT / "configs" / "benchmarks" / f"{name}_published.yaml")
            self.assertEqual(protocol["status"], "unresolved")
            self.assertTrue(protocol["missing_prerequisites"])
            with self.assertRaisesRegex(ValueError, "unresolved"):
                make_splits(self.dataset(), protocol, 0)

    def test_config_validation_rejects_future_methods_and_fake_budgets(self):
        changes = {"methods": ["baseline", "U1"], "seeds": [0, 0],
                   "threshold": {"name": "test_optimal"}, "validation_fraction": 0,
                   "tuning_budget": {"n_candidates": 0}, "status": "unknown"}
        for key, value in changes.items():
            protocol = self.protocol()
            protocol[key] = value
            with self.assertRaises(ValueError):
                validate_protocol(protocol)
        protocol = self.protocol("baf_base")
        protocol["split"]["validation_months"] = [6]
        with self.assertRaisesRegex(ValueError, "validation5"):
            validate_protocol(protocol)

    def test_baf_has_one_hundred_explicit_official_space_candidates(self):
        protocol = self.protocol("baf_base")
        candidates = protocol["candidate_parameters"]["lightgbm"]
        self.assertEqual(len(candidates), 100)
        for candidate in candidates:
            self.assertTrue(20 <= candidate["n_estimators"] <= 10000)
            self.assertTrue(3 <= candidate["max_depth"] <= 30)
            self.assertTrue(.02 <= candidate["learning_rate"] <= .1)
            self.assertTrue(10 <= candidate["num_leaves"] <= 100)
            self.assertTrue(5 <= candidate["min_data_in_leaf"] <= 200)
            self.assertTrue(100 <= candidate["max_bin"] <= 500)
            self.assertIn(candidate["boosting_type"], {"gbdt", "goss"})
            self.assertIsInstance(candidate["enable_bundle"], bool)
        bad = copy.deepcopy(protocol)
        bad["candidate_parameters"]["lightgbm"].pop()
        with self.assertRaisesRegex(ValueError, "exactly n_candidates"):
            validate_protocol(bad)

    def test_archive_hash_and_member_hash_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data.csv", "number,label\n1,0\n")
            spec = {"archive": path.name, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(_archive(spec, directory, False), path)
            expected = hashlib.sha256(b"number,label\n1,0\n").hexdigest()
            self.assertEqual(_member(path, "data.csv", expected).read(), b"number,label\n1,0\n")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                _member(path, "data.csv", "incorrect")
            spec["archive_sha256"] = "incorrect"
            with self.assertRaisesRegex(ValueError, "checksum"):
                _archive(spec, directory, False)
            path.unlink()
            with self.assertRaises(FileNotFoundError), patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
                _archive(spec, directory, False)

    def test_duplicate_csv_headers_are_not_silently_mangled(self):
        with self.assertRaisesRegex(ValueError, "duplicate CSV"):
            _csv(io.BytesIO(b"number,number,label\n1,2,0\n"))

    def test_missing_kaggle_base_never_attempts_network_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            with self.assertRaisesRegex(FileNotFoundError, "no credentials"):
                load_dataset(self.protocol("baf_base"), directory, download=True)

    def test_schema_missing_sentinel_and_target_mapping_are_stateless(self):
        frame = pd.DataFrame({"fraud_bool": [0, 1, 0, 1], "bank_months_count": [-1, 2, 3, 4],
                              "payment_type": ["AA", "BB", "AA", "BB"], "month": [0, 0, 1, 1]})
        spec = {"columns": list(frame), "rows": 4, "class_counts": {0: 2, 1: 2},
                "numeric_features": ["bank_months_count", "month"], "missing_values": {"bank_months_count": -1},
                "source": "unit-fixture", "version": "fixture", "license": "fixture",
                "entity_fields": [], "time_fields": ["month"], "climb_overlap": {"status": "fixture"}}
        data = _dataset("baf_base", [frame], spec, {}, "fraud_bool", {0: 0, 1: 1}, ["payment_type"])
        self.assertTrue(pd.isna(data["X"].loc[0, "bank_months_count"]))
        self.assertEqual(data["metadata"]["missingness"]["bank_months_count"], 1)
        self.assertIsInstance(data["X"]["payment_type"].dtype, pd.CategoricalDtype)
        np.testing.assert_array_equal(data["y"], np.array([0, 1, 0, 1], dtype=np.int64))
        self.assertEqual(frame.loc[0, "bank_months_count"], -1)
        malformed = frame.copy()
        malformed["bank_months_count"] = "not numeric"
        with self.assertRaises(ValueError):
            _dataset("baf_base", [malformed], spec, {}, "fraud_bool", {0: 0, 1: 1}, ["payment_type"])

    def test_prespecified_objective_and_cost_cannot_be_changed(self):
        protocol = self.protocol("aps")
        protocol["threshold"]["fn_cost"] = 10
        with self.assertRaisesRegex(ValueError, "10FP"):
            validate_protocol(protocol)
        protocol = self.protocol()
        protocol["primary_metric"] = "macro_f1"
        with self.assertRaisesRegex(ValueError, "prespecified"):
            validate_protocol(protocol)
        with self.assertRaisesRegex(ValueError, "Dataset name"):
            make_splits(self.dataset(name="aps"), self.protocol(), 0)


if __name__ == "__main__":
    unittest.main()
