import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchmarks.f2 import evaluate_f2_cell, load_f1_source
from benchmarks.protocols import load_dataset, make_splits
from benchmarks.run import candidate_configs, prepare_partitions, smoke_split, split_fingerprint
from experiments import RunArtifacts, data_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/smoke/20261007T193854-789c042995bf"


class ShoppersF2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((SOURCE / "manifest.json").read_text())
        cls.protocol = manifest["protocol_snapshot"]
        cls.data = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        original = next(iter(make_splits(cls.data, cls.protocol, 0)))
        cls.split = smoke_split(cls.data, original, 0, 400, 120, cls.protocol["validation_fraction"])
        cls.settings = {**manifest["arguments"]["effective_settings"], "f1_source_run": str(SOURCE)}
        cls.candidates = candidate_configs(cls.protocol, "logistic_regression", 0, 1)
        cls.metadata = {"seed": 0, "algorithm": "logistic_regression", "experiment_type": "setup_smoke",
            "dataset_fingerprint": data_fingerprint(cls.data["X"], cls.data["y"], cls.data["row_ids"]),
            "split_fingerprint": split_fingerprint(cls.data, cls.split), "sampler_search": {"f2_q": [None]}}

    def test_source_identity_uses_split_fingerprint_and_reuses_f1_core(self):
        partitions = prepare_partitions(self.data, self.split)
        record, selected, gates, calibration, reference = load_f1_source(self.data, self.split,
            "logistic_regression", 0, self.candidates, self.settings, self.protocol, self.metadata, partitions)
        self.assertEqual(record["split_fingerprint"], self.metadata["split_fingerprint"])
        self.assertIs(gates["inner"].core, gates["outer"].core)
        self.assertEqual(reference["f1_selected_tau"], selected["inner"]["sampler_parameters"]["f1_tau"])
        fit, cal, validation, test = map(set, (gates["inner"].core.fit_row_ids.tolist(),
            calibration.row_ids.tolist(), gates["inner"].validation_row_ids.tolist(),
            self.data["row_ids"][self.split["test_indices"]].tolist()))
        self.assertFalse(fit & cal or fit & validation or cal & validation or (fit | cal | validation) & test)
        _, _, other, other_cal, _ = load_f1_source(self.data, self.split, "random_forest", 0,
            candidate_configs(self.protocol, "random_forest", 0, 1), self.settings, self.protocol,
            self.metadata, partitions)
        self.assertIs(other["inner"].core, gates["inner"].core)
        self.assertIs(other_cal, calibration)

    def test_off_reproduces_f1_predictions_original_coordinates_and_ancestry(self):
        with tempfile.TemporaryDirectory() as root:
            artifacts = RunArtifacts(root, {"random_state": 0})
            row = evaluate_f2_cell(self.data, self.split, "logistic_regression", "F2_support_off",
                self.candidates, self.settings, self.protocol, artifacts, self.metadata)
            expected = pd.read_csv(SOURCE / "shoppers/fold-0/seed-0/logistic_regression/F1/predictions.csv",
                                   float_precision="round_trip")
            actual = pd.read_csv(row["predictions_path"], float_precision="round_trip")
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            folder = Path(row["predictions_path"]).parent
            outer = json.loads((folder / "outer.json").read_text())
            self.assertTrue(row["support_off_matches_F1"])
            for proposal in outer["support_summary"]["candidate_log"]:
                self.assertEqual(proposal["original_coordinates"], proposal["final_coordinates"])
                self.assertEqual(proposal["relocation_index"], -1)
            budget = outer["support_summary"]["proposal_budget"]
            self.assertEqual(budget["attempted"], budget["accepted"] + budget["rejected"])
            self.assertLessEqual(budget["attempted"], 40)

    def test_selection_and_generation_do_not_use_outer_features_or_labels(self):
        # Source identity must still validate exactly; its immutable raw table is
        # original. Only already-prepared test queries/labels are poisoned, after
        # checking identity. They are never inputs to q selection or generation.
        snapshots = []
        with tempfile.TemporaryDirectory() as root:
            for poisoned in (False, True):
                partitions = prepare_partitions(self.data, self.split)
                data = self.data
                metadata = {**self.metadata, "sampler_search": {"f2_q": [.01, .05, .10]}}
                if poisoned:
                    partitions["X_test"] = np.full_like(partitions["X_test"], 1e6)
                    data = {**self.data, "y": self.data["y"].copy()}
                    data["y"][self.split["test_indices"]] = 1 - data["y"][self.split["test_indices"]]
                artifacts = RunArtifacts(Path(root) / str(poisoned), {"random_state": 0})
                row = evaluate_f2_cell(data, self.split, "logistic_regression", "F2",
                    self.candidates, self.settings, self.protocol, artifacts, metadata, partitions)
                folder = Path(row["predictions_path"]).parent
                selection = json.loads((folder / "selection.json").read_text())
                outer = json.loads((folder / "outer.json").read_text())
                snapshots.append((selection["selected_q"], selection["locked_tau"],
                    [trial["metrics"] for trial in selection["quantile_candidates"]],
                    outer["support_summary"]["candidate_log"], outer["epoch_sequence"]))
        self.assertEqual(snapshots[0], snapshots[1])

    def test_source_settings_or_protocol_drift_is_rejected(self):
        for settings, protocol, candidates in (
                ({**self.settings, "augmentation": {"mode": "matched", "total": 41}}, self.protocol, self.candidates),
                (self.settings, {**self.protocol, "threshold": {"name": "fixed", "threshold": .7}}, self.candidates),
                (self.settings, self.protocol, [{"C": 99.}])):
            with self.subTest(settings=settings, protocol=protocol):
                with self.assertRaises(ValueError):
                    load_f1_source(self.data, self.split, "logistic_regression", 0, candidates, settings,
                        protocol, self.metadata, prepare_partitions(self.data, self.split))


if __name__ == "__main__":
    unittest.main()
