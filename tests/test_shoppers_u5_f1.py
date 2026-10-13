"""U5+F1 behavior on real Shoppers rows, with actual learners and frozen gates."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import benchmarks.run as runner
import benchmarks.scoring as scoring
from benchmarks.flow_partitions import flow_for_partition
from benchmarks.protocols import load_dataset, load_protocol
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns, make_preprocessor
from experiments import RunArtifacts, _preprocessing_state
from samplers.controls import resample
from samplers.flow_support import CalibratedFlowGate, calibrate_flow
from samplers.protected import prepare_protected_groups

ROOT = Path(__file__).resolve().parents[1]
SEED = 19
METHOD = "ugo_combined_f1"


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def learned_state(value):
    if isinstance(value, dict):
        return {key: learned_state(item) for key, item in value.items()
                if not isinstance(key, str) or not key.endswith("seconds")}
    if isinstance(value, (list, tuple)):
        return [learned_state(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class ShoppersU5F1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real["y"])), train_size=400,
                                    stratify=real["y"], random_state=SEED)
        cls.data = {"name": "shoppers", "X": real["X"].iloc[chosen].reset_index(drop=True),
                    "y": real["y"][chosen].copy(), "row_ids": real["row_ids"][chosen].copy()}
        train, test = train_test_split(np.arange(len(chosen)), test_size=.2,
                                      stratify=cls.data["y"], random_state=SEED)
        fit, validation = train_test_split(train, test_size=.25,
                                          stratify=cls.data["y"][train], random_state=SEED)
        cls.split = {"fold": 0, "train_indices": train, "fit_indices": fit,
                     "validation_indices": validation, "test_indices": test}
        cls.parameters = {"grouping": "kmeans", "allocation": "stability", "group_alpha": .5,
                          "alpha": 1., "n_clusters": 3, "repeats": 3,
                          "flow_epochs": 3, "f1_tau": .99}
        cls.settings = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 4,
                        "augmentation": {"mode": "matched", "total": 20},
                        "method_parameters": {METHOD: cls.parameters}}

    def inner_inputs(self):
        partitions = runner.prepare_partitions(self.data, self.split)
        indices = self.split["fit_indices"]
        X, y = partitions["X_fit"], self.data["y"][indices]
        oof = scoring.prepare_oof_partitions(self.data["X"].iloc[indices], y,
                                            self.data["row_ids"][indices], SEED)
        groups = prepare_protected_groups(X, y, SEED,
            runner.augmentation_targets(y, self.settings["augmentation"]), self.parameters)
        groups["real_row_ids"] = self.data["row_ids"][indices].copy()
        gate = flow_for_partition(self.data, self.split, partitions, "inner", SEED, self.parameters, "cpu")
        return partitions, X, y, oof, groups, gate

    def recorded_training(self, inputs):
        partitions, X, y, oof, groups, gate = inputs
        downstream, crossfits, scores_seen, sampled = [], [], [], []
        original_resample = runner.resample

        def record_fit(target):
            def fit(estimator, features, labels, **kwargs):
                model = fit_estimator(estimator, features, labels, **kwargs)
                target.append({"model": model, "X": array(features), "y": array(labels),
                               "validation": kwargs.get("validation_data")})
                return model
            return fit

        def record_resample(*args, **kwargs):
            original_score = kwargs["seed_score"]

            def score(model, features, labels):
                result = original_score(model, features, labels)
                scores_seen.append(array(result))
                return result

            result = original_resample(*args, **{**kwargs, "seed_score": score})
            sampled.append(result)
            return result

        with patch.object(runner, "fit_estimator", side_effect=record_fit(downstream)), \
                patch.object(scoring, "fit_estimator", side_effect=record_fit(crossfits)), \
                patch.object(runner, "resample", side_effect=record_resample):
            model, audit = runner.train_method("logistic_regression", METHOD, {"C": 1.}, X, y,
                categorical_columns(partitions["inner"]), SEED, self.settings,
                oof=oof, group_partition=groups, flow_support=gate)
        return model, audit, downstream, crossfits, scores_seen, sampled[0]

    def assert_actual_oof(self, audit, fitted, scores_seen):
        indices = self.split["fit_indices"]
        frame, labels = self.data["X"].iloc[indices], self.data["y"][indices]
        row_ids = self.data["row_ids"][indices]
        seed = audit["seed_scoring"]
        self.assertEqual(len(fitted), 3)
        self.assertEqual(seed["n_splits"], 3)
        self.assertEqual(seed["synthetic_training_count"], 0)
        self.assertEqual(seed["training_source"], "real_only")
        self.assertEqual(seed["refresh"], "fixed_before_generation")
        np.testing.assert_array_equal(seed["row_ids"], row_ids)
        probabilities = np.empty((len(labels), 2), dtype=np.float32)
        coverage = np.zeros(len(labels), dtype=int)
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(frame, labels)
        for (train, score), recorded, fold in zip(folds, fitted, seed["folds"]):
            transform = make_preprocessor(frame.iloc[train])
            X_train = transform.fit_transform(frame.iloc[train])
            X_score = transform.transform(frame.iloc[score])
            np.testing.assert_array_equal(recorded["X"], X_train)
            np.testing.assert_array_equal(recorded["y"], labels[train])
            self.assertIsNone(recorded["validation"])
            self.assertEqual(learned_state(fold["preprocessing"]),
                             learned_state(_preprocessing_state(transform)))
            np.testing.assert_array_equal(fold["train_row_ids"], row_ids[train])
            np.testing.assert_array_equal(fold["score_row_ids"], row_ids[score])
            self.assertFalse(set(fold["train_row_ids"]) & set(fold["score_row_ids"]))
            probabilities[score] = recorded["model"].predict_proba(X_score)
            coverage[score] += 1
        np.testing.assert_array_equal(coverage, 1)
        np.testing.assert_allclose(seed["probabilities"], probabilities, atol=1e-7)
        positions = np.searchsorted(seed["classes"], labels)
        competing = probabilities.copy()
        competing[np.arange(len(labels)), positions] = -np.inf
        scores = competing.max(axis=1) - probabilities[np.arange(len(labels)), positions]
        np.testing.assert_allclose(seed["scores"], scores, atol=1e-7)
        self.assertTrue(scores_seen)
        for actual in scores_seen:
            np.testing.assert_array_equal(actual, scores)
        return scores

    def test_actual_oof_original_gate_replay_ancestry_groups_caps_and_no_refill(self):
        inputs = self.inner_inputs()
        _, X, y, _, groups, gate = inputs
        before = {key: value.clone() for key, value in gate.core.model.state_dict().items()}
        model, audit, downstream, crossfits, scores_seen, sampled = self.recorded_training(inputs)
        scores = self.assert_actual_oof(audit, crossfits, scores_seen)
        support = audit["support_summary"]
        rows = support["candidate_log"]
        self.assertTrue(rows)
        rng = np.random.default_rng(SEED)
        real = np.asarray(X, dtype=np.float32)
        categorical = list(categorical_columns(inputs[0]["inner"]))
        for round_id in sorted({row["round"] for row in rows}):
            proposals = [row for row in rows if row["round"] == round_id]
            rng.random(len(proposals))  # Actual partner-choice draw precedes interpolation.
            interpolation = rng.random((len(proposals), 1)).astype(np.float32)
            seeds = np.asarray([row["seed_index"] for row in proposals])
            partners = np.asarray([row["partner_index"] for row in proposals])
            expected = real[seeds] + interpolation * (real[partners] - real[seeds])
            expected[:, categorical] = real[partners][:, categorical]
            np.testing.assert_array_equal([row["original_coordinates"] for row in proposals], expected)
        replay = gate.predict(np.asarray([row["original_coordinates"] for row in rows]),
                              np.asarray([row["class"] for row in rows]), .99)
        self.assertEqual([row["accepted"] for row in rows], replay["accepted"].tolist())
        for i, row in enumerate(rows):
            self.assertEqual(row["original_coordinates"], row["final_coordinates"])
            self.assertEqual(row["relocation_index"], -1)
            for source in ("seed", "partner"):
                index = row[f"{source}_index"]
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(y))
                self.assertEqual(y[index], row["class"])
                self.assertEqual(row[f"{source}_row_id"], str(groups["real_row_ids"][index]))
            self.assertEqual(row["group_id"], groups["row_group_ids"][row["seed_index"]])
            self.assertEqual(row["reason"], replay["reasons"][i])
            for key in ("log_ratio", "compatibility_score", "log_likelihood_intended",
                        "log_likelihood_opposing", "log_determinant"):
                value = float(replay[key][i])
                self.assertEqual(row[key], value if np.isfinite(value) else None)
        accepted = [row for row in rows if row["accepted"]]
        self.assertTrue(accepted)
        budget = support["proposal_budget"]
        self.assertEqual(budget["attempted"], 20)
        self.assertEqual(budget["cap"], 20)
        self.assertEqual(budget["accepted"], len(accepted))
        self.assertGreater(budget["rejected"], 0)
        self.assertEqual(budget["accepted"] + budget["rejected"], budget["attempted"])
        self.assertEqual(audit["augmentation"]["actual_count"], len(accepted))
        self.assertEqual(audit["augmentation"]["underspend"], budget["rejected"])
        targets = runner.augmentation_targets(y, self.settings["augmentation"])
        for label in np.unique(y):
            self.assertLessEqual(int((array(sampled.y) == label).sum()), targets[int(label)])
        np.testing.assert_array_equal(array(sampled.X)[:len(y)], np.asarray(X, dtype=np.float32))
        np.testing.assert_array_equal(array(sampled.y)[:len(y)], y)
        if accepted:
            np.testing.assert_array_equal(array(sampled.X)[len(y):],
                                          [row["original_coordinates"] for row in accepted])
        self.assertEqual(len(audit["provenance"]), len(accepted))
        for ancestry, proposal in zip(audit["provenance"], accepted):
            for key in ("seed_index", "partner_index", "relocation_index", "group_id", "proposal_id"):
                self.assertEqual(ancestry[key], proposal[key])
            index = ancestry["row_index"]
            self.assertGreaterEqual(index, len(y))
            np.testing.assert_array_equal(array(sampled.X)[index], proposal["original_coordinates"])
            self.assertEqual(array(sampled.y)[index], proposal["class"])
        allocation = support["group_support"]["allocation"]
        attempted = Counter(row["seed_index"] for row in rows)
        self.assertLessEqual(max(attempted.values()), allocation["seed_cap"])
        np.testing.assert_array_equal(budget["attempted_seed_multiplicity"],
                                      [attempted[i] for i in range(len(y))])
        group_attempts = Counter(row["group_id"] for row in rows)
        group_accepts = Counter(row["group_id"] for row in accepted)
        by_id = {group["id"]: group for group in groups["groups"]}
        for group in support["group_support"]["groups"]:
            self.assertLessEqual(group_attempts[group["id"]],
                                 int(np.floor(allocation["group_cap_multiplier"] * group["real_support"])))
            self.assertEqual(group["actual_generated"], group_accepts[group["id"]])
            np.testing.assert_array_equal(group["row_ids"], groups["real_row_ids"][group["indices"]])
        for record in support["group_support"]["allocation_records"]:
            class_indices = np.flatnonzero(y == record["target_class"])
            percentiles = np.zeros(len(y))
            percentiles[class_indices] = (rankdata(scores[class_indices]) - .5) / len(class_indices)
            for trace in record["group_trace"]:
                group = by_id[trace["group_id"]]
                self.assertAlmostEqual(trace["h"], float(percentiles[group["indices"]].mean()))
                if not trace["exclusions"]:
                    expected = group["stability"] * (trace["h"] + allocation["epsilon"]) / (
                        group["real_support"] + allocation["epsilon"]) ** allocation["group_alpha"]
                    self.assertAlmostEqual(trace["weight"], expected)
        self.assertEqual(audit["classifier_fit_count"], len(crossfits) + len(downstream))
        for fit in downstream:
            np.testing.assert_array_equal(fit["X"][:len(y)], np.asarray(X, dtype=np.float32))
            np.testing.assert_array_equal(fit["y"][:len(y)], y)
        self.assertIs(model, downstream[-1]["model"])
        fit_ids = set(gate.core.fit_row_ids.tolist())
        self.assertTrue(fit_ids <= set(groups["real_row_ids"]))
        self.assertFalse(fit_ids & set(self.data["row_ids"][self.split["test_indices"]]))
        self.assertFalse(fit_ids & set(self.data["row_ids"][self.split["validation_indices"]]))
        self.assertFalse(fit_ids & set(gate.calibration_row_ids))
        for key, value in gate.core.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_all_rejected_reuses_real_model_after_exactly_three_oof_and_one_downstream_fit(self):
        inputs = self.inner_inputs()
        _, X, y, _, _, gate = inputs
        # Exercise actual flow likelihood rejection, without mocking gate decisions.
        with torch.no_grad():
            gate.core.model.means[1].fill_(1e6)
            gate.core.model.log_stds.zero_()
        model, audit, downstream, crossfits, scores_seen, sampled = self.recorded_training(inputs)
        self.assert_actual_oof(audit, crossfits, scores_seen)
        self.assertEqual(audit["stop_reason"], "f1_no_candidates_accepted")
        self.assertEqual(audit["augmentation"]["actual_count"], 0)
        self.assertEqual(audit["classifier_fit_count"], 4)
        self.assertEqual(len(downstream), 1)
        self.assertTrue(audit["unchanged_final_fit_skipped"])
        self.assertIs(model, downstream[0]["model"])
        self.assertEqual(audit["support_summary"]["proposal_budget"]["rejected"], 20)
        np.testing.assert_array_equal(array(sampled.X), np.asarray(X, dtype=np.float32))
        np.testing.assert_array_equal(array(sampled.y), y)
        reference = fit_estimator(make_estimator("logistic_regression", SEED,
                                  parameters={"C": 1.}, device="cpu"), X, y)
        np.testing.assert_array_equal(model.predict_proba(X), reference.predict_proba(X))

    def test_missing_oof_frozen_gate_and_f2_composition_are_rejected(self):
        partitions, X, y, oof, groups, gate = self.inner_inputs()
        scores, _ = scoring.oof_mma(oof, y, "logistic_regression", {"C": 1.}, SEED, self.settings)

        def seed_score(model, features, labels):
            return torch.as_tensor(scores, dtype=features.dtype, device=features.device)

        def train(features, labels):
            return fit_estimator(make_estimator("logistic_regression", SEED,
                                 parameters={"C": 1.}, device="cpu"), features, labels)

        common = {"device": "cpu", "model_fn": train, "group_partition": groups,
                  "categorical_columns": categorical_columns(partitions["inner"]),
                  "seed_score": seed_score, "flow_support": gate, "parameters": self.parameters}
        reserved = np.flatnonzero(np.isin(self.data["row_ids"], gate.calibration_row_ids))
        calibration = calibrate_flow(gate.core, self.data["X"].iloc[reserved],
                                    self.data["y"][reserved], self.data["row_ids"][reserved])
        invalid = [("OOF MMA", {"seed_score": None}),
                   ("frozen", {"flow_support": None}),
                   ("plain frozen", {"flow_support": object()}),
                   ("Calibrated F2", {"flow_support": CalibratedFlowGate(gate, calibration, q=None)}),
                   ("Calibrated F2", {"flow_support": CalibratedFlowGate(gate, calibration, q=.1)}),
                   ("F2 support parameters", {"parameters": {**self.parameters, "f2_q": None}}),
                   ("F2 support parameters", {"parameters": {**self.parameters, "f2_q": .1}}),
                   ("F2 support parameters", {"parameters": {**self.parameters, "support_filter": False}}),
                   ("F2 support parameters", {"parameters": {**self.parameters, "support_filter": True}})]
        for message, overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                resample(METHOD, X, y, SEED, **{**common, **overrides})
        for mutate in (lambda: gate.core.model.train(), lambda: gate.core.model.requires_grad_(True)):
            mutate()
            with self.assertRaisesRegex(ValueError, "plain frozen"):
                resample(METHOD, X, y, SEED, **common)
            gate.core.model.eval()
            gate.core.model.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "cross-fit partitions"):
            runner.train_method("logistic_regression", METHOD, {"C": 1.}, X, y,
                categorical_columns(partitions["inner"]), SEED, self.settings, flow_support=gate)

    def test_outer_test_poisoning_preserves_u5_learner_tau_seed_scores_and_training(self):
        changed = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        test = self.split["test_indices"]
        changed["X"].loc[test, "Administrative_Duration"] = 1e12
        changed["y"][test] = 1 - changed["y"][test]
        candidates = [{"C": .1}, {"C": 1.}]
        metadata = {"seed": SEED, "sampler_parameters": self.parameters,
                    "sampler_search": {"group_alpha": [0., .5], "f1_tau": [.5, .8]}}
        snapshots = []
        original_train = runner.train_method
        with tempfile.TemporaryDirectory() as root:
            for index, data in enumerate((self.data, changed)):
                calls = []

                def record_train(*args, **kwargs):
                    model, audit = original_train(*args, **kwargs)
                    calls.append({"method": args[1], "X": array(args[3]), "y": array(args[4]),
                                  "audit": audit, "training_probabilities": model.predict_proba(args[3])})
                    return model, audit

                artifacts = RunArtifacts(Path(root) / str(index), {"random_state": SEED})
                with patch.object(runner, "train_method", side_effect=record_train):
                    runner.evaluate_cell(data, self.split, "logistic_regression", METHOD, "U5_F1",
                        candidates, self.settings, self.protocol, artifacts, metadata)
                family = artifacts.fold_path("shoppers", 0) / f"seed-{SEED}" / "logistic_regression"
                folder = family / "U5_F1"
                selection = json.loads((folder / "selection.json").read_text())
                canonical = selection["u5_selection"]
                self.assertEqual(selection["inner"]["parameters"], canonical["inner"]["parameters"])
                self.assertEqual(selection["selected_candidate"], canonical["selected_candidate"])
                canonical_folder = family / selection["canonical_variant"]
                outer = json.loads((folder / "outer.json").read_text())
                outer.pop("group_recall")
                canonical_outer = json.loads((canonical_folder / "outer.json").read_text())
                for call in calls:
                    if len(call["y"]) == len(self.split["train_indices"]):
                        call["audit"].pop("group_recall", None)
                snapshots.append(learned_state({"selection": selection, "outer": outer,
                    "canonical_outer": canonical_outer, "calls": calls}))
        self.assertEqual(snapshots[0], snapshots[1])


if __name__ == "__main__":
    unittest.main()
