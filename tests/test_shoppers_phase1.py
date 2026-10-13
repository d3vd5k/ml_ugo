import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.results import enforce_stage_gate, load_study, require_variant
from benchmarks.run import evaluate_cell
from benchmarks.scoring import oof_mma, prepare_oof_partitions
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import make_preprocessor
from experiments import RunArtifacts


ROOT = Path(__file__).resolve().parents[1]
SEED = 19
SETTINGS = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 2,
            "augmentation": {"mode": "matched", "total": 6}, "method_parameters": {}}


class ShoppersPhase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        selected, _ = train_test_split(np.arange(len(real["y"])), train_size=180,
                                       stratify=real["y"], random_state=SEED)
        cls.dataset = {"name": "shoppers", "X": real["X"].iloc[selected].reset_index(drop=True),
                       "y": real["y"][selected].copy(), "row_ids": real["row_ids"][selected].copy()}

    def folds(self, frame=None):
        data = self.dataset
        return prepare_oof_partitions(data["X"] if frame is None else frame,
                                      data["y"], data["row_ids"], SEED)

    def test_every_real_row_has_one_disjoint_oof_score_and_correct_mma(self):
        data = self.dataset
        partitions = self.folds()
        scores, audit = oof_mma(partitions, data["y"], "logistic_regression", {"C": .1}, SEED, SETTINGS)
        self.assertEqual(scores.dtype, np.dtype("float32"))
        self.assertEqual(scores.shape, data["y"].shape)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(audit["row_ids"], data["row_ids"])
        coverage = np.zeros(len(scores), dtype=int)
        all_indices = set(range(len(scores)))
        for partition, fold_audit in zip(partitions, audit["folds"]):
            train, heldout = partition["train_indices"], partition["score_indices"]
            self.assertFalse(set(train) & set(heldout))
            self.assertEqual(set(train) | set(heldout), all_indices)
            coverage[heldout] += 1
            np.testing.assert_array_equal(partition["train_row_ids"], data["row_ids"][train])
            np.testing.assert_array_equal(partition["score_row_ids"], data["row_ids"][heldout])
            np.testing.assert_array_equal(fold_audit["train_row_ids"], data["row_ids"][train])
            np.testing.assert_array_equal(fold_audit["score_row_ids"], data["row_ids"][heldout])
            np.testing.assert_array_equal(np.asarray(audit["fold_ids"])[heldout], partition["fold"])
            self.assertEqual(fold_audit["train_count"], len(train))
            self.assertEqual(fold_audit["score_count"], len(heldout))
            self.assertEqual(fold_audit["synthetic_training_count"], 0)
            estimator = make_estimator("logistic_regression", fold_audit["model_seed"],
                                       parameters={"C": .1}, device="cpu", mlp_epochs=2)
            reference_model = fit_estimator(estimator, partition["X_train"], data["y"][train])
            np.testing.assert_allclose(np.asarray(audit["probabilities"])[heldout],
                                       reference_model.predict_proba(partition["X_score"]), atol=1e-6)
        np.testing.assert_array_equal(coverage, np.ones(len(scores), dtype=int))
        probabilities = np.asarray(audit["probabilities"])
        self.assertEqual(probabilities.shape, (len(scores), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        classes = np.asarray(audit["classes"])
        true_columns = np.searchsorted(classes, data["y"])
        true_probability = probabilities[np.arange(len(scores)), true_columns]
        competing = probabilities.copy()
        competing[np.arange(len(scores)), true_columns] = -np.inf
        np.testing.assert_allclose(scores, competing.max(axis=1) - true_probability, atol=1e-6)

    def test_fold_preprocessing_excludes_heldout_numeric_missingness_and_categories(self):
        frame = self.dataset["X"].copy()
        initial = self.folds(frame)[0]
        train, heldout = initial["train_indices"], initial["score_indices"]
        numeric = "Administrative_Duration"
        category = "VisitorType"
        frame[numeric] = frame[numeric].astype(float)
        frame[category] = frame[category].astype(object)
        frame.loc[train[:5], numeric] = np.nan
        frame.loc[train[:3], category] = np.nan
        before = self.folds(frame)[0]
        changed = frame.copy()
        changed.loc[heldout, numeric] = 1e12
        changed.loc[heldout[::2], numeric] = np.nan
        changed.loc[heldout, category] = "heldout-only-category"
        after = self.folds(changed)[0]
        np.testing.assert_array_equal(before["train_indices"], after["train_indices"])
        np.testing.assert_array_equal(before["X_train"], after["X_train"])
        reference = make_preprocessor(frame.iloc[train])
        expected_train = reference.fit_transform(frame.iloc[train])
        expected_score = reference.transform(changed.iloc[heldout])
        np.testing.assert_allclose(after["X_train"], expected_train, rtol=0, atol=0)
        np.testing.assert_allclose(after["X_score"], expected_score, rtol=0, atol=0)
        self.assertTrue(np.isfinite(after["X_train"]).all())
        self.assertTrue(np.isfinite(after["X_score"]).all())
        state = after["preprocessing"]
        self.assertNotIn("heldout-only-category", str(state))
        numeric_position = state["numeric"]["columns"].index(numeric)
        self.assertAlmostEqual(float(state["numeric"]["impute"]["statistics_"][numeric_position]),
                               float(frame.iloc[train][numeric].median()))
        numeric_imputed = frame.iloc[train][numeric].fillna(frame.iloc[train][numeric].median())
        self.assertAlmostEqual(float(state["numeric"]["scale"]["mean_"][numeric_position]),
                               float(numeric_imputed.mean()))
        columns = reference.named_steps["columns"]
        categorical_columns = list(columns.transformers_[1][2])
        encoder = columns.named_transformers_["categorical"].named_steps["encode"]
        category_index = categorical_columns.index(category)
        start = columns.output_indices_["categorical"].start + sum(
            len(values) for values in encoder.categories_[:category_index])
        stop = start + len(encoder.categories_[category_index])
        np.testing.assert_array_equal(after["X_score"][:, start:stop], 0)

    def test_fold_predictions_do_not_learn_heldout_labels(self):
        partitions = self.folds()
        heldout = partitions[0]["score_indices"]
        changed_labels = self.dataset["y"].copy()
        changed_labels[heldout] = 1 - changed_labels[heldout]
        for algorithm, parameters in (("logistic_regression", {"C": .1}),
                                      ("mlp", {"hidden_dim": 8})):
            with self.subTest(algorithm=algorithm):
                scores, original = oof_mma(partitions, self.dataset["y"], algorithm, parameters, SEED, SETTINGS)
                changed_scores, changed = oof_mma(partitions, changed_labels, algorithm, parameters, SEED, SETTINGS)
                np.testing.assert_array_equal(np.asarray(original["probabilities"])[heldout],
                                              np.asarray(changed["probabilities"])[heldout])
                np.testing.assert_allclose(changed_scores[heldout], -scores[heldout], atol=1e-6)

    def test_extreme_outer_test_cannot_change_u1_selection_or_inner_oof(self):
        data = self.dataset
        train, test = train_test_split(np.arange(len(data["y"])), test_size=.2,
                                      stratify=data["y"], random_state=SEED)
        fit, validation = train_test_split(train, test_size=.25, stratify=data["y"][train], random_state=SEED)
        split = {"fold": 0, "train_indices": train, "fit_indices": fit,
                 "validation_indices": validation, "test_indices": test}
        changed = {**data, "X": data["X"].copy(), "y": data["y"].copy()}
        for column in changed["X"]:
            values = changed["X"][column]
            if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
                changed["X"][column] = values.astype(float)
                changed["X"].loc[test, column] = 1e9
            else:
                changed["X"][column] = values.astype(object)
                changed["X"].loc[test, column] = "outer-test-only-category"
        changed["y"][test] = 1 - changed["y"][test]
        selections, candidates, outers = [], [], []
        with tempfile.TemporaryDirectory() as root:
            for index, dataset in enumerate((data, changed)):
                artifacts = RunArtifacts(Path(root) / str(index), {"random_state": SEED})
                row = evaluate_cell(dataset, split, "logistic_regression", "ugo_oof_mma", "U1_oof_mma",
                                    [{"C": .1}, {"C": 1.}], SETTINGS, self.protocol, artifacts, {"seed": SEED})
                self.assertEqual(row["status"], "completed")
                folder = artifacts.fold_path("shoppers", 0) / f"seed-{SEED}" / "logistic_regression" / "U1_oof_mma"
                selections.append(json.loads((folder / "selection.json").read_text()))
                candidates.append([json.loads((folder / f"candidate-{candidate}.json").read_text())
                                   for candidate in range(2)])
                outers.append(json.loads((folder / "outer.json").read_text()))
        def without_runtime(value):
            # Wall-clock measurements are not learned state and cannot repeat exactly.
            if isinstance(value, dict):
                return {key: without_runtime(item) for key, item in value.items()
                        if key not in {"fit_score_seconds", "generation_seconds"}}
            if isinstance(value, list):
                return [without_runtime(item) for item in value]
            return value

        self.assertEqual(without_runtime(selections[0]), without_runtime(selections[1]))
        self.assertEqual(without_runtime(candidates[0]), without_runtime(candidates[1]))
        self.assertEqual(without_runtime(outers[0]), without_runtime(outers[1]))
        inner_ids = set(data["row_ids"][fit])
        outer_ids = set(data["row_ids"][train])
        for candidate in candidates[0]:
            self.assertEqual(set(candidate["seed_scoring"]["row_ids"]), inner_ids)
            self.assertFalse(set(candidate["seed_scoring"]["row_ids"]) & set(data["row_ids"][test]))
        final = outers[0]["seed_scoring"]
        self.assertEqual(set(final["row_ids"]), outer_ids)
        self.assertNotEqual(set(final["row_ids"]), inner_ids)
        self.assertEqual(final["synthetic_training_count"], 0)
        for fold in final["folds"]:
            self.assertFalse(set(fold["train_row_ids"]) & set(fold["score_row_ids"]))
            self.assertEqual(set(fold["train_row_ids"]) | set(fold["score_row_ids"]), outer_ids)


class Phase1UserGateTests(unittest.TestCase):
    def study(self):
        return load_study(ROOT / "configs/ablations/stage1_components.yaml")

    def test_ready_u2_waits_for_explicit_u1_keep_or_discard(self):
        study = self.study()
        study["manual_review"]["decisions"]["U1"] = "pending"
        study["variants"]["U2"].update(available=True, status="ready", missing_components=[])
        before = copy.deepcopy(study)
        with self.assertRaisesRegex(ValueError, "awaits user decisions: U1"):
            require_variant(study, "U2")
        self.assertEqual(study, before)
        for decision in ("keep", "discard"):
            with self.subTest(decision=decision):
                decided = copy.deepcopy(study)
                decided["manual_review"]["decisions"]["U1"] = decision
                before = copy.deepcopy(decided)
                require_variant(decided, "U2")
                self.assertEqual(decided, before)

    def test_running_available_candidate_does_not_choose_retention(self):
        study = self.study()
        study["manual_review"]["decisions"] = {"U1": "pending", "U2": "discard", "U3": "keep",
                                               "U4": "pending", "U5": "discard"}
        before = copy.deepcopy(study)
        require_variant(study, "U1_oof_mma")
        self.assertEqual(study, before)

    def test_freeze_rejects_each_pending_manual_decision_even_with_evidence(self):
        study = self.study()
        variants = ["U0", "U1", "U2", "U3", "U4", "U5", "W5", "R5"]
        for name in variants:
            study["variants"][name].update(available=True, status="ready", missing_components=[])
        study["manual_review"]["decisions"] = {name: "keep" for name in ("U1", "U2", "U3", "U4", "U5")}
        study["freeze"] = {"status": "frozen", "evidence": {
            "component_justifications": {name: "accepted by user" for name in variants},
            "completed_ablations": variants, "generation_vs_W5_R5": "comparison recorded",
            "simulation_diagnostics": "recorded", "configuration_sha256": "a" * 64}}
        for name in study["manual_review"]["decisions"]:
            with self.subTest(component=name):
                pending = copy.deepcopy(study)
                pending["manual_review"]["decisions"][name] = "pending"
                before = copy.deepcopy(pending)
                with self.assertRaisesRegex(ValueError, "explicit user keep/discard decisions"):
                    enforce_stage_gate(pending, "confirmatory")
                self.assertEqual(pending, before)
        before = copy.deepcopy(study)
        enforce_stage_gate(study, "confirmatory")
        self.assertEqual(study, before)


if __name__ == "__main__":
    unittest.main()
