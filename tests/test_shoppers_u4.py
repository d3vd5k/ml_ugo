import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import benchmarks.run as runner
from benchmarks.protocols import load_dataset, load_protocol
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from experiments import RunArtifacts


ROOT = Path(__file__).resolve().parents[1]
SEED = 19
METHOD = "ugo_stopping_only"
FRACTIONS = [0., .25, .5, 1.]


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def learned_state(value):
    if isinstance(value, dict):
        return {key: learned_state(item) for key, item in value.items()
                if not key.endswith("seconds")}
    if isinstance(value, list):
        return [learned_state(item) for item in value]
    return value


class U4ShoppersConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real["y"])), train_size=360,
                                    stratify=real["y"], random_state=SEED)
        cls.data = {"name": "shoppers", "X": real["X"].iloc[chosen].reset_index(drop=True),
                    "y": real["y"][chosen].copy(), "row_ids": real["row_ids"][chosen].copy()}
        cls.data["X"]["VisitorType"] = cls.data["X"]["VisitorType"].astype(object)
        train, test = train_test_split(np.arange(len(chosen)), test_size=.2,
                                       stratify=cls.data["y"], random_state=SEED)
        fit, validation = train_test_split(train, test_size=.25,
                                           stratify=cls.data["y"][train], random_state=SEED)
        cls.split = {"fold": 0, "train_indices": train, "fit_indices": fit,
                     "validation_indices": validation, "test_indices": test}
        cls.settings = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 4,
                        "augmentation": {"mode": "matched", "total": 40}, "method_parameters": {}}
        cls.candidates = [{"C": .01}, {"C": .1}, {"C": 1.}]

    def run_cell(self, root, dataset, policy, candidates=None):
        candidates = self.candidates if candidates is None else candidates
        partitions = runner.prepare_partitions(dataset, self.split)
        artifacts = RunArtifacts(Path(root), {"random_state": SEED})
        fits, calls = [], []
        depth = 0
        original_train = runner.train_method
        original_fit = runner.fit_estimator

        def record_fit(estimator, X, y, **kwargs):
            model = original_fit(estimator, X, y, **kwargs)
            fits.append({"model": model, "X": array(X), "y": array(y),
                         "validation": kwargs.get("validation_data")})
            return model

        def record_train(*args, **kwargs):
            nonlocal depth
            start = len(fits)
            depth += 1
            try:
                model, audit = original_train(*args, **kwargs)
            finally:
                depth -= 1
            if depth == 0:
                calls.append({"model": model, "audit": audit, "fits": fits[start:],
                              "X": array(args[3]), "y": array(args[4])})
            return model, audit

        with patch.object(runner, "fit_estimator", side_effect=record_fit), \
                patch.object(runner, "train_method", side_effect=record_train):
            row = runner.evaluate_cell(dataset, self.split, "logistic_regression", METHOD, policy,
                                       candidates, self.settings, self.protocol, artifacts,
                                       {"seed": SEED, "sampler_parameters": {
                                           "policy": policy, "budget_fractions": FRACTIONS}},
                                       partitions=partitions)
        folder = artifacts.fold_path("shoppers", 0) / f"seed-{SEED}" / "logistic_regression" / policy
        audit = {name: json.loads((folder / f"{name}.json").read_text())
                 for name in ("selection", "outer", "preprocessing", "reference_rows")}
        audit["candidates"] = [json.loads((folder / f"candidate-{index}.json").read_text())
                               for index in range(len(candidates))]
        return {"audit": audit, "row": row, "calls": calls, "partitions": partitions,
                "predictions": pd.read_csv(folder / "predictions.csv")}

    def assert_real_prefix_and_ancestry(self, call):
        audit = call["audit"]
        selected_fit = next(item for item in call["fits"] if item["model"] is call["model"])
        real_count = len(call["y"])
        np.testing.assert_array_equal(selected_fit["X"][:real_count], call["X"])
        np.testing.assert_array_equal(selected_fit["y"][:real_count], call["y"])
        self.assertEqual(len(selected_fit["y"]) - real_count, audit["augmentation"]["actual_count"])
        self.assertEqual(len(audit["provenance"]), audit["augmentation"]["actual_count"])
        for parent in audit["provenance"]:
            self.assertGreaterEqual(parent["row_index"], real_count)
            self.assertLess(parent["row_index"], len(selected_fit["y"]))
            for key in ("seed_index", "partner_index", "relocation_index"):
                index = parent.get(key)
                if index is None or (key == "relocation_index" and index == -1):
                    continue
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, real_count)
                self.assertEqual(int(call["y"][index]), parent["class"])
        for fitted in call["fits"]:
            np.testing.assert_array_equal(fitted["X"][:real_count], call["X"])
            np.testing.assert_array_equal(fitted["y"][:real_count], call["y"])
        return selected_fit

    def assert_selection_and_predictions(self, snapshot, dataset, policy):
        audit, calls = snapshot["audit"], snapshot["calls"]
        partitions = snapshot["partitions"]
        self.assertEqual(snapshot["row"]["status"], "completed")
        self.assertEqual(len(calls), len(self.candidates) + 1)
        fit, validation = self.split["fit_indices"], self.split["validation_indices"]
        train, test = self.split["train_indices"], self.split["test_indices"]
        self.assertEqual(audit["reference_rows"]["inner_real_row_ids"], dataset["row_ids"][fit].tolist())
        self.assertEqual(audit["reference_rows"]["outer_real_row_ids"], dataset["row_ids"][train].tolist())
        options = []
        for candidate_id, (candidate, call) in enumerate(zip(audit["candidates"], calls)):
            np.testing.assert_array_equal(call["X"], partitions["X_fit"])
            np.testing.assert_array_equal(call["y"], dataset["y"][fit])
            selection = candidate["stopping_selection"]
            self.assertEqual(selection["policy"], policy)
            self.assertEqual(selection["objective"], self.protocol["primary_metric"])
            real_only_fits = [item for item in call["fits"] if len(item["y"]) == len(fit)]
            self.assertEqual(len(real_only_fits), 4 if policy == "fixed_grid" else 1)
            for fitted in call["fits"]:
                self.assertIsNotNone(fitted["validation"])
                np.testing.assert_array_equal(fitted["validation"][0], partitions["X_validation"])
                np.testing.assert_array_equal(fitted["validation"][1], dataset["y"][validation])
            self.assertEqual(selection["maximum_count"], self.settings["augmentation"]["total"])
            self.assertEqual(selection["fit_count"], len(call["fits"]))
            checkpoints = selection["checkpoints"]
            self.assertGreaterEqual(len(checkpoints), 1)
            self.assertLessEqual(len(checkpoints), len(FRACTIONS))
            self.assertEqual(checkpoints[0]["fraction"], 0.)
            self.assertEqual(checkpoints[0]["actual_count"], 0)
            self.assertEqual(checkpoints[0]["requested_count"], 0)
            for checkpoint in checkpoints:
                self.assertIn(checkpoint["fraction"], FRACTIONS)
                self.assertEqual(checkpoint["threshold"], .5)
                self.assertEqual(checkpoint["requested_count"],
                                 int(checkpoint["fraction"] * selection["maximum_count"]))
                self.assertLessEqual(checkpoint["actual_count"], checkpoint["requested_count"])
                self.assertEqual(checkpoint["score"], checkpoint["metrics"]["primary_metric_value"])
                self.assertIn("epoch_sequence", checkpoint)
                options.append((checkpoint, candidate_id))
            best_checkpoint = min(checkpoints, key=lambda item: (-item["score"], item["actual_count"],
                                                                item["requested_count"]))
            self.assertEqual(selection["selected_fraction"], best_checkpoint["fraction"])
            self.assertEqual(selection["selected_actual_count"], best_checkpoint["actual_count"])
            self.assertEqual(selection["selected_requested_count"], best_checkpoint["requested_count"])
            self.assertGreaterEqual(selection["generated_count_searched"], selection["selected_actual_count"])
            selected_fit = self.assert_real_prefix_and_ancestry(call)
            independent = fit_estimator(make_estimator("logistic_regression", SEED,
                                                       parameters=self.candidates[candidate_id], device="cpu"),
                                        selected_fit["X"], selected_fit["y"])
            probabilities = independent.predict_proba(partitions["X_validation"])
            np.testing.assert_allclose(call["model"].predict_proba(partitions["X_validation"]),
                                       probabilities, atol=1e-7)
            recomputed = average_precision_score(dataset["y"][validation], probabilities[:, 1])
            self.assertAlmostEqual(recomputed, best_checkpoint["score"], places=12)
            self.assertAlmostEqual(recomputed, candidate["metrics"]["primary_metric_value"], places=12)
        chosen, candidate_id = min(options, key=lambda item: (-item[0]["score"], item[0]["actual_count"],
                                                              item[0]["requested_count"], item[1]))
        self.assertEqual(audit["selection"]["selected_candidate"], candidate_id)
        inner = audit["selection"]["inner"]
        self.assertEqual(inner["sampler_parameters"]["budget_fraction"], chosen["fraction"])
        self.assertEqual(inner["sampler_parameters"]["policy"], policy)
        self.assertEqual(inner["epoch_sequence"], chosen["epoch_sequence"])
        outer, outer_call = audit["outer"], calls[-1]
        self.assertNotIn("stopping_selection", outer)
        self.assertIn("stopping_replay", outer)
        self.assertIn("stopping", outer["support_summary"])
        self.assertEqual(outer["sampler_parameters"]["budget_fraction"], chosen["fraction"])
        self.assertEqual(outer["sampler_parameters"]["policy"], policy)
        self.assertEqual(outer["augmentation"]["requested_count"], chosen["requested_count"])
        self.assertLessEqual(len(outer["rounds"]), self.settings["ugo_max_iterations"])
        np.testing.assert_array_equal(outer_call["X"], partitions["X_train"])
        np.testing.assert_array_equal(outer_call["y"], dataset["y"][train])
        self.assertTrue(all(item["validation"] is None for item in outer_call["fits"]))
        outer_fit = self.assert_real_prefix_and_ancestry(outer_call)
        independent = fit_estimator(make_estimator("logistic_regression", SEED,
                                                   parameters=self.candidates[candidate_id], device="cpu"),
                                    outer_fit["X"], outer_fit["y"])
        probabilities = independent.predict_proba(partitions["X_test"])
        predictions = snapshot["predictions"]
        self.assertEqual(predictions["row_id"].tolist(), dataset["row_ids"][test].tolist())
        np.testing.assert_array_equal(predictions["y_true"], dataset["y"][test])
        np.testing.assert_allclose(predictions[["probability_0", "probability_1"]], probabilities, atol=1e-7)
        np.testing.assert_array_equal(predictions["y_pred"], (probabilities[:, 1] >= .5).astype(int))

    def test_real_lr_selection_ancestry_and_predictions_ignore_outer_test_perturbations(self):
        changed = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        test = self.split["test_indices"]
        changed["X"].loc[test, "Administrative_Duration"] = 1e12
        changed["X"].loc[test, "VisitorType"] = "u4-test-only-category"
        changed["y"][test] = 1 - changed["y"][test]
        with tempfile.TemporaryDirectory() as root:
            for policy in ("checkpoint", "fixed_grid"):
                with self.subTest(policy=policy):
                    before = self.run_cell(Path(root) / policy / "original", self.data, policy)
                    after = self.run_cell(Path(root) / policy / "perturbed", changed, policy)
                    self.assertEqual(learned_state(before["audit"]), learned_state(after["audit"]))
                    self.assert_selection_and_predictions(before, self.data, policy)
                    self.assert_selection_and_predictions(after, changed, policy)
                    for role in ("X_fit", "X_validation", "X_train"):
                        np.testing.assert_array_equal(before["partitions"][role], after["partitions"][role])
                    self.assertFalse(np.array_equal(before["partitions"]["X_test"], after["partitions"]["X_test"]))

    def test_constant_classifier_ties_choose_zero_and_first_candidate_with_one_outer_fit(self):
        # A genuine constant-negative classifier gives every augmentation the same AP.
        # It still participates in real MMA scoring, support filtering and generation.
        with tempfile.TemporaryDirectory() as root, patch.object(
                runner, "make_estimator", side_effect=lambda *args, **kwargs:
                DummyClassifier(strategy="constant", constant=0)):
            for policy in ("checkpoint", "fixed_grid"):
                with self.subTest(policy=policy):
                    snapshot = self.run_cell(Path(root) / policy, self.data, policy, [{"C": .1}, {"C": 10.}])
                    audit = snapshot["audit"]
                    self.assertEqual(audit["selection"]["selected_candidate"], 0)
                    self.assertEqual(audit["selection"]["inner"]["sampler_parameters"]["budget_fraction"], 0.)
                    for candidate in audit["candidates"]:
                        selection = candidate["stopping_selection"]
                        self.assertEqual(selection["selected_fraction"], 0.)
                        self.assertEqual(selection["selected_actual_count"], 0)
                        self.assertGreater(selection["generated_count_searched"], 0)
                        self.assertEqual(len({item["score"] for item in selection["checkpoints"]}), 1)
                        self.assertEqual(candidate["provenance"], [])
                    outer_call = snapshot["calls"][-1]
                    self.assertEqual(len(outer_call["fits"]), 1)
                    self.assertEqual(audit["outer"]["augmentation"]["actual_count"], 0)
                    self.assertNotIn("stopping_selection", audit["outer"])
                    self.assertIn("stopping_replay", audit["outer"])
                    self.assert_real_prefix_and_ancestry(outer_call)
                    np.testing.assert_array_equal(snapshot["predictions"]["probability_1"], 0.)
                    np.testing.assert_array_equal(snapshot["predictions"]["y_pred"], 0)

    def test_nonzero_refit_replays_fraction_not_capped_inner_count(self):
        partitions = runner.prepare_partitions(self.data, self.split)
        fit, train = self.split["fit_indices"], self.split["train_indices"]
        validation = self.split["validation_indices"]
        outer_counts = np.bincount(self.data["y"][train])
        outer_maximum = min(1000, int(outer_counts.max() * len(outer_counts) - outer_counts.sum()))
        for policy in ("checkpoint", "fixed_grid"):
            with self.subTest(policy=policy):
                settings = copy.deepcopy(self.settings)
                settings["ugo_max_iterations"] = 1
                settings["augmentation"]["total"] = 1000
                settings["method_parameters"][METHOD] = {"policy": policy, "budget_fractions": FRACTIONS}
                _, inner = runner.train_method("logistic_regression", METHOD, {"C": .1},
                                               partitions["X_fit"], self.data["y"][fit],
                                               categorical_columns(partitions["inner"]), SEED, settings,
                                               validation=(partitions["X_validation"], self.data["y"][validation]),
                                               stopping_objective=self.protocol, positive_class=1)
                checkpoint = next(item for item in inner["stopping_selection"]["checkpoints"]
                                  if item["fraction"] > 0)
                self.assertLess(checkpoint["actual_count"], checkpoint["requested_count"])
                fraction = checkpoint["fraction"]
                settings["method_parameters"][METHOD]["budget_fraction"] = fraction
                fits = []
                def record_fit(estimator, X, y, **kwargs):
                    model = fit_estimator(estimator, X, y, **kwargs)
                    fits.append({"model": model, "X": array(X), "y": array(y),
                                 "validation": kwargs.get("validation_data")})
                    return model
                with patch.object(runner, "fit_estimator", side_effect=record_fit):
                    model, outer = runner.train_method(
                        "logistic_regression", METHOD, {"C": .1},
                        partitions["X_train"], self.data["y"][train],
                        categorical_columns(partitions["outer"]), SEED, settings,
                        replay_epochs=checkpoint["epoch_sequence"])
                self.assert_real_prefix_and_ancestry({
                    "model": model, "audit": outer, "fits": fits,
                    "X": partitions["X_train"], "y": self.data["y"][train]})
                self.assertTrue(all(item["validation"] is None for item in fits))
                self.assertLessEqual(len(fits), settings["ugo_max_iterations"] + 1)
                self.assertEqual(outer["augmentation"]["requested_count"], int(outer_maximum * fraction))
                self.assertNotEqual(outer["augmentation"]["requested_count"], checkpoint["requested_count"])
                self.assertNotEqual(outer["augmentation"]["requested_count"], checkpoint["actual_count"])
                self.assertLess(outer["augmentation"]["actual_count"], int(outer_maximum * fraction))
                self.assertLessEqual(len(outer["rounds"]), 1)
                self.assertEqual(outer["sampler_parameters"]["budget_fraction"], fraction)
                self.assertNotIn("stopping_selection", outer)
                self.assertIn("stopping_replay", outer)
                self.assertIn("stopping", outer["support_summary"])


if __name__ == "__main__":
    unittest.main()
