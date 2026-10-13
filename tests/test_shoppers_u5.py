import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import benchmarks.run as runner
import benchmarks.scoring as scoring
from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.results import enforce_stage_gate, load_study, require_variant
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import make_preprocessor
from experiments import RunArtifacts, _preprocessing_state
from samplers.protected import prepare_protected_groups


ROOT = Path(__file__).resolve().parents[1]
SEED = 19
METHOD = "ugo_combined"
GROUPS = {"grouping": "kmeans", "allocation": "stability"}
ALPHAS = [0., .5, 1.]


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


class U5ShoppersConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real["y"])), train_size=600,
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
                        "augmentation": {"mode": "matched", "total": 24}, "method_parameters": {}}
        cls.candidates = runner.candidate_configs(cls.protocol, "logistic_regression", SEED, 10)

    def run_cell(self, root, dataset, *, method=METHOD, candidates=None, partitions=None):
        candidates = self.candidates if candidates is None else candidates
        partitions = runner.prepare_partitions(dataset, self.split) if partitions is None else partitions
        artifacts = RunArtifacts(Path(root), {"random_state": SEED})
        fits, scoring_fits, calls = [], [], []
        original_train = runner.train_method

        def record_fit(estimator, X, y, **kwargs):
            model = fit_estimator(estimator, X, y, **kwargs)
            fits.append({"model": model, "X": array(X), "y": array(y),
                         "validation": kwargs.get("validation_data")})
            return model

        def record_scoring_fit(estimator, X, y, **kwargs):
            model = fit_estimator(estimator, X, y, **kwargs)
            scoring_fits.append({"model": model, "X": array(X), "y": array(y),
                                 "validation": kwargs.get("validation_data")})
            return model

        def record_train(*args, **kwargs):
            start, scoring_start = len(fits), len(scoring_fits)
            model, audit = original_train(*args, **kwargs)
            calls.append({"model": model, "audit": audit, "fits": fits[start:],
                          "scoring_fits": scoring_fits[scoring_start:],
                          "X": array(args[3]), "y": array(args[4]), "oof": kwargs.get("oof")})
            return model, audit

        metadata = {"seed": SEED}
        if method in (METHOD, "ugo_groups_only"):
            metadata.update(sampler_parameters=GROUPS, sampler_search={"group_alpha": ALPHAS})
        with patch.object(runner, "fit_estimator", side_effect=record_fit), \
                patch.object(scoring, "fit_estimator", side_effect=record_scoring_fit), \
                patch.object(runner, "train_method", side_effect=record_train):
            row = runner.evaluate_cell(dataset, self.split, "logistic_regression", method, "candidate",
                                       candidates, self.settings, self.protocol, artifacts, metadata,
                                       partitions=partitions)
        folder = artifacts.fold_path("shoppers", 0) / f"seed-{SEED}" / "logistic_regression" / "candidate"
        audit = {name: json.loads((folder / f"{name}.json").read_text())
                 for name in ("selection", "outer", "preprocessing", "reference_rows")}
        audit["candidates"] = [json.loads((folder / f"candidate-{index}.json").read_text())
                               for index in range(len(candidates))]
        return {"audit": audit, "row": row, "calls": calls, "partitions": partitions,
                "predictions": pd.read_csv(folder / "predictions.csv")}

    def assert_oof(self, call, audit, dataset, indices):
        frame = dataset["X"].iloc[indices]
        labels, row_ids = dataset["y"][indices], dataset["row_ids"][indices]
        seed_audit = audit["seed_scoring"]
        self.assertEqual(seed_audit["n_splits"], 3)
        self.assertEqual(seed_audit["synthetic_training_count"], 0)
        self.assertEqual(seed_audit["training_source"], "real_only")
        self.assertEqual(seed_audit["refresh"], "fixed_before_generation")
        np.testing.assert_array_equal(seed_audit["row_ids"], row_ids)
        self.assertEqual(len(call["scoring_fits"]), 3)
        self.assertEqual(len(call["oof"]), 3)
        probabilities = np.empty((len(indices), 2), dtype=np.float32)
        coverage = np.zeros(len(indices), dtype=int)
        splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(frame, labels)
        for (train, score), fold, fitted, partition in zip(
                splits, seed_audit["folds"], call["scoring_fits"], call["oof"]):
            transform = make_preprocessor(frame.iloc[train])
            X_train = transform.fit_transform(frame.iloc[train])
            X_score = transform.transform(frame.iloc[score])
            self.assertEqual(learned_state(fold["preprocessing"]),
                             learned_state(_preprocessing_state(transform)))
            np.testing.assert_array_equal(fitted["X"], X_train)
            np.testing.assert_array_equal(fitted["y"], labels[train])
            self.assertIsNone(fitted["validation"])
            np.testing.assert_array_equal(partition["X_train"], X_train)
            np.testing.assert_array_equal(partition["X_score"], X_score)
            np.testing.assert_array_equal(fold["train_row_ids"], row_ids[train])
            np.testing.assert_array_equal(fold["score_row_ids"], row_ids[score])
            self.assertFalse(set(fold["train_row_ids"]) & set(fold["score_row_ids"]))
            self.assertEqual(set(fold["train_row_ids"]) | set(fold["score_row_ids"]), set(row_ids))
            self.assertEqual(fold["train_count"], len(train))
            self.assertEqual(fold["score_count"], len(score))
            self.assertEqual(fold["synthetic_training_count"], 0)
            np.testing.assert_array_equal(np.asarray(seed_audit["fold_ids"])[score], fold["fold"])
            model = fit_estimator(make_estimator("logistic_regression", SEED,
                                                parameters=audit["parameters"], device="cpu"),
                                  X_train, labels[train])
            probabilities[score] = model.predict_proba(X_score)
            np.testing.assert_allclose(fitted["model"].predict_proba(X_score),
                                       probabilities[score], atol=1e-6)
            coverage[score] += 1
        np.testing.assert_array_equal(coverage, 1)
        np.testing.assert_allclose(seed_audit["probabilities"], probabilities, atol=1e-6)
        positions = np.searchsorted(seed_audit["classes"], labels)
        competing = probabilities.copy()
        competing[np.arange(len(labels)), positions] = -np.inf
        scores = competing.max(axis=1) - probabilities[np.arange(len(labels)), positions]
        np.testing.assert_allclose(seed_audit["scores"], scores, atol=1e-6)
        return scores

    def assert_groups_and_ancestry(self, call, audit, dataset, indices, scores):
        labels, row_ids = dataset["y"][indices], dataset["row_ids"][indices]
        support = audit["support_summary"]["group_support"]
        self.assertNotIn("constraint_support", audit["support_summary"])
        self.assertNotIn("stopping", audit["support_summary"])
        self.assertNotIn("stopping_selection", audit)
        self.assertEqual(support["configuration"]["grouping"], "kmeans")
        self.assertEqual(support["allocation"]["allocation"], "stability")
        self.assertEqual(support["allocation"]["group_alpha"], audit["sampler_parameters"]["group_alpha"])
        self.assertEqual(support["synthetic_training_count"], 0)
        np.testing.assert_array_equal(support["real_row_ids"], row_ids)
        self.assertFalse(set(row_ids) & set(dataset["row_ids"][self.split["test_indices"]]))
        if np.array_equal(indices, self.split["fit_indices"]):
            self.assertFalse(set(row_ids) & set(dataset["row_ids"][self.split["validation_indices"]]))
        expected = prepare_protected_groups(call["X"], labels, SEED,
                                            runner.augmentation_targets(labels, self.settings["augmentation"]),
                                            audit["sampler_parameters"])
        np.testing.assert_array_equal(support["row_group_ids"], expected["row_group_ids"])
        groups = {group["id"]: group for group in support["groups"]}
        for group, reference in zip(support["groups"], expected["groups"]):
            for key, value in reference.items():
                self.assertEqual(learned_state(group[key]), learned_state(value))
            np.testing.assert_array_equal(group["row_ids"], row_ids[group["indices"]])
            self.assertEqual(group["real_support"], len(group["row_ids"]))
            self.assertLessEqual(group["actual_generated"],
                                 int(np.floor(support["allocation"]["group_cap_multiplier"] * group["real_support"])))
        final_fit = next(item for item in call["fits"] if item["model"] is call["model"])
        for fitted in call["fits"]:
            np.testing.assert_array_equal(fitted["X"][:len(labels)], call["X"])
            np.testing.assert_array_equal(fitted["y"][:len(labels)], labels)
        generated = audit["augmentation"]["actual_count"]
        self.assertEqual(len(final_fit["y"]) - len(labels), generated)
        self.assertEqual(len(audit["provenance"]), generated)
        self.assertLessEqual(generated, audit["augmentation"]["requested_count"])
        self.assertEqual(audit["augmentation"]["underspend"],
                         audit["augmentation"]["requested_count"] - generated)
        for label in np.unique(labels):
            actual_count = int((final_fit["y"] == label).sum())
            self.assertEqual(actual_count, audit["augmentation"]["actual_counts"][str(label)])
            self.assertLessEqual(actual_count, audit["support_summary"]["target_counts"][str(label)])
        self.assertLessEqual(len(audit["rounds"]), self.settings["ugo_max_iterations"])
        seeds = Counter()
        for parent in audit["provenance"]:
            self.assertGreaterEqual(parent["row_index"], len(labels))
            self.assertLess(parent["row_index"], len(final_fit["y"]))
            self.assertEqual(final_fit["y"][parent["row_index"]], parent["class"])
            for key in ("seed_index", "partner_index", "relocation_index"):
                index = parent.get(key)
                if index is None or (key == "relocation_index" and index == -1):
                    continue
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(labels))
                self.assertEqual(labels[index], parent["class"])
            seed = parent["seed_index"]
            seeds[seed] += 1
            self.assertEqual(parent["group_id"], support["row_group_ids"][seed])
            self.assertTrue(groups[parent["group_id"]]["supported"])
        self.assertEqual(support["seed_multiplicity"], {str(seed): count for seed, count in seeds.items()})
        self.assertTrue(all(count <= support["allocation"]["seed_cap"] for count in seeds.values()))
        for group in groups.values():
            self.assertEqual(group["actual_generated"], sum(seeds[index] for index in group["indices"]))
        self.assertTrue(support["allocation_records"])
        for record in support["allocation_records"]:
            class_indices = np.flatnonzero(labels == record["target_class"])
            percentiles = np.zeros(len(labels))
            ranks = rankdata(scores[class_indices].astype(float), method="average")
            percentiles[class_indices] = (ranks - .5) / len(class_indices)
            self.assertLessEqual(record["allocated"], record["class_remaining"])
            self.assertEqual(sum(trace["quota"] for trace in record["group_trace"]), record["allocated"])
            for trace in record["group_trace"]:
                group = groups[trace["group_id"]]
                h = float(percentiles[group["indices"]].mean())
                self.assertAlmostEqual(trace["h"], h, places=12)
                if group["supported"]:
                    allocation = support["allocation"]
                    weight = group["stability"] * (h + allocation["epsilon"]) / (
                        group["real_support"] + allocation["epsilon"]) ** allocation["group_alpha"]
                    self.assertAlmostEqual(trace["weight"], weight, places=12)
                self.assertLessEqual(trace["quota"], trace["capacity"])
                self.assertLessEqual(trace["used_group"] + trace["quota"],
                                     int(np.floor(support["allocation"]["group_cap_multiplier"] * trace["n_real"])))
        return final_fit

    def assert_joint_selection(self, snapshot, dataset):
        audit, calls, partitions = snapshot["audit"], snapshot["calls"], snapshot["partitions"]
        fit, validation, train, test = (self.split[key] for key in
                                      ("fit_indices", "validation_indices", "train_indices", "test_indices"))
        self.assertEqual(snapshot["row"]["status"], "completed")
        self.assertEqual(len(calls), 11)
        self.assertEqual(audit["selection"]["candidate_configs"], self.candidates)
        self.assertEqual([item["sampler_parameters"]["group_alpha"] for item in audit["candidates"]],
                         [ALPHAS[index % 3] for index in range(10)])
        self.assertEqual(Counter(item["sampler_parameters"]["group_alpha"] for item in audit["candidates"]),
                         {0.: 4, .5: 3, 1.: 3})
        self.assertEqual(audit["reference_rows"]["inner_real_row_ids"], dataset["row_ids"][fit].tolist())
        self.assertEqual(audit["reference_rows"]["outer_real_row_ids"], dataset["row_ids"][train].tolist())
        aps, final_fits = [], []
        for candidate_id, (candidate, call) in enumerate(zip(audit["candidates"], calls)):
            self.assertEqual(candidate["parameters"], self.candidates[candidate_id])
            self.assertEqual(candidate["threshold"], .5)
            np.testing.assert_array_equal(call["X"], partitions["X_fit"])
            np.testing.assert_array_equal(call["y"], dataset["y"][fit])
            scores = self.assert_oof(call, candidate, dataset, fit)
            final_fits.append(self.assert_groups_and_ancestry(call, candidate, dataset, fit, scores))
            probabilities = call["model"].predict_proba(partitions["X_validation"])
            ap = average_precision_score(dataset["y"][validation], probabilities[:, 1])
            self.assertAlmostEqual(candidate["metrics"]["primary_metric_value"], ap, places=12)
            aps.append(ap)
        selected_id = int(np.argmax(aps))
        self.assertGreater(len(set(aps)), 1)
        self.assertEqual(audit["selection"]["selected_candidate"], selected_id)
        self.assertEqual(snapshot["row"]["selected_candidate"], selected_id)
        inner = audit["selection"]["inner"]
        self.assertEqual(inner["parameters"], self.candidates[selected_id])
        self.assertEqual(inner["sampler_parameters"], audit["candidates"][selected_id]["sampler_parameters"])
        outer, call = audit["outer"], calls[-1]
        self.assertEqual(outer["parameters"], inner["parameters"])
        self.assertEqual(outer["sampler_parameters"], inner["sampler_parameters"])
        self.assertEqual(snapshot["row"]["config"]["selected_parameters"], inner["parameters"])
        np.testing.assert_array_equal(call["X"], partitions["X_train"])
        np.testing.assert_array_equal(call["y"], dataset["y"][train])
        self.assertTrue(all(item["validation"] is None for item in call["fits"]))
        scores = self.assert_oof(call, outer, dataset, train)
        outer_fit = self.assert_groups_and_ancestry(call, outer, dataset, train, scores)
        self.assertGreater(inner["augmentation"]["actual_count"], 0)
        self.assertGreater(outer["augmentation"]["actual_count"], 0)
        for fitted, X_predict, actual in (
                (final_fits[selected_id], partitions["X_validation"], calls[selected_id]["model"]),
                (outer_fit, partitions["X_test"], call["model"])):
            independent = fit_estimator(make_estimator("logistic_regression", SEED,
                                                       parameters=inner["parameters"], device="cpu"),
                                        fitted["X"], fitted["y"])
            np.testing.assert_allclose(independent.predict_proba(X_predict), actual.predict_proba(X_predict), atol=1e-7)
        predictions = snapshot["predictions"]
        self.assertEqual(audit["selection"]["threshold"], .5)
        self.assertEqual(snapshot["row"]["threshold"], .5)
        np.testing.assert_array_equal(predictions["row_id"], dataset["row_ids"][test])
        np.testing.assert_array_equal(predictions["y_true"], dataset["y"][test])
        probabilities = call["model"].predict_proba(partitions["X_test"])
        np.testing.assert_allclose(predictions[["probability_0", "probability_1"]], probabilities, atol=1e-7)
        np.testing.assert_array_equal(predictions["y_pred"], (probabilities[:, 1] >= .5).astype(int))

    def test_joint_real_oof_groups_selection_and_outer_refit_ignore_test_perturbations(self):
        changed = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        test = self.split["test_indices"]
        changed["X"].loc[test, "Administrative_Duration"] = 1e12
        changed["X"].loc[test, "VisitorType"] = "test-only-category"
        changed["y"][test] = 1 - changed["y"][test]
        with tempfile.TemporaryDirectory() as root:
            before = self.run_cell(Path(root) / "original", self.data)
            after = self.run_cell(Path(root) / "changed", changed)
        self.assert_joint_selection(before, self.data)
        self.assert_joint_selection(after, changed)
        self.assertEqual(learned_state(before["audit"]), learned_state(after["audit"]))
        for role in ("X_fit", "X_validation", "X_train"):
            np.testing.assert_array_equal(before["partitions"][role], after["partitions"][role])
        self.assertFalse(np.array_equal(before["partitions"]["X_test"], after["partitions"]["X_test"]))
        for original, perturbed in zip(before["calls"], after["calls"]):
            for left, right in zip(original["fits"], perturbed["fits"]):
                np.testing.assert_array_equal(left["X"], right["X"])
                np.testing.assert_array_equal(left["y"], right["y"])
                np.testing.assert_array_equal(left["model"].coef_, right["model"].coef_)
                np.testing.assert_array_equal(left["model"].intercept_, right["model"].intercept_)

    def test_combination_preserves_shared_real_oof_and_group_learned_state(self):
        shared = runner.prepare_partitions(self.data, self.split)
        candidates = [{"C": .1}]
        with tempfile.TemporaryDirectory() as root:
            u1 = self.run_cell(Path(root) / "u1-before", self.data, method="ugo_oof_mma",
                               candidates=candidates, partitions=shared)
            u2 = self.run_cell(Path(root) / "u2-before", self.data, method="ugo_groups_only",
                               candidates=candidates, partitions=shared)
            learned = learned_state({key: shared[key] for key in ("inner_oof", "outer_oof", "protected_groups")})
            combined = self.run_cell(Path(root) / "combined", self.data, candidates=candidates, partitions=shared)
            self.assertEqual(learned, learned_state({key: shared[key] for key in
                                                   ("inner_oof", "outer_oof", "protected_groups")}))
            u1_after = self.run_cell(Path(root) / "u1-after", self.data, method="ugo_oof_mma",
                                     candidates=candidates, partitions=shared)
            u2_after = self.run_cell(Path(root) / "u2-after", self.data, method="ugo_groups_only",
                                     candidates=candidates, partitions=shared)
        self.assertEqual(learned_state(u1["audit"]), learned_state(u1_after["audit"]))
        self.assertEqual(learned_state(u2["audit"]), learned_state(u2_after["audit"]))
        for call, audit, role in zip(combined["calls"],
                                     [combined["audit"]["candidates"][0], combined["audit"]["outer"]],
                                     ("fit_indices", "train_indices")):
            scores = self.assert_oof(call, audit, self.data, self.split[role])
            self.assert_groups_and_ancestry(call, audit, self.data, self.split[role], scores)
        for name in ("candidates", "outer"):
            before = u1["audit"][name][0] if name == "candidates" else u1["audit"][name]
            actual = combined["audit"][name][0] if name == "candidates" else combined["audit"][name]
            self.assertEqual(before["seed_scoring"], actual["seed_scoring"])
            before = u2["audit"][name][0] if name == "candidates" else u2["audit"][name]
            for key in ("configuration", "real_row_ids", "row_group_ids"):
                self.assertEqual(before["support_summary"]["group_support"][key],
                                 actual["support_summary"]["group_support"][key])


class U5UserGateTests(unittest.TestCase):
    def test_discarded_u4_opens_experiment_not_retention_or_confirmatory_gate(self):
        study = copy.deepcopy(load_study(ROOT / "configs/ablations/stage1_components.yaml"))
        study["manual_review"]["decisions"] = {"U1": "keep", "U2": "keep", "U3": "discard",
                                               "U4": "discard", "U5": "pending"}
        study["variants"]["U5"].update(available=True, missing_components=[],
                                         requires_user_decisions=["U1", "U2", "U3", "U4"])
        study["freeze"] = {"status": "not_frozen", "evidence": {}}
        study["stage_gate"]["stage2_allowed"] = False
        before = copy.deepcopy(study)
        require_variant(study, "U5")
        enforce_stage_gate(study, "development")
        with self.assertRaises(ValueError):
            enforce_stage_gate(study, "confirmatory")
        self.assertEqual(study, before)
        for component in ("U1", "U2", "U4"):
            undecided = copy.deepcopy(study)
            undecided["manual_review"]["decisions"][component] = "pending"
            with self.subTest(component=component), self.assertRaises(ValueError):
                require_variant(undecided, "U5")
        premature = copy.deepcopy(study)
        premature["freeze"]["status"] = "frozen"
        with self.assertRaises(ValueError):
            enforce_stage_gate(premature, "confirmatory")


if __name__ == "__main__":
    unittest.main()
