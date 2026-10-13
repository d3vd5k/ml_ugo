"""Native random selection and post-choice F1 on real Shoppers and analytic pools."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import prepare_partitions
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from samplers.flow_gmm import FlowGate
from samplers.flow_support import CalibratedFlowGate
from samplers.learning_benefit import CandidatePoolSelector, _coordinate_key, model_state_hash
from samplers.pool_random import RandomPoolSelector
from samplers.protected import prepare_protected_groups, protected_ugo
from samplers.ugo import Candidates, GroupBudget, UGOConfig, accept_relocation, build_support, propose_interpolation, score_seeds

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    X = np.array([[-3., 1.], [-2.8, 0.], [-2., -.5], [-1.5, .2],
                  [1.2, .5], [1.8, -.5], [2.4, .2], [3., 1.]], dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cfg = UGOConfig(k_pre=2, alpha=1., device="cpu")
    real = build_support(torch.from_numpy(X), torch.from_numpy(y), cfg)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    plan = GroupBudget(1, torch.tensor([6, 4, 6, 5, 7, 4]), 6)
    record = {"requested": 6, "group_trace": [
        {"group_id": 10, "quota": 1, "used_seed": {4: 1, 5: 1}},
        {"group_id": 11, "quota": 2, "used_seed": {6: 1, 7: 1}}]}
    return X, y, cfg, real, model, plan, record


class RandomPoolAnalyticTests(unittest.TestCase):
    def test_exact_first_occurrence_expansion_and_historical_random_parity(self):
        X, y, cfg, real, model, plan, record = fixture()
        new = RandomPoolSelector(19, k=5)
        old = CandidatePoolSelector("random", X, y, 19, k=5, verification_size=0)
        for round in range(2):
            new.score_seeds(model, real.X, real.y)
            old.score_seeds(model, real.X, real.y)
            pool = new.propose(real, plan, np.random.default_rng(77 + round), cfg)
            historical = old.propose(real, plan, np.random.default_rng(77 + round), cfg)
            expected = propose_interpolation(real, plan, np.random.default_rng(77 + round), cfg,
                seed_indices=torch.tensor(np.repeat([6, 4, 5, 7], 5)))
            torch.testing.assert_close(pool.X, expected.X, rtol=0, atol=0)
            torch.testing.assert_close(pool.X, historical.X, rtol=0, atol=0)
            chosen = new.select(real, accept_relocation(real, pool, cfg), cfg, record)
            previous = old.select(real, accept_relocation(real, historical, cfg), cfg, record)
            new.finalize_acceptance(np.ones(len(chosen.X), dtype=bool))
            for key in ("X", "y", "seed_indices", "partner_indices", "relocation_indices"):
                torch.testing.assert_close(getattr(chosen, key), getattr(previous, key), rtol=0, atol=0)
            np.testing.assert_array_equal(new.rounds[-1]["selected"], old.rounds[-1]["selected"])
        json.dumps(new.summary(), allow_nan=False)

    def test_first_owner_signed_zero_finite_real_matches_and_group_caps(self):
        _, _, cfg, real, model, plan, record = fixture()
        selector = RandomPoolSelector(3, k=2)
        selector.score_seeds(model, real.X, real.y)
        generated = selector.propose(real, plan, np.random.default_rng(2), cfg)
        # Anchors [6,6,4,4,5,5,7,7]: zero owned by 6, and real row owned by 4.
        points = torch.tensor([[0., -0.], [-0., 0.], real.X[4].tolist(), real.X[4].tolist(),
                               [float("inf"), 1.], [8., 2.], [9., 3.], [10., 4.]])
        pool = Candidates(points, generated.y, generated.seed_indices, generated.partner_indices,
                          generated.relocation_indices)
        selected = selector.select(real, pool, cfg, record)
        row = selector.rounds[-1]
        np.testing.assert_array_equal(row["keep"], [1, 0, 1, 0, 0, 1, 1, 1])
        self.assertEqual(row["rejection_reasons"][1], "pool_duplicate")
        self.assertEqual(row["rejection_reasons"][4], "invalid_features")
        self.assertTrue(row["real_matches"][2])
        self.assertEqual(row["attempted"], 3)
        self.assertEqual(len(np.unique(selected.seed_indices)), 3)
        self.assertEqual(np.sum(row["attempted_selected"] & (row["group_ids"] == 10)), 1)
        self.assertEqual(np.sum(row["attempted_selected"] & (row["group_ids"] == 11)), 2)
        selector.finalize_acceptance(np.ones(3, dtype=bool))
        self.assertEqual(row["group_shortfalls"], {10: 0, 11: 0})

    def test_rejects_never_enter_dedup_history_or_promote_other_k_choices(self):
        _, _, cfg, real, model, plan, record = fixture()
        selector = RandomPoolSelector(19, k=5)
        selector.score_seeds(model, real.X, real.y)
        pool = selector.propose(real, plan, np.random.default_rng(17), cfg)
        chosen = selector.select(real, pool, cfg, record)
        row = selector.rounds[-1]
        attempts = row["attempted_selected"].copy()
        keep = np.array([True, False, True])
        scores = {"valid": np.ones(3, dtype=bool), "accepted": keep,
            "reasons": np.where(keep, "accepted", "low_ratio"), "log_ratio": np.array([2., -2., 3.]),
            "log_likelihood_intended": np.array([-1., -5., -1.]),
            "log_likelihood_opposing": np.array([-3., -3., -4.])}
        selector.finalize_acceptance(keep, scores)
        self.assertEqual(row["accepted"], 2)
        self.assertEqual(row["attempted"], 3)
        np.testing.assert_array_equal(row["attempted_selected"], attempts)
        np.testing.assert_array_equal(np.flatnonzero(row["selected"]), np.flatnonzero(attempts)[keep])
        self.assertEqual(len(selector._accepted_coordinates), 2)
        rejected = chosen.X[1].numpy()
        self.assertNotIn(_coordinate_key(rejected), selector._accepted_coordinates)
        selector.score_seeds(model, real.X, real.y)
        next_pool = selector.propose(real, plan, np.random.default_rng(18), cfg)
        # Replay the very same coordinates: only the two genuinely accepted owners disappear.
        next_pool = Candidates(pool.X.clone(), next_pool.y, next_pool.seed_indices,
                               next_pool.partner_indices, next_pool.relocation_indices)
        selector.select(real, next_pool, cfg, record)
        second = selector.rounds[-1]
        self.assertTrue(second["keep"][np.flatnonzero(attempts)[1]])
        self.assertEqual(second["accumulated_duplicate_count"], 2)
        selector.finalize_acceptance(np.ones(second["attempted"], dtype=bool))

    def test_native_state_is_frozen_across_generation_and_selection(self):
        _, _, cfg, real, model, plan, record = fixture()
        selector = RandomPoolSelector(9)
        selector.score_seeds(model, real.X, real.y)
        pool = selector.propose(real, plan, np.random.default_rng(3), cfg)
        model.coef_[0, 0] += 1.
        with self.assertRaises(RuntimeError):
            selector.select(real, pool, cfg, record)


class ShoppersRandomPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        original = load_dataset(protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(original["y"])), train_size=400,
                                    stratify=original["y"], random_state=19)
        cls.data = {"name": "shoppers", "X": original["X"].iloc[chosen].reset_index(drop=True),
                    "y": original["y"][chosen], "row_ids": original["row_ids"][chosen]}
        train, test = train_test_split(np.arange(400), test_size=.2, stratify=cls.data["y"], random_state=19)
        fit, validation = train_test_split(train, test_size=.25, stratify=cls.data["y"][train], random_state=19)
        cls.split = {"fold": 0, "train_indices": train, "fit_indices": fit,
                     "validation_indices": validation, "test_indices": test}
        cls.partitions = prepare_partitions(cls.data, cls.split)
        cls.X, cls.y = cls.partitions["X_fit"], cls.data["y"][fit]
        cls.categorical = categorical_columns(cls.partitions["inner"])
        cls.params = {"grouping": "kmeans", "allocation": "stability", "alpha": 1., "group_alpha": .5,
                      "n_clusters": 3, "repeats": 3, "seed_cap": 2, "lambda_step": .04, "flow_epochs": 3}
        cls.targets = {int(c): int(np.sum(cls.y == c)) + (20 if c == 1 else 0) for c in np.unique(cls.y)}
        cls.prepared = prepare_protected_groups(cls.X, cls.y, 19, cls.targets, cls.params)
        cls.prepared["real_row_ids"] = cls.data["row_ids"][fit].copy()
        cls.gate = flow_for_partition(cls.data, cls.split, cls.partitions, "inner", 19, cls.params, "cpu")

    def learner(self, algorithm):
        parameters = ({"C": .7} if algorithm == "logistic_regression" else
                      {"n_estimators": 12, "max_depth": 3, "min_samples_leaf": 3} if algorithm == "random_forest" else
                      {"n_estimators": 12, "num_leaves": 7, "min_child_samples": 5})
        estimator = make_estimator(algorithm, 19, parameters=parameters, device="cpu")
        return lambda X, y: fit_estimator(estimator, X, y)

    def run_pool(self, selector, *, algorithm="logistic_regression", gate=None, rounds=20, parameters=None):
        return protected_ugo(self.X, self.y, 19, self.categorical, rounds, "cpu", self.learner(algorithm),
            self.targets, {**self.params, **(parameters or {})}, self.prepared,
            candidate_selector=selector, flow_support=gate)

    def test_lr_rf_lightgbm_actual_native_scoring_freeze_and_checkpoint_callback(self):
        for algorithm in ("logistic_regression", "random_forest", "lightgbm"):
            with self.subTest(algorithm=algorithm):
                model = self.learner(algorithm)(self.X, self.y)
                before = model_state_hash(model)
                captured = []
                selector = RandomPoolSelector(19, checkpoint_callback=lambda native, checkpoint: captured.append((native, checkpoint)))
                real = build_support(torch.from_numpy(self.X), torch.from_numpy(self.y), UGOConfig(alpha=1., device="cpu"))
                expected = score_seeds(model, real.X, real.y)
                actual = selector.score_seeds(model, real.X, real.y)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertIs(captured[0][0], model)
                self.assertEqual(captured[0][1]["round"], 1)
                self.assertEqual(before, model_state_hash(model))
                result = self.run_pool(RandomPoolSelector(19), algorithm=algorithm, rounds=2)
                self.assertGreater(len(result.y), len(self.y))
                for row in result.support_summary["candidate_pool"]["rounds"]:
                    self.assertEqual(row["state_before_hash"], row["state_after_hash"])
                filtered = self.run_pool(RandomPoolSelector(19), algorithm=algorithm, gate=self.gate, rounds=2)
                self.assertLessEqual(filtered.support_summary["proposal_budget"]["attempted"], 20)
                self.assertEqual(filtered.support_summary["proposal_budget"]["accepted"], len(filtered.y) - len(self.y))

    def test_native_plain_random_matches_historical_shoppers_training_prefixes(self):
        native, historical = RandomPoolSelector(19), CandidatePoolSelector("random", self.X, self.y, 19, verification_size=0)
        actual = self.run_pool(native)
        previous = self.run_pool(historical)
        np.testing.assert_array_equal(actual.X, previous.X)
        np.testing.assert_array_equal(actual.y, previous.y)
        self.assertEqual(actual.provenance, previous.provenance)
        self.assertIn("candidate_pool", actual.support_summary)
        self.assertNotIn("learning_benefit", actual.support_summary)
        self.assertIn("learning_benefit", previous.support_summary)
        for pool in native.rounds:
            self.assertLessEqual(pool["attempted"], len(np.unique(pool["planned_anchor_indices"])))
        multiplicities = actual.support_summary["group_support"]["seed_multiplicity"]
        self.assertTrue(all(count <= 2 for count in multiplicities.values()))
        json.dumps(native.summary(), allow_nan=False)

    def test_f1_replaces_relocation_scores_only_attempts_and_preserves_full_pool_lineage(self):
        selector = RandomPoolSelector(19)
        before = {key: value.clone() for key, value in self.gate.core.model.state_dict().items()}
        inference_before = self.gate.inference_rows
        with patch("samplers.protected.accept_relocation", side_effect=AssertionError("F1 must replace relocation")):
            result = self.run_pool(selector, gate=self.gate, parameters={"f1_tau": .5})
        support = result.support_summary
        self.assertEqual(self.gate.inference_rows - inference_before, sum(pool["attempted"] for pool in selector.rounds))
        attempts, accepted = [], []
        for pool in selector.rounds:
            np.testing.assert_array_equal(pool["original_coordinates"], pool["final_coordinates"])
            self.assertTrue(np.all(pool["relocation_indices"] == -1))
            subset = pool["attempted_selected"]
            predictions = self.gate.predict(pool["original_coordinates"][subset], pool["labels"][subset], .5)
            np.testing.assert_array_equal(pool["selected"][subset], predictions["accepted"])
            np.testing.assert_allclose(pool["f1_log_ratio"], predictions["log_ratio"], rtol=0, atol=0)
            self.assertFalse(pool["selected"][~subset].any())
            attempts.extend(pool["proposal_ids"][subset].tolist())
            accepted.extend(pool["proposal_ids"][pool["selected"]].tolist())
            for group, quota in pool["group_quotas"].items():
                self.assertLessEqual(np.sum(subset & (pool["group_ids"] == group)), quota)
        self.assertEqual([row["proposal_id"] for row in support["candidate_log"]], attempts)
        self.assertEqual([row["proposal_id"] for row in result.provenance], accepted)
        self.assertEqual(support["proposal_budget"]["attempted"], len(attempts))
        self.assertLessEqual(len(attempts), 20)
        by_id = {row["proposal_id"]: row for row in support["candidate_log"]}
        for ancestry in result.provenance:
            row = by_id[ancestry["proposal_id"]]
            self.assertTrue(row["accepted"])
            self.assertEqual(ancestry["seed_index"], row["seed_index"])
            self.assertEqual(ancestry["partner_index"], row["partner_index"])
            np.testing.assert_array_equal(result.X[ancestry["row_index"]], row["original_coordinates"])
            self.assertLess(row["seed_index"], len(self.y))
            self.assertLess(row["partner_index"], len(self.y))
        for key, value in self.gate.core.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_actual_flow_all_rejected_consumes_attempts_and_never_regenerates(self):
        core = deepcopy(self.gate.core)
        with torch.no_grad():
            core.model.means[1].fill_(1e6)
            core.model.log_stds.zero_()
        gate = FlowGate(core, self.partitions["inner"], calibration_row_ids=self.gate.calibration_row_ids,
                        validation_row_ids=self.gate.validation_row_ids)
        selector = RandomPoolSelector(19)
        prefixes = []
        train = self.learner("logistic_regression")
        def fit(X, y):
            prefixes.append(len(y))
            return train(X, y)
        result = protected_ugo(self.X, self.y, 19, self.categorical, 20, "cpu", fit,
            self.targets, {**self.params, "lambda_step": 1., "f1_tau": .8}, self.prepared,
            candidate_selector=selector, flow_support=gate)
        self.assertEqual(prefixes, [len(self.y)])
        self.assertEqual(len(selector.rounds), 1)
        self.assertEqual(result.stop_reason, "f1_no_candidates_accepted")
        np.testing.assert_array_equal(result.X, self.X)
        row = selector.rounds[0]
        self.assertEqual(len(row["original_coordinates"]), 5 * len(np.unique(row["planned_anchor_indices"])))
        self.assertFalse(row["selected"].any())
        self.assertEqual(len(selector._accepted_coordinates), 0)
        self.assertEqual(gate.inference_rows, row["attempted"])
        self.assertEqual(result.support_summary["proposal_budget"]["attempted"], row["attempted"])
        self.assertEqual(result.support_summary["proposal_budget"]["rejected"], row["attempted"])

    def test_real_partial_flow_rejection_keeps_random_choices_and_exhausts_attempt_cap(self):
        pilot = RandomPoolSelector(19)
        self.run_pool(pilot, gate=self.gate, rounds=1, parameters={"f1_tau": .5})
        scores = np.unique(pilot.rounds[0]["f1_compatibility_score"][pilot.rounds[0]["f1_valid"]])
        self.assertGreater(len(scores), 1)
        middle = len(scores) // 2
        tau = float((scores[middle - 1] + scores[middle]) / 2)
        selector = RandomPoolSelector(19)
        result = self.run_pool(selector, gate=self.gate, parameters={"f1_tau": tau})
        first = selector.rounds[0]
        np.testing.assert_array_equal(first["attempted_selected"], pilot.rounds[0]["attempted_selected"])
        self.assertGreater(first["accepted"], 0)
        self.assertGreater(first["rejected"], 0)
        self.assertEqual(first["attempted"], first["accepted"] + first["rejected"])
        self.assertFalse(first["selected"][~first["attempted_selected"]].any())
        budget = result.support_summary["proposal_budget"]
        self.assertEqual(budget["attempted"], 20)
        self.assertEqual(result.stop_reason, "proposal_cap_reached")
        self.assertEqual(budget["accepted"], sum(row["accepted"] for row in selector.rounds))
        self.assertTrue(all(count <= 2 for count in budget["attempted_seed_multiplicity"]))

    def test_group_capacity_is_binding_on_actual_shoppers_path(self):
        selector = RandomPoolSelector(19)
        result = self.run_pool(selector, parameters={"group_cap_multiplier": .25})
        group_rows = result.support_summary["group_support"]["groups"]
        generated = 0
        for group in group_rows:
            cap = int(np.floor(.25 * group["real_support"]))
            self.assertLessEqual(group["actual_generated"], cap)
            generated += group["actual_generated"]
        self.assertEqual(generated, len(result.y) - len(self.y))
        self.assertLess(generated, 20)
        self.assertTrue(any("group_cap_exhausted" in trace["exclusions"]
            for record in result.support_summary["group_support"]["allocation_records"]
            for trace in record["group_trace"]))

    def test_incompatible_handlers_rejected_without_changing_old_utility_contract(self):
        from samplers.flow_support import calibrate_flow
        calibration_indices = np.flatnonzero(np.isin(self.data["row_ids"], self.gate.calibration_row_ids))
        calibration = calibrate_flow(self.gate.core, self.data["X"].iloc[calibration_indices],
            self.data["y"][calibration_indices], self.data["row_ids"][calibration_indices])
        for selector, handler in ((CandidatePoolSelector("random", self.X, self.y, 19), self.gate),
                                  (RandomPoolSelector(19), CalibratedFlowGate(self.gate, calibration))):
            with self.assertRaises(ValueError):
                self.run_pool(selector, gate=handler)
        for keyword in ("h1_support", "reject_neighbors", "seed_score"):
            with self.assertRaises(ValueError):
                protected_ugo(self.X, self.y, 19, self.categorical, 1, "cpu", self.learner("logistic_regression"),
                    self.targets, self.params, self.prepared, candidate_selector=RandomPoolSelector(19),
                    **{keyword: True})
        for key in ("stop", "policy", "budget_fraction", "budget_fractions"):
            with self.assertRaises(ValueError):
                self.run_pool(RandomPoolSelector(19), parameters={key: object()})


if __name__ == "__main__":
    unittest.main()
