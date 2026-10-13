import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import f1_score

from benchmarks.results import (BenchmarkArtifacts, choose_threshold, enforce_stage_gate,
                                load_study, metric_results, paired_summary, read_records,
                                require_variant, snapshot_sha256, sos_weighted_f1_summary,
                                summarize_records)
from experiments import RunArtifacts


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return {"protocol_id": "isolated_controlled_v1", "dataset": "isolated",
            "primary_metric": "average_precision", "threshold": {"name": "fixed_0.5"},
            "tuning_budget": 1, "status": "resolved"}


def study():
    return load_study(ROOT / "configs/ablations/stage1_controls.yaml")


def cell(**changes):
    row = {"dataset": "isolated", "dataset_fingerprint": "dataset-sha",
           "split_fingerprint": "split-sha", "seed": 42, "algorithm": "logistic_regression",
           "method": "baseline", "variant_id": "baseline", "budget_mode": "matched",
           "config": {"C": 1}, "tuning_budget": 1, "candidate_configs": [{"C": 1}],
           "augmentation": {"mode": "matched", "requested_count": 0, "actual_count": 0,
                            "actual_counts": {0: 20, 1: 5}, "underspend": 0},
           "status": "completed", "average_precision": .7}
    row.update(changes)
    return row


class ThresholdTests(unittest.TestCase):
    def test_cost_minimizes_validation_cost_over_all_attainable_thresholds(self):
        y, scores = np.array([0, 0, 0, 1, 1]), np.array([.8, .2, .2, .4, .2])
        threshold = choose_threshold(y, scores, "aps_cost")
        candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
        costs = [10 * np.sum((scores >= t) & (y == 0)) + 500 * np.sum((scores < t) & (y == 1)) for t in candidates]
        achieved = 10 * np.sum((scores >= threshold) & (y == 0)) + 500 * np.sum((scores < threshold) & (y == 1))
        self.assertEqual(achieved, min(costs))
        self.assertEqual(threshold, .2)

    def test_fpr_ties_are_not_split_or_interpolated(self):
        labels = np.r_[np.zeros(20), np.ones(2)]
        scores = np.r_[np.full(2, .9), np.full(18, .1), [.9, .8]]
        threshold = choose_threshold(labels, scores, {"name": "recall_at_fpr", "fpr": .05})
        self.assertGreater(threshold, .9)
        self.assertFalse(np.any(scores >= threshold))
        metrics = metric_results(labels, np.c_[1 - scores, scores], [0, 1], 1, .8, "recall_at_fpr")
        self.assertEqual(metrics["recall_at_fpr_test"], 0)
        self.assertEqual(metrics["realized_recall_test"], 1)
        self.assertEqual(metrics["realized_fpr_test"], .1)

    def test_fpr_selects_largest_tpr_and_highest_threshold_on_plateau(self):
        labels = np.r_[np.zeros(20), np.ones(2)]
        scores = np.r_[.95, np.full(19, .1), [.9, .8]]
        self.assertEqual(choose_threshold(labels, scores, "fpr"), .8)
        self.assertEqual(choose_threshold([0, 1], [1, 1], "fpr"), np.nextafter(1., np.inf))
        self.assertEqual(choose_threshold([0, 1], [.1, .9], "default"), .5)

    def test_invalid_selection_inputs_refused(self):
        for labels, scores in (([1, 1], [.2, .3]), ([0, 1], [np.nan, .2]), ([0, 1], [1.1, .2])):
            with self.assertRaises(ValueError):
                choose_threshold(labels, scores, "cost")
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [.2, .3], "outer_optimal")

    def test_positive_column_order_and_challenge_cost(self):
        labels = np.array([0, 0, 1, 1])
        ppositive = np.array([.6, .1, .8, .3])
        metrics = metric_results(labels, np.c_[ppositive, 1 - ppositive], [1, 0], 1, .5, "aps_cost")
        self.assertEqual((metrics["fp"], metrics["fn"], metrics["cost"]), (1, 1, 510))
        self.assertEqual(metrics["primary_metric_value"], 510)
        self.assertEqual(metrics["minority_f1"], .5)

    def test_multiclass_and_sos_inverse_support_metric_are_separate(self):
        labels = np.array([0, 0, 0, 1, 2])
        prediction = np.array([0, 0, 1, 1, 1])
        probabilities = np.eye(3)[prediction]
        metrics = metric_results(labels, probabilities, [0, 1, 2], 2, None, "weighted_f1")
        class_f1 = f1_score(labels, prediction, labels=[0, 1, 2], average=None, zero_division=0)
        self.assertAlmostEqual(metrics["sos_inverse_support_f1"], np.dot([.2, .4, .4], class_f1))
        self.assertNotEqual(metrics["weighted_f1"], metrics["sos_inverse_support_f1"])
        self.assertIsNone(metrics["cost"])
        self.assertIsNone(metrics["recall_at_fpr_test"])


class ArtifactTests(unittest.TestCase):
    def test_snapshot_and_indexes_reuse_run_registry(self):
        p, s = protocol(), study()
        with tempfile.TemporaryDirectory() as output:
            artifacts = BenchmarkArtifacts(output, {"experiment_type": "setup_smoke", "random_state": 42}, p, s)
            self.assertIsInstance(artifacts, RunArtifacts)
            p["primary_metric"] = "cost"
            s["freeze"]["status"] = "frozen"
            returned_protocol = artifacts.protocol
            returned_protocol["threshold"]["name"] = "cost"
            records = artifacts.finish([cell()])
            manifest = json.loads((artifacts.path / "manifest.json").read_text())
            self.assertEqual(manifest["protocol_snapshot"]["primary_metric"], "average_precision")
            self.assertEqual(artifacts.protocol["threshold"]["name"], "fixed_0.5")
            self.assertEqual(snapshot_sha256(manifest["protocol_snapshot"]), manifest["protocol_sha256"])
            index = read_records(Path(output) / "benchmark_runs.csv")
            with (Path(output) / "experiment_registry.csv").open(newline="") as handle:
                registry = list(csv.DictReader(handle))
            self.assertEqual(index[0]["run_id"], registry[0]["run_id"])
            self.assertEqual(index[0]["run_id"], artifacts.run_id)
            self.assertEqual(index[0]["record_id"], records[0]["record_id"])
            self.assertEqual(index[0]["artifact_path"], str(artifacts.path.resolve()))
            self.assertEqual(json.loads(registry[0]["metrics"])["record_id"], index[0]["record_id"])
            self.assertEqual(summarize_records(index), [])
            self.assertEqual(len(summarize_records(index, include_smoke=True)), 1)
            with self.assertRaises(RuntimeError):
                artifacts.finish([cell()])

    def test_partial_failure_preserves_completed_cells_and_no_failed_metrics(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = BenchmarkArtifacts(output, {"experiment_type": "setup_smoke"}, protocol(), study())
            records = artifacts.finish([cell(), cell(method="smote", variant_id="smote", status="failed", error="too few neighbors", metrics={"weighted_f1": .99})], error="run failed")
            self.assertEqual(records[0]["status"], "completed")
            self.assertEqual(records[0]["metrics"]["average_precision"], .7)
            self.assertEqual(records[1]["status"], "failed")
            self.assertEqual(records[1]["metrics"], {})
            self.assertNotIn("average_precision", records[1])
            self.assertEqual(artifacts.manifest["status"], "failed")

    def test_prepared_and_acquisition_failure_have_explicit_records(self):
        with tempfile.TemporaryDirectory() as output:
            prepared = BenchmarkArtifacts(output, {"experiment_type": "setup_smoke"}, protocol(), study())
            records = prepared.finish([cell(status="prepared")], status="prepared")
            self.assertEqual(records[0]["metrics"], {})
            self.assertEqual(prepared.manifest["status"], "prepared")
            failed = BenchmarkArtifacts(output, {"experiment_type": "setup_smoke"}, protocol(), study())
            records = failed.finish([], error="source unavailable")
            self.assertEqual(records[0]["record_scope"], "run_setup")
            self.assertEqual(records[0]["dataset_fingerprint"], "unavailable:not_loaded")
            self.assertEqual(records[0]["error"], "source unavailable")
            self.assertEqual(len(read_records(Path(output) / "benchmark_runs.csv")), 2)

    def test_required_identity_and_failure_error(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = BenchmarkArtifacts(output, {}, protocol(), study())
            with self.assertRaises(ValueError):
                artifacts.finish([cell(split_fingerprint=None)])
            with self.assertRaises(ValueError):
                artifacts.finish([cell(status="unsupported", error=None)])
            with self.assertRaises(ValueError):
                artifacts.finish([cell(), cell()])

    def test_config_hash_preserves_seed_as_distinct_identity_not_configuration(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = BenchmarkArtifacts(output, {}, protocol(), study())
            rows = artifacts.finish([cell(), cell(seed=43, split_fingerprint="split-2")])
            self.assertEqual(rows[0]["config_sha256"], rows[1]["config_sha256"])
            self.assertNotEqual(rows[0]["record_id"], rows[1]["record_id"])
            summary = summarize_records(rows)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["seed_count"], 2)

    def test_fixed_policy_aggregates_selected_configs_and_separates_round_caps(self):
        with tempfile.TemporaryDirectory() as output:
            args = {"effective_settings": {"ugo_max_iterations": 1}}
            first = BenchmarkArtifacts(output, args, protocol(), study())
            records = first.finish([cell(), cell(seed=43, split_fingerprint="second", config={"C": 10, "threshold": .3})])
            self.assertNotEqual(records[0]["config_sha256"], records[1]["config_sha256"])
            self.assertEqual(records[0]["evaluation_policy_sha256"], records[1]["evaluation_policy_sha256"])
            self.assertEqual(summarize_records(records)[0]["seed_count"], 2)
            second = BenchmarkArtifacts(output, {"effective_settings": {"ugo_max_iterations": 20}}, protocol(), study())
            alternative = second.finish([cell(method="smote", variant_id="smote")])[0]
            self.assertEqual(paired_summary([records[0], alternative], "baseline"), [])
            self.assertEqual(len(summarize_records([records[0], alternative])), 2)

    def test_prepared_cells_do_not_enter_study_summaries(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = BenchmarkArtifacts(output, {"experiment_type": "setup_prepare"}, protocol(), study())
            artifacts.finish([cell(status="prepared")], status="prepared")
            row = read_records(Path(output) / "benchmark_runs.csv")[0]
            self.assertEqual(summarize_records([row]), [])


class StudyAndSummaryTests(unittest.TestCase):
    def test_unavailable_or_incomplete_variants_cannot_execute(self):
        declaration = {"study_id": "guard-fixture", "variants": {}}
        for metadata in ({"available": False},
                         {"available": True, "missing_components": ["required-component"]}):
            with self.subTest(metadata=metadata):
                declaration["variants"]["candidate"] = metadata
                with self.assertRaises(ValueError):
                    require_variant(declaration, "candidate")
        with self.assertRaises(ValueError):
            require_variant(declaration, "undeclared")

    def test_unfrozen_or_evidence_free_studies_cannot_certify_confirmatory_runs(self):
        components = {"study_id": "gate-fixture", "study_type": "component_ablation",
                      "freeze": {"status": "not_frozen"}}
        with self.assertRaises(ValueError):
            enforce_stage_gate(components, "confirmatory")
        components["freeze"]["status"] = "frozen"
        with self.assertRaises(ValueError):
            enforce_stage_gate(components, "development")

    def test_folds_are_aggregated_within_seed_and_protocols_are_separate(self):
        common = {**cell(), "study_id": "s", "study_sha256": "s-sha", "protocol_id": "p", "protocol_sha256": "p-sha", "config_sha256": "c-sha", "experiment_type": "development"}
        rows = [common, {**common, "split_fingerprint": "second", "average_precision": .9},
                {**common, "seed": 43, "average_precision": .6},
                {**common, "protocol_id": "another"}]
        summaries = summarize_records(rows)
        self.assertEqual(len(summaries), 2)
        self.assertAlmostEqual(summaries[0]["mean"], .7)
        self.assertEqual(summaries[0]["seed_count"], 2)
        with self.assertRaises(ValueError):
            summarize_records([common, common])

    def test_paired_actual_count_mismatch_is_not_a_matched_mechanism_claim(self):
        common = {**cell(), "study_id": "s", "study_sha256": "s-sha", "protocol_id": "p", "protocol_sha256": "p-sha", "config_sha256": "c-sha", "experiment_type": "development"}
        alternative = {**common, "variant_id": "smote", "method": "smote", "average_precision": .8}
        pair = paired_summary([common, alternative], "baseline")[0]
        self.assertAlmostEqual(pair["delta"], .1)
        self.assertFalse(pair["matched_count_comparison"])  # baseline is not a generation mechanism
        alternative["augmentation"] = {"actual_counts": {0: 20, 1: 6}, "underspend": 2}
        self.assertFalse(paired_summary([common, alternative], "baseline")[0]["matched_count_comparison"])
        self.assertEqual(paired_summary([{**common, "seed": 43}, alternative], "baseline"), [])
        ros = {**common, "method": "ros", "variant_id": "ros"}
        self.assertTrue(paired_summary([ros, {**alternative, "augmentation": ros["augmentation"]}], "ros")[0]["matched_count_comparison"])
        changed_candidates = {**alternative, "candidate_configs": [{"C": 2}]}
        self.assertFalse(paired_summary([ros, changed_candidates], "ros")[0]["same_candidate_configs"])

    def test_sos_layout_and_unresolved_aggregation_do_not_claim_verification(self):
        p = protocol()
        p.update(status="unresolved", aggregation={"classifiers": ["decision_tree", "logistic_regression", "adaboost", "mlp"],
                                                   "seeds": [0, 1, 2, 3, 4], "metric": "sos_inverse_support_f1",
                                                   "order": "classifier_mean_per_seed_then_seed_mean", "spread": "sample_std"})
        records = [{**cell(), "algorithm": algorithm, "seed": seed, "split_fingerprint": str(seed),
                    "protocol_id": p["protocol_id"], "protocol_sha256": snapshot_sha256(p),
                    "experiment_type": "reproduction", "sos_inverse_support_f1": .6 + seed / 100}
                   for algorithm in p["aggregation"]["classifiers"] for seed in p["aggregation"]["seeds"]]
        with self.assertRaisesRegex(ValueError, "resolved"):
            sos_weighted_f1_summary(records, p)
        with self.assertRaises(ValueError):
            sos_weighted_f1_summary(records[:-1], p)
        p["status"] = "resolved"
        for record in records:
            record["protocol_sha256"] = snapshot_sha256(p)
        self.assertEqual(sos_weighted_f1_summary(records, p)["published_comparability"], "verified")

    def test_two_classifier_repeated_seed_author_layout(self):
        p = protocol()
        p["aggregation"] = {"classifiers": ["decision_tree", "mlp"],
                            "repetitions": [{"id": 0, "seed": 2021}, {"id": 1, "seed": 2021}],
                            "metric": "sos_inverse_support_f1", "order": "all_cells", "spread": "population_std"}
        rows = [{**cell(), "algorithm": algorithm, "seed": 2021, "repetition_id": repetition,
                 "protocol_id": p["protocol_id"], "protocol_sha256": snapshot_sha256(p),
                 "experiment_type": "reproduction", "sos_inverse_support_f1": .6}
                for algorithm in p["aggregation"]["classifiers"] for repetition in (0, 1)]
        summary = sos_weighted_f1_summary(rows, p)
        self.assertEqual(summary["classifier_count"], 2)
        self.assertEqual(summary["trial_count"], 2)
        self.assertEqual(summary["seed_count"], 1)


if __name__ == "__main__":
    unittest.main()
