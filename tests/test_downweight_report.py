"""Focused report-helper regressions; no scientific result fixtures."""
import json

import numpy as np
import pytest

from benchmarks.downweight_report import (evidence_json, native_metrics, paired_bootstrap,
                                          verified_root)
from benchmarks.tabdiff_u2_report import sha


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_metrics_preserve_probability_precision(dtype):
    y = np.array([0, 1, 0])
    p = np.array([[1., 0.], [0., 1.], [.4, .6]], dtype=dtype)
    observed = native_metrics(y, p)
    epsilon = np.finfo(dtype).eps
    losses = -np.log(np.array([1-epsilon, 1-epsilon, .4], dtype=dtype))
    assert observed["ordinary_cross_entropy"] == pytest.approx(float(losses.mean()), rel=0, abs=1e-8)
    assert observed["balanced_cross_entropy"] == pytest.approx(
        float(np.average(losses, weights=[.75, 1.5, .75])), rel=0, abs=1e-8)



def comparison(algorithm="mlp"):
    return {"algorithm": algorithm, "comparison": "selected vs unit",
            "left": "selected", "right": "unit"}


def test_identical_arms_have_zero_paired_uncertainty():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([.1, .7, .4, .9, .6, .3])
    predictions = {"mlp": {seed: {"unit": p, "selected": p.copy()} for seed in range(5)}}
    result = paired_bootstrap(y, predictions, [comparison()], draws=20)
    observed = result["comparisons"][0]
    assert observed["matched_seed_deltas"] == {str(seed): 0. for seed in range(5)}
    assert observed["mean_delta"] == 0.
    assert observed["paired_stratified_shared_row_percentile_CI95"] == [0., 0.]
    assert observed["shared_row_and_matched_seed_joint_percentile_CI95"] == [0., 0.]


def test_rows_and_seed_draws_are_shared_across_classifiers():
    y = np.array([0, 0, 0, 1, 1, 1])
    unit = np.array([.1, .7, .4, .9, .6, .3])
    chosen = np.array([.1, .2, .5, .8, .9, .4])
    predictions = {a: {s: {"unit": np.roll(unit, s), "selected": np.roll(chosen, s)}
                       for s in range(5)} for a in ("mlp", "random_forest")}
    comparisons = [comparison(a) for a in predictions]
    first = paired_bootstrap(y, predictions, comparisons, draws=20)
    second = paired_bootstrap(y, predictions, comparisons, draws=20)
    assert first == second
    left, right = first["comparisons"]
    for field in ("matched_seed_deltas", "mean_delta", "paired_stratified_shared_row_percentile_CI95",
                  "shared_row_and_matched_seed_joint_percentile_CI95"):
        assert left[field] == right[field]


def test_bootstrap_rejects_unmatched_seeds():
    y = np.array([0, 1])
    p = np.array([.1, .9])
    predictions = {"mlp": {0: {"unit": p, "selected": p}},
                   "random_forest": {1: {"unit": p, "selected": p}}}
    with pytest.raises(ValueError, match="matched seeds"):
        paired_bootstrap(y, predictions, [comparison()], draws=2)


def test_bootstrap_rejects_unaligned_probabilities():
    predictions = {"mlp": {0: {"unit": np.array([.2]), "selected": np.array([.2, .8])}}}
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap(np.array([0, 1]), predictions, [comparison()], draws=2)


def test_bootstrap_rejects_duplicate_comparisons():
    p = np.array([.2, .8])
    predictions = {"mlp": {0: {"unit": p, "selected": p}}}
    with pytest.raises(ValueError, match="Duplicate"):
        paired_bootstrap(np.array([0, 1]), predictions, [comparison(), comparison()], draws=2)


def bound_root(tmp_path, status="completed"):
    root = tmp_path / "root"
    root.mkdir()
    manifest = root / "manifest.json"
    records = root / "benchmark_records.json"
    manifest.write_text(json.dumps({"status": status}))
    records.write_text("[]")
    proof = root / "independent_verification.json"
    proof.write_text(json.dumps({"status": "verified", "root": str(root.resolve()),
                                "actual_file_sha256": {str(path.resolve()): sha(path)
                                                       for path in (manifest, records)}}))
    return root, proof


def test_completed_verified_root_requires_manifest_and_records_binding(tmp_path):
    root, proof = bound_root(tmp_path)
    hashes = {}
    manifest, _ = verified_root(root, proof, hashes)
    assert manifest["status"] == "completed"
    assert evidence_json(root / "benchmark_records.json", hashes) == []
    data = json.loads(proof.read_text())
    del data["actual_file_sha256"][str((root / "benchmark_records.json").resolve())]
    proof.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not bind"):
        verified_root(root, proof, {})


def test_running_root_is_never_reportable(tmp_path):
    root, proof = bound_root(tmp_path, "running")
    with pytest.raises(ValueError, match="completed"):
        verified_root(root, proof, {})


def test_stale_verification_is_rejected(tmp_path):
    root, proof = bound_root(tmp_path)
    (root / "benchmark_records.json").write_text("[{}]")
    with pytest.raises(ValueError, match="binding changed"):
        verified_root(root, proof, {})


def test_new_artifacts_are_not_silently_added_to_verification(tmp_path):
    path = tmp_path / "unverified.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="not bound"):
        evidence_json(path, {})
