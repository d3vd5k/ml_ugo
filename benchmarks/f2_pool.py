"""Shoppers F2 replay of immutable F1 source proposals, without flow/U2 refits.

Run ``python -m benchmarks.f2_pool --smoke`` for training-only saved-pool replay,
or ``--run`` for the complete saved LR/RF/LGB x five-seed grid. Support off is
F1 exactly; thinning samples F1-accepted original proposals per original group.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from benchmarks.f1_pool import (CLASSIFIERS, accounting, flow_decisions,
                                original_pool, save_decisions, validation_fit)
from benchmarks.h1_pool import GROUP_CONVENTION, fit_replay, group_recall, predictions
from benchmarks.protocols import load_dataset, make_splits
from benchmarks.results import BenchmarkArtifacts, snapshot_sha256
from benchmarks.run import prepare_partitions, split_fingerprint
from experiments import data_fingerprint, seed_everything, write_json

Q_GRID = (None, .01, .05, .10)
THINNING_SEEDS = tuple(range(5))
FULL_SOURCE = "results/runs/20261007T172857-728d008031d6"
SMOKE_SOURCE = "results/smoke/20261007T180216-278047c1ae7f"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_f1_thinning(group_ids, f1_mask, f2_mask, seed, proposal_ids=None):
    """Uniform per-group samples from F1 only; stable under proposal reordering.

    Proposal IDs bind the random draw to original source ordering, independently
    of batch size, chosen classifier cells, or which groups contain zero accepts.
    """
    groups = np.asarray(group_ids)
    eligible, target = np.asarray(f1_mask, dtype=bool), np.asarray(f2_mask, dtype=bool)
    if groups.ndim != 1 or groups.shape != eligible.shape or groups.shape != target.shape:
        raise ValueError("Expected aligned one-dimensional groups and masks")
    if np.any(target & ~eligible):
        raise ValueError("F2 must be a subset of F1")
    ids = np.arange(len(groups)) if proposal_ids is None else np.asarray(proposal_ids)
    if ids.shape != groups.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Proposal IDs must be unique and aligned")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Thinning seed must be a nonnegative integer")
    result = np.zeros(len(groups), dtype=bool)
    for group in np.unique(groups):
        positions = np.flatnonzero((groups == group) & eligible)
        count = int(np.sum((groups == group) & target))
        if not count:
            continue
        if count == len(positions):
            result[positions] = True
            continue
        positions = positions[np.argsort(ids[positions], kind="stable")]
        key = hashlib.sha256(json.dumps([int(seed), int(group)]).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:16], "little"))
        result[rng.choice(positions, count, replace=False)] = True
    return result


def choose_q(diagnostics):
    """AP first; exact ties choose off, then the smaller active quantile."""
    return max(diagnostics, key=lambda row: (
        row["metrics"]["average_precision"], row["q"] is None,
        -row["q"] if row["q"] is not None else 0))["q"]


def source_manifest(path, smoke):
    manifest = read_json(path / "manifest.json")
    arguments = manifest["arguments"]
    if manifest["status"] != "completed" or manifest["protocol_snapshot"]["dataset"] != "shoppers":
        raise ValueError("Source must be a completed Shoppers run")
    if manifest["study_snapshot"].get("study_id") != "shoppers_f1_saved_u2_pool_v1":
        raise ValueError("Source must be the immutable F1 pool study")
    if bool(arguments.get("smoke")) != smoke or (manifest["experiment_type"] == "setup_smoke") != smoke:
        raise ValueError("Smoke must use a training-only smoke source; full replay must use a full source")
    for key in ("protocol", "study"):
        if snapshot_sha256(manifest[key + "_snapshot"]) != manifest[key + "_sha256"]:
            raise ValueError("Source snapshot hash mismatch")
    settings = arguments["effective_settings"]
    if (settings["augmentation"]["mode"] != "matched" or
            not settings["method_parameters"]["ugo_groups_only"].get("capture_proposals")):
        raise ValueError("Source must preserve matched-budget original U2 proposals")
    if sha256(Path(__file__).resolve().parents[1] / "samplers/flow_gmm.py") != manifest["source_sha256"]["samplers/flow_gmm.py"]:
        raise ValueError("Pinned source FlowGMM implementation differs")
    return manifest


def indices_for(dataset, row_ids):
    identities = np.asarray(dataset["row_ids"]).astype(str)
    if len(np.unique(identities)) != len(identities):
        raise ValueError("Dataset row IDs must be unique")
    lookup = {value: index for index, value in enumerate(identities)}
    requested = np.asarray(row_ids).astype(str)
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise ValueError("Source row IDs must be unique vectors")
    try:
        return np.asarray([lookup[value] for value in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Unknown source row ID") from error


def load_source_cell(source, dataset, algorithm, seed, fold, manifest):
    root = source / "shoppers" / f"fold-{fold}" / f"seed-{seed}" / algorithm
    selection = read_json(root / "F1_pool_selection/selection.json")
    replay = read_json(root / "F1_pool/replay.json")
    for key in ("selected_config", "selected_u2_config"):
        if snapshot_sha256(selection[key]) != selection[key + "_sha256"]:
            raise ValueError("Source selected settings hash mismatch")
    if selection["selected_tau"] != selection["selected_config"]["f1_tau"]:
        raise ValueError("Source selected tau mismatch")
    if (selection["dataset_fingerprint"] != manifest["datasets"]["shoppers"]["fingerprint"] or
            selection["algorithm"] != algorithm or selection["seed"] != seed or selection["fold"] != fold or
            replay["selected_config"] != selection["selected_config"]):
        raise ValueError("Source cell identity/settings mismatch")
    if any(selection["selected_config"].get(key) != value for key, value in selection["selected_u2_config"].items()):
        raise ValueError("F1 settings did not preserve locked U2 settings")
    u2 = selection["selected_u2_config"]["sampler_parameters"]
    if u2.get("grouping") != "kmeans" or u2.get("allocation") != "stability" or u2.get("group_alpha") not in (0., .5, 1.):
        raise ValueError("Source must be locked canonical U2 KMeans/stability")
    flow = selection["selected_config"]["flow_parameters"]
    if any(flow.get(key) != value for key, value in manifest["arguments"]["effective_settings"].get("flow_parameters", {}).items()):
        raise ValueError("Source flow settings mismatch")
    pools, audits, real_ids, real_arrays, assignments = {}, {}, {}, {}, {}
    for role in ("inner", "outer"):
        folder = root / "saved_pool" / role
        reference = selection[role + "_pool"]
        for filename, hash_key in (("pool.npz", "pool_sha256"), ("proposals.npz", "proposals_sha256"),
                                   ("candidate_log.json", "candidate_log_sha256")):
            if sha256(folder / filename) != reference[hash_key]:
                raise ValueError("Source pool hash mismatch")
        report = read_json(folder / "pool.json")
        if report["selected_u2_config"] != selection["selected_u2_config"]:
            raise ValueError("Source pool locked U2 settings mismatch")
        audits[role] = report["u2_audit"]
        settings = manifest["arguments"]["effective_settings"]
        if (audits[role]["augmentation"]["requested_count"] != settings["augmentation"]["total"] or
                audits[role]["sampler_parameters"] != selection["selected_u2_config"]["sampler_parameters"]):
            raise ValueError("Source pool effective budget/sampler settings mismatch")
        with np.load(folder / "pool.npz") as saved:
            real_ids[role] = saved["real_row_ids"].copy()
            real_arrays[role] = (saved["X_real"].copy(), saved["y_real"].copy())
        pools[role] = original_pool(audits[role], real_arrays[role][0].shape[1])
        with np.load(folder / "proposals.npz") as saved:
            for key, field in (("X", "original_coordinates"), ("y", "classes"),
                               ("group_ids", "original_source_group_ids"), ("proposal_ids", "proposal_ids")):
                if not np.array_equal(pools[role][key], saved[field], equal_nan=True):
                    raise ValueError("Original source proposal mismatch")
        assignments[role] = pd.read_csv(folder / "minority_evaluation_groups.csv")
    validation = pd.read_csv(root / "F1_pool/validation_predictions.csv", dtype={"row_id": str}, float_precision="round_trip")
    test = pd.read_csv(root / "F1_pool/test_predictions.csv", dtype={"row_id": str}, float_precision="round_trip")
    split = {"fold": fold, "fit_indices": indices_for(dataset, real_ids["inner"]),
             "train_indices": indices_for(dataset, real_ids["outer"]),
             "validation_indices": indices_for(dataset, validation["row_id"]),
             "test_indices": indices_for(dataset, test["row_id"])}
    if split_fingerprint(dataset, split) != selection["split_fingerprint"]:
        raise ValueError("Source split fingerprint mismatch")
    if manifest["arguments"]["smoke"]:
        original = next(item for item in make_splits(dataset, manifest["protocol_snapshot"], seed) if item["fold"] == fold)
        used = set(split["train_indices"]) | set(split["test_indices"])
        if not used <= set(original["train_indices"]) or used & set(original["test_indices"]):
            raise ValueError("Smoke source accesses official outer test")
        if not read_json(root.parent / "split.json")["original_outer_test_untouched"]:
            raise ValueError("Smoke source lacks training-only provenance")
    partitions = prepare_partitions(dataset, split)
    for role, key, indices in (("inner", "X_fit", split["fit_indices"]), ("outer", "X_train", split["train_indices"])):
        if not np.array_equal(partitions[key], real_arrays[role][0]) or not np.array_equal(dataset["y"][indices], real_arrays[role][1]):
            raise ValueError("Source preprocessing or labels differ from raw source split")
    if sha256(root / "saved_pool/flow.pt") != selection["flow_model"]["sha256"]:
        raise ValueError("Source flow checkpoint hash mismatch")
    return root, selection, replay, pools, audits, assignments, split, partitions, {"inner": validation, "outer": test}


def save_f2_decisions(folder, pool, scores, link, tau, q):
    save_decisions(folder, pool, scores["accepted"], link, scores, tau)
    report = read_json(folder / "candidate_log.json")
    for index, row in enumerate(report["rows"]):
        row.update(support_filter=q is not None, support_quantile=q)
        for key in ("compatibility_accepted", "likelihood_accepted", "likelihood_threshold"):
            value = np.asarray(scores[key])[index].item()
            if key == "likelihood_threshold" and not np.isfinite(value):
                value = "-inf" if np.isneginf(value) else None
            row[key] = value
    write_json(folder / "candidate_log.json", report)
    np.savez_compressed(folder / "decisions.npz", proposal_ids=pool["proposal_ids"], **{
        key: np.asarray(value, dtype=str) if key == "reasons" else value
        for key, value in scores.items() if isinstance(value, np.ndarray)})


def assert_f1_reproduction(root, role, pool, scores, model, X, saved_predictions):
    path = root / "F1_pool" / ("inner/candidate_log.json" if role == "inner" else "candidate_log.json")
    rows = read_json(path)["rows"]
    if ([row["proposal_id"] for row in rows] != pool["proposal_ids"].tolist() or
            not np.array_equal(scores["accepted"], [row["accepted"] for row in rows])):
        raise ValueError("Support off does not reproduce saved F1 acceptance")
    probability = model.predict_proba(X)
    expected = saved_predictions[[f"probability_{label}" for label in model.classes_]].to_numpy()
    if not np.array_equal(probability, expected):
        raise ValueError("Support off does not exactly reproduce saved F1 probabilities")


def evaluate_cell(source, dataset, algorithm, seed, fold, manifest, artifacts, rows):
    from samplers.flow_gmm import FlowGMMCore, FlowGate
    from samplers.flow_support import CalibratedFlowGate, calibrate_flow

    started = perf_counter()
    root, selection, previous, pools, audits, assignments, split, partitions, saved_predictions = load_source_cell(
        source, dataset, algorithm, seed, fold, manifest)
    config, tau = selection["selected_config"], selection["selected_tau"]
    y, ids, positive = dataset["y"], dataset["row_ids"], previous["positive_class"]
    fit, validation, train, test = (split[key] for key in ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    settings = manifest["arguments"]["effective_settings"]
    destination = artifacts.fold_path("shoppers", fold) / f"seed-{seed}" / algorithm
    destination.mkdir(parents=True, exist_ok=False)
    core = FlowGMMCore.load(root / "saved_pool/flow.pt", device=previous["detector"]["resolved_device"])
    if core.seed != seed:
        raise ValueError("Source flow seed differs from classifier/sampler seed")
    for role in ("inner_detector", "detector"):
        if set(previous[role]["fit_row_ids"]) != set(core.fit_row_ids.tolist()):
            raise ValueError("Source flow fit roles differ from checkpoint")
        if previous[role]["model_checksum"] != core.audit["model_checksum"] or previous[role]["model_config"] != core.audit["model_config"]:
            raise ValueError("Source flow model/config provenance mismatch")
    calibration_ids = previous["inner_detector"]["calibration_row_ids"]
    if set(core.fit_row_ids.tolist()) | set(calibration_ids) != set(ids[fit].tolist()):
        raise ValueError("Frozen flow fit/calibration must partition saved real inner fit")
    cal_indices = indices_for(dataset, calibration_ids)
    calibration = calibrate_flow(core, dataset["X"].iloc[cal_indices], y[cal_indices], ids[cal_indices])
    for q in Q_GRID:
        calibration.thresholds(q)
    base, gates, scores, scoring_times = {}, {}, {}, {}
    for role, detector_key in (("inner", "inner_detector"), ("outer", "detector")):
        detector = previous[detector_key]
        base[role] = FlowGate(core, partitions[role], calibration_row_ids=detector["calibration_row_ids"],
                              validation_row_ids=detector["validation_row_ids"])
        base[role].role = role
        base[role].calibration_class_counts = calibration.class_counts
        if set(detector["validation_row_ids"]) != set(ids[validation].tolist()):
            raise ValueError("Source flow validation roles differ from classifier validation")
    preprocessing = {}
    for role in ("inner", "outer"):
        path = destination / f"{role}_preprocessor.joblib"
        joblib.dump(partitions[role], path)
        preprocessing[role] = {"path": str(path.resolve()), "sha256": sha256(path)}
    write_json(destination / "calibration.json", {**calibration.audit,
        "validation_row_ids": ids[validation], "roles_disjoint": True,
        "source_checkpoint": selection["flow_model"], "classifier_preprocessing": preprocessing})
    source_link = {"source_run": str(source.resolve()), "source_manifest_sha256": sha256(source / "manifest.json"),
                   "source_selection_sha256": sha256(root / "F1_pool_selection/selection.json"),
                   "source_replay_sha256": sha256(root / "F1_pool/replay.json"),
                   "inner_pool": selection["inner_pool"], "outer_pool": selection["outer_pool"],
                   "flow_model": selection["flow_model"], "source_effective_settings_sha256": snapshot_sha256(settings)}
    source_link["classifier_preprocessing"] = preprocessing
    source_link["source_classifiers"] = {role: {"path": str((root / "F1_pool" / f"{role}_classifier.joblib").resolve()),
        "sha256": sha256(root / "F1_pool" / f"{role}_classifier.joblib")} for role in ("inner", "outer")}
    metadata = {key: selection[key] for key in ("dataset", "dataset_fingerprint", "split_fingerprint", "seed", "algorithm", "fold", "budget_mode", "tuning_budget", "candidate_configs")}
    common = {**metadata, **source_link, "experiment_type": manifest["experiment_type"],
              "selected_u2_config": selection["selected_u2_config"], "f1_tau": tau,
              "manual_retention": "user_only_pending", "group_assignment_convention": GROUP_CONVENTION}
    tuning = destination / "F2_pool_selection"
    tuning.mkdir()
    diagnostics, inner_models, thresholds, fit_seconds = [], {}, {}, {}
    tick = perf_counter()
    for q in Q_GRID:
        gate = CalibratedFlowGate(base["inner"], calibration, q=q)
        gates["inner", q] = gate
        score_started = perf_counter()
        scores["inner", q] = gate.predict(pools["inner"]["X"], pools["inner"]["y"], tau)
        scoring_times["inner", q] = perf_counter() - score_started
        off = flow_decisions(scores["inner", q], tau)["accepted"]
        if np.any(scores["inner", q]["accepted"] & ~off):
            raise ValueError("F2 acceptance escaped F1")
        fit_started = perf_counter()
        if q is None:
            model = joblib.load(root / "F1_pool/inner_classifier.joblib")
            threshold = selection["threshold"]
            assert_f1_reproduction(root, "inner", pools["inner"], scores["inner", q], model,
                                  partitions["X_validation"], saved_predictions["inner"])
            metrics = previous["validation_metrics"]
        elif np.array_equal(scores["inner", q]["accepted"], scores["inner", None]["accepted"]):
            model, threshold, metrics = inner_models[None], thresholds[None], previous["validation_metrics"]
        else:
            model, threshold, metrics = validation_fit(algorithm, config["parameters"], settings, seed, partitions,
                y, fit, validation, pools["inner"], scores["inner", q]["accepted"], config["epoch"], positive, manifest["protocol_snapshot"])
        reused_f1 = q is None or model is inner_models[None]
        inner_models[q], thresholds[q] = model, threshold
        fit_seconds[q] = 0. if reused_f1 else perf_counter() - fit_started
        q_folder = tuning / ("off" if q is None else f"q-{q:.2f}")
        q_folder.mkdir()
        q_link = {**common, "variant_id": "F2_pool_selection", "support_quantile": q,
                  "selected_config_sha256": snapshot_sha256({**config, "f2_q": q})}
        save_f2_decisions(q_folder, pools["inner"], scores["inner", q], q_link, tau, q)
        predictions(model, partitions["X_validation"], y[validation], ids[validation], positive,
                    threshold, manifest["protocol_snapshot"], q_folder, "validation", q_link)
        diagnostic = {"q": q, "metrics": metrics, "threshold": threshold,
                      "accepted_count": int(scores["inner", q]["accepted"].sum()), "detector": gate.audit,
                      "classifier_fit_seconds": fit_seconds[q], "scoring_seconds": scoring_times["inner", q]}
        diagnostic["new_classifier_fit_count"] = int(not reused_f1)
        write_json(q_folder / "diagnostics.json", {**q_link, **diagnostic})
        diagnostics.append(diagnostic)
    selected_q = choose_q(diagnostics)
    selection_seconds = perf_counter() - tick
    selected_config = {**config, "f2_q": selected_q, "q_grid": list(Q_GRID),
                       "f2_selection_policy": "real_inner_validation_AP_ties_off_then_lower_q"}
    common.update(selected_config=selected_config, selected_config_sha256=snapshot_sha256(selected_config))
    write_json(tuning / "selection.json", {**common, "selected_q": selected_q, "diagnostics": diagnostics,
        "selection_seconds": selection_seconds, "outer_test_used_for_selection": False})
    outer_models, outer_fit_times = {}, {}
    for q in Q_GRID:
        gate = CalibratedFlowGate(base["outer"], calibration, q=q)
        gates["outer", q] = gate
        tick = perf_counter()
        scores["outer", q] = gate.predict(pools["outer"]["X"], pools["outer"]["y"], tau)
        scoring_times["outer", q] = perf_counter() - tick
        if np.any(scores["outer", q]["accepted"] & ~scores["outer", None]["accepted"]):
            raise ValueError("Outer F2 acceptance escaped F1")
        tick = perf_counter()
        if q is None:
            outer_models[q] = joblib.load(root / "F1_pool/outer_classifier.joblib")
            assert_f1_reproduction(root, "outer", pools["outer"], scores["outer", q], outer_models[q],
                                  partitions["X_test"], saved_predictions["outer"])
        elif np.array_equal(scores["outer", q]["accepted"], scores["outer", None]["accepted"]):
            outer_models[q] = outer_models[None]
        else:
            outer_models[q] = fit_replay(algorithm, config["parameters"], settings, seed, partitions["X_train"], y[train],
                                        pools["outer"], scores["outer", q]["accepted"], config["epoch"])
        outer_fit_times[q] = 0. if q is None or outer_models[q] is outer_models[None] else perf_counter() - tick
    variants = [("F2_pool", selected_q, None), ("F2_support_off", None, None)]
    variants += [(f"F2_q_{q:.2f}", q, None) for q in Q_GRID if q is not None]
    variants += [(f"F1_matched_thinning_{repeat}", selected_q, repeat) for repeat in THINNING_SEEDS]
    for variant, q, repeat in variants:
        tick = perf_counter()
        masks, decisions = {}, {}
        for role in ("inner", "outer"):
            decisions[role] = dict(scores[role, q])
            masks[role] = np.asarray(scores[role, q]["accepted"], dtype=bool)
            if repeat is not None:
                masks[role] = matched_f1_thinning(pools[role]["group_ids"], scores[role, None]["accepted"],
                    masks[role], repeat, pools[role]["proposal_ids"])
                reasons = np.asarray(scores[role, None]["reasons"], dtype=object).copy()
                reasons[scores[role, None]["accepted"] & ~masks[role]] = "matched_thinning_rejected"
                reasons[masks[role]] = "accepted"
                decisions[role] = {**scores[role, None], "accepted": masks[role], "reasons": reasons}
        if repeat is None:
            inner_model, outer_model, threshold = inner_models[q], outer_models[q], thresholds[q]
            classifier_seconds = fit_seconds[q] + outer_fit_times[q]
        else:
            fit_started = perf_counter()
            if np.array_equal(masks["inner"], scores["inner", None]["accepted"]):
                inner_model, threshold = inner_models[None], thresholds[None]
            else:
                inner_model, threshold, _ = validation_fit(algorithm, config["parameters"], settings, seed, partitions,
                    y, fit, validation, pools["inner"], masks["inner"], config["epoch"], positive, manifest["protocol_snapshot"])
            outer_model = (outer_models[None] if np.array_equal(masks["outer"], scores["outer", None]["accepted"]) else
                fit_replay(algorithm, config["parameters"], settings, seed, partitions["X_train"], y[train],
                           pools["outer"], masks["outer"], config["epoch"]))
            classifier_seconds = perf_counter() - fit_started
        folder = destination / variant
        folder.mkdir()
        link = {**common, "variant_id": variant, "support_quantile": q if repeat is None else None,
                "support_filter": q is not None and repeat is None, "thinning_seed": repeat,
                "classifier_seed": seed, "sampler_seed": seed, "matched_target_quantile": q}
        score_started = perf_counter()
        inner_metrics, inner_prediction, _ = predictions(inner_model, partitions["X_validation"], y[validation], ids[validation],
            positive, threshold, manifest["protocol_snapshot"], folder, "validation", link)
        metrics, prediction, prediction_path = predictions(outer_model, partitions["X_test"], y[test], ids[test],
            positive, threshold, manifest["protocol_snapshot"], folder, "test", link)
        predictive_scoring_seconds = perf_counter() - score_started
        joblib.dump(inner_model, folder / "inner_classifier.joblib")
        joblib.dump(outer_model, folder / "outer_classifier.joblib")
        reports = {}
        for role, X, indices in (("inner", partitions["X_fit"], fit), ("outer", partitions["X_train"], train)):
            decision_folder = folder / "inner" if role == "inner" else folder
            decision_folder.mkdir(exist_ok=True)
            save_f2_decisions(decision_folder, pools[role], decisions[role], {**link, "partition": role}, tau, q if repeat is None else None)
            reports[role] = accounting(pools[role], masks[role], audits[role], X, y[indices], decisions[role])
            if not masks[role].any():
                reports[role]["replay_status"] = "f2_no_candidates_accepted"
        groups = group_recall(assignments["outer"], prediction, audits["outer"]["support_summary"]["group_support"])
        inner_groups = group_recall(assignments["inner"], inner_prediction, audits["inner"]["support_summary"]["group_support"])
        row = {**link, **metrics, **reports["outer"], "config": {**selected_config, "variant_q": q,
               "thinning_seed": repeat, "threshold": threshold}, "method": "saved_u2_pool_replay", "strategy": variant,
               "status": "completed", "positive_class": positive, "threshold": threshold,
               "predictions_path": prediction_path, "validation_metrics": inner_metrics,
               "group_recall": groups, "validation_group_recall": inner_groups, "inner_accounting": reports["inner"],
               "detector": deepcopy(gates["outer", q if repeat is None else None].audit),
               "inner_detector": deepcopy(gates["inner", q if repeat is None else None].audit),
               "runtime_seconds": perf_counter() - tick, "classifier_fit_seconds": classifier_seconds if repeat is not None else 0.,
               "shared_q_classifier_fit_seconds": classifier_seconds if repeat is None else 0.,
               "predictive_scoring_seconds": predictive_scoring_seconds, "shared_new_selection_seconds": selection_seconds,
               "shared_new_calibration_seconds": calibration.audit["calibration_seconds"],
               "detector_training_seconds": 0., "detector_inference_seconds": scoring_times["inner", q] + scoring_times["outer", q],
               "shared_previous_compute": {"source_cell_wall_seconds": read_json(root / "saved_pool/study_cell.json")["total_wall_seconds"],
                    "u2_seconds": previous["shared_canonical_u2_runtime_seconds"], "flow_training_seconds": previous["detector_training_seconds"],
                    "f1_replay_seconds": previous["runtime_seconds"]}, "off_exactly_reproduces_f1": True}
        write_json(folder / "groups.json", {**link, "test": groups, "validation": inner_groups,
            "outer_acceptance": reports["outer"]["group_acceptance"], "inner_acceptance": reports["inner"]["group_acceptance"]})
        write_json(folder / "replay.json", row)
        rows.append(row)
    write_json(destination / "study_cell.json", {**common, "total_wall_seconds": perf_counter() - started,
        "new_flow_fit_count": 0, "new_u2_fit_count": 0, "off_exactly_reproduces_f1": True,
        "shared_new_selection_seconds": selection_seconds,
        "new_outer_classifier_fit_seconds": sum(outer_fit_times.values()),
        "new_detector_scoring_seconds": sum(scoring_times.values()),
        "timing_scope": "selection includes inner replay fits/validation and diagnostics; q replay fit time shared by selected/fixed variants; source compute not rerun",
        "iterative_results_equivalent": False, "classifier_real_access_unchanged": True})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    result.add_argument("--validate", action="store_true")
    result.add_argument("--source-run")
    result.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS)
    result.add_argument("--seeds", nargs="+", type=int)
    result.add_argument("--output-dir")
    result.add_argument("--data-home")
    result.add_argument("--local-data")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not (args.run or args.smoke or args.validate):
        raise ValueError("Specify --run, --smoke, or --validate")
    source = Path(args.source_run or (SMOKE_SOURCE if args.smoke else FULL_SOURCE))
    manifest = source_manifest(source, args.smoke)
    algorithms = args.classifiers or manifest["arguments"]["effective_classifiers"]
    seeds = args.seeds or manifest["arguments"]["effective_seeds"]
    if len(set(algorithms)) != len(algorithms) or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Classifiers and seeds must be unique; seeds nonnegative")
    if set(algorithms) - set(manifest["arguments"]["effective_classifiers"]) or set(seeds) - set(manifest["arguments"]["effective_seeds"]):
        raise ValueError("Requested classifiers/seeds are absent from source")
    if not args.smoke and (set(algorithms) != set(CLASSIFIERS) or set(seeds) != set(range(5))):
        raise ValueError("Full source replay requires LR/RF/LGB x seeds 0..4")
    protocol = manifest["protocol_snapshot"]
    dataset = load_dataset(protocol, args.data_home or manifest["arguments"]["data_home"],
                           local_data=args.local_data or manifest["arguments"].get("local_data"), download=False)
    if data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"]) != manifest["datasets"]["shoppers"]["fingerprint"]:
        raise ValueError("Source raw dataset fingerprint mismatch")
    cells = [(algorithm, seed, split["fold"]) for seed in seeds for split in make_splits(dataset, protocol, seed) for algorithm in algorithms]
    if args.validate:
        for algorithm, seed, fold in cells:
            load_source_cell(source, dataset, algorithm, seed, fold, manifest)
        print(json.dumps({"status": "valid", "source_run": str(source), "cells": len(cells), "q_grid": Q_GRID,
                          "thinning_seeds": THINNING_SEEDS, "raw_fingerprint_verified": True}))
        return None
    study = deepcopy(manifest["study_snapshot"])
    study.update(study_id="shoppers_f2_saved_f1_pool_v1", f2_pool_policy={"q_grid": list(Q_GRID),
        "selection": "locked_U2_and_F1_tau_then_inner_validation_AP_ties_off_lower_q",
        "thinning": "five_repeats_F1_only_original_group_matched", "retention": "F1_kept_F2_discarded_by_user"})
    study["manual_review"]["decisions"].update(F1="keep", F2="discard")
    study["variants"]["F1"]["retention_status"] = "kept_by_user"
    study["freeze"]["reason"] = "The user kept F1 and discarded F2; Phase 1 is not frozen."
    arguments = {**vars(args), "source_run": str(source.resolve()), "experiment_type": manifest["experiment_type"],
        "effective_settings": manifest["arguments"]["effective_settings"], "effective_classifiers": algorithms,
        "effective_seeds": seeds, "experiment_scope": "saved_f1_pool_f2_replay"}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, study)
    artifacts.dataset(dataset)
    rows = []
    try:
        for algorithm, seed, fold in cells:
            seed_everything(seed)
            print(f"{manifest['experiment_type']}: shoppers seed={seed} {algorithm}/F2_saved_pool", flush=True)
            evaluate_cell(source, dataset, algorithm, seed, fold, manifest, artifacts, rows)
    except Exception as error:
        artifacts.finish(rows, error=f"{type(error).__name__}: {error}")
        raise
    artifacts.finish(rows)
    print(f"Saved F2 pool replay to {artifacts.path}", flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
