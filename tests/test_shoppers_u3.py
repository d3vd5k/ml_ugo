import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import evaluate_cell, prepare_partitions
from experiments import RunArtifacts
from samplers.geometry import prepare_constraint_geometry


ROOT = Path(__file__).resolve().parents[1]
PAIRS = [{"eta": .25, "rho": .5}, {"eta": .5, "rho": 1.}, {"eta": 1., "rho": 1.5}]


def learned_state(value):
    if isinstance(value, dict):
        return {key: learned_state(item) for key, item in value.items()
                if key not in {"fit_score_seconds", "generation_seconds"}}
    if isinstance(value, list):
        return [learned_state(item) for item in value]
    return value


class U3ShoppersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real["y"])), train_size=1800,
                                    stratify=real["y"], random_state=19)
        cls.data = {"name": "shoppers", "X": real["X"].iloc[chosen].reset_index(drop=True),
                    "y": real["y"][chosen].copy(), "row_ids": real["row_ids"][chosen].copy()}
        cls.data["X"]["VisitorType"] = cls.data["X"]["VisitorType"].astype(object)
        train, test = train_test_split(np.arange(len(chosen)), test_size=.2,
                                       stratify=cls.data["y"], random_state=19)
        fit, validation = train_test_split(train, test_size=.25, stratify=cls.data["y"][train], random_state=19)
        cls.split = {"fold": 0, "train_indices": train, "fit_indices": fit,
                     "validation_indices": validation, "test_indices": test}
        cls.settings = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 4,
                        "augmentation": {"mode": "matched", "total": 6}, "method_parameters": {}}
        # Three learner configurations, retaining the protocol's ten shared proposal slots.
        cls.candidates = [{"C": [.1, 1., 10.][i % 3]} for i in range(10)]

    def test_test_perturbation_preserves_preprocessing_geometry_selection_and_generation(self):
        data, split = self.data, self.split
        fit, validation, train, test = (split[key] for key in
                                       ("fit_indices", "validation_indices", "train_indices", "test_indices"))
        changed = {**data, "X": data["X"].copy(), "y": data["y"].copy()}
        changed["X"].loc[test, "Administrative_Duration"] = 1e12
        changed["X"].loc[test, "VisitorType"] = "test-only-category"
        changed["y"][test] = 1 - changed["y"][test]
        snapshots = []
        training_features = []
        geometry_snapshots = []
        with tempfile.TemporaryDirectory() as root:
            for index, dataset in enumerate((data, changed)):
                partitions = prepare_partitions(dataset, split)
                training_features.append((partitions["X_fit"].copy(), partitions["X_train"].copy()))
                snapshot = {}
                with patch("benchmarks.run.prepare_constraint_geometry", wraps=prepare_constraint_geometry) as prepare:
                    artifacts = RunArtifacts(Path(root) / str(index), {"random_state": 19})
                    for repair in ("shorten", "relocate", "reject"):
                        metadata = {"seed": 19, "sampler_parameters": {"repair": repair},
                                    "sampler_search": {"constraint_settings": PAIRS}}
                        result = evaluate_cell(dataset, split, "logistic_regression", "ugo_constraints_only", repair,
                                               self.candidates, self.settings, self.protocol, artifacts, metadata,
                                               partitions=partitions)
                        folder = artifacts.fold_path("shoppers", 0) / "seed-19" / "logistic_regression" / repair
                        snapshot[repair] = {name: json.loads((folder / f"{name}.json").read_text())
                                            for name in ("selection", "outer", "preprocessing", "reference_rows")}
                        snapshot[repair]["candidates"] = [json.loads((folder / f"candidate-{i}.json").read_text())
                                                           for i in range(len(self.candidates))]
                        self.assertEqual(result["candidate_configs"], self.candidates)
                        self.assertEqual(result["tuning_budget"], 10)
                        self.assertEqual(result["config"]["settings"]["method_parameters"]["ugo_constraints_only"],
                                         snapshot[repair]["selection"]["inner"]["sampler_parameters"])
                    # A different classifier family must reuse exactly the same real geometry.
                    evaluate_cell(dataset, split, "random_forest", "ugo_constraints_only", "family",
                                  [{"n_estimators": 5, "max_depth": 3}], self.settings, self.protocol, artifacts,
                                  {"seed": 19, "sampler_parameters": {"repair": "reject"},
                                   "sampler_search": {"constraint_settings": PAIRS}}, partitions=partitions)
                    self.assertEqual(prepare.call_count, 2)
                cache = partitions["constraint_geometry"]
                self.assertEqual(len(cache), 2)
                self.assertNotIn("protected_groups", partitions)
                for key, geometry in cache.items():
                    self.assertEqual(json.loads(key[1]), {"graph_neighbors": 5, "graph_radius_quantile": .9})
                    expected = fit if key[0] == "inner" else train
                    np.testing.assert_array_equal(geometry.real_row_ids, dataset["row_ids"][expected])
                snapshots.append(snapshot)
                geometry_snapshots.append({key[0]: geometry.snapshot() for key, geometry in cache.items()})
        self.assertEqual(learned_state(snapshots[0]), learned_state(snapshots[1]))
        for original, perturbed in zip(training_features[0], training_features[1]):
            np.testing.assert_array_equal(original, perturbed)
        self.assertEqual(geometry_snapshots[0], geometry_snapshots[1])
        inner_ids, outer_ids = set(data["row_ids"][fit]), set(data["row_ids"][train])
        self.assertGreater(snapshots[0]["shorten"]["outer"]["augmentation"]["actual_count"], 0)
        for repair, snapshot in snapshots[0].items():
            candidates = snapshot["candidates"]
            self.assertEqual([{key: candidate["sampler_parameters"][key] for key in ("eta", "rho")}
                              for candidate in candidates], [PAIRS[i % 3] for i in range(10)])
            scores = [candidate["metrics"]["primary_metric_value"] for candidate in candidates]
            self.assertEqual(snapshot["selection"]["selected_candidate"], int(np.argmax(scores)))
            self.assertEqual(snapshot["selection"]["candidate_configs"], self.candidates)
            self.assertEqual(snapshot["outer"]["sampler_parameters"],
                             snapshot["selection"]["inner"]["sampler_parameters"])
            self.assertTrue(any(attempt["partner_index"] >= 0 and attempt["initial_metrics"] is not None
                                for candidate in candidates
                                for attempt in candidate["support_summary"]["constraint_support"]["attempts"]))
            for audit, ids, indices in [(candidate, inner_ids, fit) for candidate in candidates] + [(snapshot["outer"], outer_ids, train)]:
                support = audit["support_summary"]["constraint_support"]
                self.assertEqual(set(support["real_row_ids"]), ids)
                self.assertFalse(set(support["real_row_ids"]) & set(data["row_ids"][test]))
                if ids == inner_ids:
                    self.assertFalse(set(support["real_row_ids"]) & set(data["row_ids"][validation]))
                self.assertEqual(support["synthetic_training_count"], 0)
                self.assertEqual(len(support["row_group_ids"]), len(indices))
                self.assertNotIn("categorical_groups", audit["sampler_parameters"])
                self.assertEqual(support["configuration"]["repair"], repair)
                for key in ("eta", "rho"):
                    self.assertEqual(support["configuration"][key], audit["sampler_parameters"][key])
                for group in support["groups"]:
                    self.assertEqual(group["real_support"], len(group["row_ids"]))
                    self.assertTrue(set(group["row_ids"]) <= ids)
                self.assertEqual(len(audit["provenance"]), audit["augmentation"]["actual_count"])
                real_features = training_features[0][0 if ids == inner_ids else 1]
                for parent in audit["provenance"]:
                    for key in ("seed_index", "partner_index", "relocation_index"):
                        if key == "relocation_index" and parent[key] == -1:
                            continue
                        if parent.get(key) is not None:
                            self.assertLess(parent[key], len(indices))
                            self.assertGreaterEqual(parent[key], 0)
                            self.assertEqual(support["row_group_ids"][parent[key]], parent["group_id"])
                    self.assertTrue(parent["final_metrics"]["passes"])
                    self.assertEqual(parent["group_id"], support["row_group_ids"][parent["seed_index"]])
                    point = np.asarray(parent["point"], dtype=np.float32)
                    self.assertTrue(np.isfinite(point).all())
                    self.assertEqual(len(point), len(support["numeric_columns"]) + len(support["categorical_columns"]))
                    for block in support["categorical_groups"]:
                        self.assertTrue(set(point[block]) <= {0., 1.})
                        self.assertEqual(float(point[block].sum()), 1.)
                    columns = support["categorical_columns"]
                    np.testing.assert_array_equal(point[columns], real_features[parent["seed_index"], columns])

    def test_graph_configuration_changes_invalidate_real_geometry_cache(self):
        partitions = prepare_partitions(self.data, self.split)
        configurations = [{}, {"graph_neighbors": 6}, {"graph_radius_quantile": .8}]
        with tempfile.TemporaryDirectory() as root:
            artifacts = RunArtifacts(Path(root), {"random_state": 19})
            with patch("benchmarks.run.prepare_constraint_geometry", wraps=prepare_constraint_geometry) as prepare:
                for index, parameters in enumerate(configurations):
                    evaluate_cell(self.data, self.split, "logistic_regression", "ugo_constraints_only", str(index),
                                  [{"C": 1.}], self.settings, self.protocol, artifacts,
                                  {"seed": 19, "sampler_parameters": parameters,
                                   "sampler_search": {"constraint_settings": PAIRS}}, partitions=partitions)
                self.assertEqual(prepare.call_count, 6)
        self.assertEqual(len(partitions["constraint_geometry"]), 6)
        for role in ("inner", "outer"):
            actual = [json.loads(key[1]) for key in partitions["constraint_geometry"] if key[0] == role]
            self.assertCountEqual(actual, [{"graph_neighbors": 5, "graph_radius_quantile": .9},
                                          {"graph_neighbors": 6, "graph_radius_quantile": .9},
                                          {"graph_neighbors": 5, "graph_radius_quantile": .8}])

    def test_invalid_constraint_search_is_rejected_before_generation(self):
        invalid = [[], [{"eta": .5}], [{"eta": .5, "rho": 1., "repair": "shorten"}],
                   [{"eta": 0., "rho": 1.}], [{"eta": -.5, "rho": 1.}],
                   [{"eta": float("nan"), "rho": 1.}], [{"eta": .5, "rho": float("inf")}],
                   [{"eta": True, "rho": 1.}], [{"eta": ".5", "rho": 1.}],
                   [{"eta": .5, "rho": False}], {"eta": .5, "rho": 1.}, [None],
                   ({"eta": .5, "rho": 1.},)]
        partitions = prepare_partitions(self.data, self.split)
        with tempfile.TemporaryDirectory() as root:
            artifacts = RunArtifacts(Path(root), {"random_state": 19})
            for pairs in invalid:
                with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, "finite positive"):
                    evaluate_cell(self.data, self.split, "logistic_regression", "ugo_constraints_only", "invalid",
                                  self.candidates, self.settings, self.protocol, artifacts,
                                  {"seed": 19, "sampler_search": {"constraint_settings": pairs}}, partitions=partitions)
            for method, search in (("ugo_constraints_only", {"group_alpha": [.5]}),
                                   ("ugo_groups_only", {"constraint_settings": PAIRS}),
                                   ("ugo_corrected", {"constraint_settings": PAIRS}),
                                   ("ugo_constraints_only", {"constraint_settings": PAIRS, "group_alpha": [.5]})):
                with self.subTest(method=method, search=search), self.assertRaisesRegex(ValueError, "supports only"):
                    evaluate_cell(self.data, self.split, "logistic_regression", method, "invalid",
                                  self.candidates, self.settings, self.protocol, artifacts,
                                  {"seed": 19, "sampler_search": search}, partitions=partitions)
        self.assertNotIn("constraint_geometry", partitions)


if __name__ == "__main__":
    unittest.main()
