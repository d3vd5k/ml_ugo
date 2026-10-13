import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import evaluate_cell
from experiments import RunArtifacts
from samplers.controls import resample


ROOT = Path(__file__).resolve().parents[1]


def learned_state(value):
    if isinstance(value, dict):
        return {key: learned_state(item) for key, item in value.items()
                if key not in {"fit_score_seconds", "generation_seconds"}}
    if isinstance(value, list):
        return [learned_state(item) for item in value]
    return value


class U2IntegrationTests(unittest.TestCase):
    def test_protected_conflicting_islands_keep_real_support_and_expose_cap_underspend(self):
        X = np.concatenate((np.linspace(0, .12, 24), [.03, .04, 10., 10.1]))[:, None].astype(np.float32)
        y = np.array([0]*24 + [1]*4)
        def train(features, labels):
            return LogisticRegression().fit(features.cpu().numpy(), labels.cpu().numpy())
        common = dict(model_fn=train, device="cpu", max_iterations=20, target_counts={0:24, 1:13})
        baseline = resample("ugo_corrected", X, y, 17, parameters={"alpha":1.}, **common)
        self.assertEqual(len(baseline.y), len(y))
        result = resample("ugo_groups_only", X, y, 17,
                          parameters={"alpha":1., "n_clusters":2, "sample_fraction":1., "repeats":3}, **common)
        self.assertEqual(len(result.y)-len(y), 8)
        np.testing.assert_array_equal(result.X[:len(y)].cpu().numpy(), X)
        np.testing.assert_array_equal(result.y[:len(y)].cpu().numpy(), y)
        audit = result.support_summary["group_support"]
        self.assertEqual(set(audit["protected_conflict_indices"]), {24,25,26,27})
        self.assertEqual([group["real_support"] for group in audit["groups"]], [2,2])
        self.assertEqual([group["actual_generated"] for group in audit["groups"]], [4,4])
        self.assertEqual(sum(audit["seed_multiplicity"].values()), 8)
        self.assertLessEqual(max(audit["seed_multiplicity"].values()), 4)
        self.assertNotEqual(result.stop_reason, "target_counts_reached")
        for parent in result.provenance:
            for key in ("seed_index", "partner_index", "relocation_index"):
                index = parent[key]
                if index >= 0:
                    self.assertLess(index, len(y))
                    self.assertEqual(y[index], parent["class"])
            self.assertEqual(parent["group_id"], audit["row_group_ids"][parent["seed_index"]])
        for allocation in audit["allocation_records"]:
            for group in allocation["group_trace"]:
                self.assertLessEqual(group["used_group"] + group["quota"], 2*group["n_real"])

    def test_outer_test_does_not_select_groups_alpha_or_generation_and_refit_is_real_only(self):
        protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real["y"])), train_size=180, stratify=real["y"], random_state=19)
        data = {"name":"shoppers", "X":real["X"].iloc[chosen].reset_index(drop=True),
                "y":real["y"][chosen].copy(), "row_ids":real["row_ids"][chosen].copy()}
        train, test = train_test_split(np.arange(len(chosen)), test_size=.2, stratify=data["y"], random_state=19)
        data["X"]["VisitorType"] = data["X"]["VisitorType"].astype(object)
        fit, validation = train_test_split(train, test_size=.25, stratify=data["y"][train], random_state=19)
        split = {"fold":0,"train_indices":train,"fit_indices":fit,"validation_indices":validation,"test_indices":test}
        changed = {**data,"X":data["X"].copy(),"y":data["y"].copy()}
        changed["X"].loc[test,"Administrative_Duration"] = 1e12
        changed["X"].loc[test,"VisitorType"] = "test-only-category"
        changed["y"][test] = 1-changed["y"][test]
        settings = {"device":"cpu","mlp_epochs":2,"ugo_max_iterations":4,
                    "augmentation":{"mode":"matched","total":6},"method_parameters":{}}
        metadata = {"seed":19,"sampler_parameters":{"grouping":"kmeans","n_clusters":3,"repeats":3},
                    "sampler_search":{"group_alpha":[0.,.5,1.]}}
        snapshots = []
        with tempfile.TemporaryDirectory() as root:
            for index, dataset in enumerate((data, changed)):
                artifacts = RunArtifacts(Path(root)/str(index), {"random_state":19})
                evaluate_cell(dataset,split,"logistic_regression","ugo_groups_only","U2",
                              [{"C":.1},{"C":1.},{"C":10.}],settings,protocol,artifacts,metadata)
                folder = artifacts.fold_path("shoppers",0)/"seed-19"/"logistic_regression"/"U2"
                snapshots.append({"selection":json.loads((folder/"selection.json").read_text()),
                                  "outer":json.loads((folder/"outer.json").read_text()),
                                  "candidates":[json.loads((folder/f"candidate-{i}.json").read_text()) for i in range(3)]})
        self.assertEqual(learned_state(snapshots[0]),learned_state(snapshots[1]))
        audit = snapshots[0]
        self.assertEqual([candidate["sampler_parameters"]["group_alpha"] for candidate in audit["candidates"]], [0.,.5,1.])
        self.assertEqual(audit["outer"]["sampler_parameters"],audit["selection"]["inner"]["sampler_parameters"])
        inner_ids, outer_ids = set(data["row_ids"][fit]), set(data["row_ids"][train])
        for result, ids in [(candidate,inner_ids) for candidate in audit["candidates"]]+[(audit["outer"],outer_ids)]:
            support = result["support_summary"]["group_support"]
            self.assertEqual(set(support["real_row_ids"]),ids)
            self.assertFalse(set(support["real_row_ids"]) & set(data["row_ids"][test]))
            for group in support["groups"]:
                self.assertEqual(group["real_support"],len(group["row_ids"]))
                self.assertTrue(set(group["row_ids"]) <= ids)
                self.assertFalse(set(group["row_ids"]) & set(data["row_ids"][validation]) if ids==inner_ids else set())


if __name__ == "__main__":
    unittest.main()
