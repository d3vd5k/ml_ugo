import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets.scripts.climb import categorical_columns, load_climb_datasets, make_preprocessor
from datasets.scripts.synthetic import DATASETS, corrupt_labels, create_synthetic_dataset, create_synthetic_split


class RawPreprocessingTests(unittest.TestCase):
    def test_imputation_and_scale_are_fitted_on_training_only(self):
        train = pd.DataFrame({"number": [0.0, 2.0, np.nan], "kind": ["a", "b", None]})
        heldout = pd.DataFrame({"number": [1000.0, np.nan], "kind": ["new", "new"]})
        preprocessor = make_preprocessor(train)
        transformed = preprocessor.fit_transform(train)
        numeric = preprocessor.named_steps["columns"].named_transformers_["numeric"]
        np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, [1.0])
        np.testing.assert_array_equal(numeric.named_steps["scale"].mean_, [1.0])
        state = numeric.named_steps["scale"].mean_.copy()
        result = preprocessor.transform(heldout)
        np.testing.assert_array_equal(numeric.named_steps["scale"].mean_, state)
        self.assertGreater(result[0, 0], 100)
        self.assertEqual(result[1, 0], 0)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_unknown_categories_do_not_expand_training_schema(self):
        train = pd.DataFrame({"number": [1, 2], "kind": pd.Categorical(["a", "b"])})
        preprocessor = make_preprocessor(train)
        fit = preprocessor.fit_transform(train)
        result = preprocessor.transform(pd.DataFrame({"number": [3], "kind": ["heldout-only"]}))
        self.assertEqual(result.shape[1], fit.shape[1])
        self.assertEqual(categorical_columns(preprocessor), (1, 2))
        np.testing.assert_array_equal(result[:, categorical_columns(preprocessor)], [[0, 0]])
        self.assertEqual(len(preprocessor.get_feature_names_out()), 3)

    def test_nullable_and_empty_training_features_are_retained(self):
        train = pd.DataFrame({"number": pd.Series([1, pd.NA, 3], dtype="Int64"),
                              "empty": [np.nan] * 3,
                              "kind": pd.Series(["a", pd.NA, "b"], dtype="string")})
        preprocessor = make_preprocessor(train)
        result = preprocessor.fit_transform(train)
        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(categorical_columns(preprocessor), (2, 3))

    def test_single_feature_type_tables(self):
        for frame, expected in [(pd.DataFrame({"n": [0, 1]}), ()),
                                (pd.DataFrame({"c": ["a", "b"]}), (0, 1))]:
            preprocessor = make_preprocessor(frame)
            self.assertEqual(preprocessor.fit_transform(frame).dtype, np.float32)
            self.assertEqual(categorical_columns(preprocessor), expected)

    def test_loader_preserves_raw_rows_ids_and_deterministic_labels(self):
        frame = pd.DataFrame({"id": ["row-c", "row-a", "row-b"],
                              "number": [2.0, np.nan, 8.0], "category": ["b", "a", None]},
                             index=[31, 12, 95])
        target = pd.Series(["z", "a", "z"], index=frame.index)
        def get_data(target=None, **kwargs):
            return frame.copy(), None if target is None else target_values.copy(), [], list(frame)
        target_values = target
        raw = SimpleNamespace(name="raw-test", version=7, default_target_attribute="label",
                              row_id_attribute="id", md5_checksum="raw-checksum", get_data=get_data)
        with tempfile.TemporaryDirectory() as directory, \
                patch("openml.datasets.get_dataset", return_value=raw), \
                patch("numpy.load", side_effect=AssertionError("NPZ must not be read")):
            first = load_climb_datasets(dataset_names=["mozilla4"], data_home=directory)[0]
            second = load_climb_datasets(dataset_names=["mozilla4"], data_home=directory)[0]
        pd.testing.assert_frame_equal(first["X"], frame.drop(columns="id"))
        np.testing.assert_array_equal(first["row_ids"], frame["id"].to_numpy())
        np.testing.assert_array_equal(first["y"], [1, 0, 1])
        self.assertEqual(first["metadata"]["version"], 7)
        self.assertEqual(first["metadata"]["raw_fingerprint"], second["metadata"]["raw_fingerprint"])
        json.dumps(first["metadata"])


class SyntheticControlsTests(unittest.TestCase):
    def test_exact_clean_prevalence_and_no_duplicates_for_all_geometries(self):
        for dataset in DATASETS:
            for weight in (0.07, 0.23):
                with self.subTest(dataset=dataset, weight=weight):
                    X, y = create_synthetic_dataset(n_samples=400, dataset=dataset,
                                                    minority_weight=weight, noise_ratio=0, random_state=5)
                    self.assertEqual(int((y == 1).sum()), round(400 * weight))
                    self.assertEqual(len(torch.unique(X, dim=0)), 400)

    def test_nonlinear_clean_labels_follow_original_boundary(self):
        X, y = create_synthetic_dataset(dataset="nonlinear", n_samples=600, noise_ratio=0)
        values = X.numpy()
        latent = (np.sin(values[:, 0]) ** 2 + np.cos(values[:, 1])
                  + np.sin(values[:, 0]) * np.cos(values[:, 1]))
        np.testing.assert_array_equal(y.numpy(), latent >= 0)

    def test_corruption_does_not_change_features_or_other_random_streams(self):
        for dataset in DATASETS:
            clean_X, clean_y = create_synthetic_dataset(dataset=dataset, noise_ratio=0, random_state=17)
            noisy_X, noisy_y = create_synthetic_dataset(dataset=dataset, noise_ratio=0.2, random_state=17)
            torch.testing.assert_close(clean_X, noisy_X, rtol=0, atol=0)
            self.assertEqual(int((clean_y != noisy_y).sum()), 200)

    def test_overlap_changes_features_not_labels(self):
        for dataset in DATASETS:
            X, y = create_synthetic_dataset(dataset=dataset, noise_ratio=0.1, overlap=0, random_state=8)
            overlap_X, overlap_y = create_synthetic_dataset(dataset=dataset, noise_ratio=0.1, overlap=1, random_state=8)
            self.assertFalse(torch.equal(X, overlap_X))
            torch.testing.assert_close(y, overlap_y, rtol=0, atol=0)

    def test_irrelevant_dimensions_do_not_change_informative_features_or_labels(self):
        for dataset in DATASETS:
            X, y = create_synthetic_dataset(dataset=dataset, noise_ratio=0.15, overlap=0.4, random_state=9)
            extra_X, extra_y = create_synthetic_dataset(dataset=dataset, noise_ratio=0.15, overlap=0.4,
                                                       irrelevant_features=5, random_state=9)
            torch.testing.assert_close(X, extra_X[:, :2], rtol=0, atol=0)
            torch.testing.assert_close(y, extra_y, rtol=0, atol=0)
            self.assertEqual(extra_X.shape[1], 7)

    def test_separate_geometry_pools_are_disjoint_and_both_labels_remain_clean(self):
        for dataset in DATASETS:
            split = create_synthetic_split(dataset=dataset, n_samples=500, minority_weight=0.1,
                                            random_state=31)
            self.assertFalse(set(split["train_ids"]) & set(split["test_ids"]))
            train_rows = set(map(tuple, split["X_train"].numpy()))
            test_rows = set(map(tuple, split["X_test"].numpy()))
            self.assertFalse(train_rows & test_rows)
            self.assertEqual(int(split["y_train"].sum()), 40)
            self.assertEqual(int(split["y_test"].sum()), 10)

    def test_caller_corrupts_only_inner_fit_and_records_indices(self):
        split = create_synthetic_split(n_samples=500, random_state=3)
        train_indices, validation_indices = train_test_split(
            np.arange(len(split["y_train"])), test_size=0.2, stratify=split["y_train"], random_state=3
        )
        clean = split["y_train"].clone()
        corrupted, changed = corrupt_labels(clean[train_indices], 0.25, random_state=29, return_indices=True)
        self.assertEqual(len(changed), round(len(train_indices) * 0.25))
        np.testing.assert_array_equal(torch.nonzero(corrupted != clean[train_indices]).flatten().numpy(), changed)
        torch.testing.assert_close(split["y_train"][validation_indices], clean[validation_indices])
        numpy_labels = np.array([0, 1, 0, 1])
        flipped = corrupt_labels(numpy_labels, 1, random_state=0)
        np.testing.assert_array_equal(flipped, 1 - numpy_labels)
        np.testing.assert_array_equal(numpy_labels, [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
