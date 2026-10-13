import numpy as np
import torch
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons


DATASETS = ("linear", "nonlinear", "moons", "circles", "blobs", "classification")


def _seed_sequence(random_state):
    return random_state if isinstance(random_state, np.random.SeedSequence) else np.random.SeedSequence(random_state)


def _validate_controls(n_samples, minority_weight, noise_ratio, overlap, irrelevant_features):
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least two.")
    if not 0 < minority_weight < 0.5:
        raise ValueError("minority_weight must be between zero and 0.5.")
    if not 0 <= noise_ratio <= 1:
        raise ValueError("noise_ratio must be between zero and one.")
    if not np.isfinite(overlap) or overlap < 0:
        raise ValueError("overlap must be a finite, nonnegative feature-noise scale.")
    if not isinstance(irrelevant_features, (int, np.integer)) or irrelevant_features < 0:
        raise ValueError("irrelevant_features must be a nonnegative integer.")
    if round(n_samples * minority_weight) < 1:
        raise ValueError("n_samples and minority_weight must yield at least one minority observation.")


def _geometry_pool(dataset, n_samples, rng):
    """Pools depend on geometry/size, never the requested class prevalence."""
    pool_size = 4 * n_samples
    seed = int(rng.integers(0, 2**31 - 1))
    if dataset == "linear":
        X = np.vstack((rng.normal(0.0, 3.0, (n_samples, 2)),
                       rng.normal(6.0, 2.0, (n_samples, 2))))
        return X, np.repeat([0, 1], n_samples)
    if dataset == "nonlinear":
        # Rejection sampling: every selected point is a distinct original draw.
        # Gathering enough of both classes avoids prevalence-driven duplication.
        pools = [[], []]
        counts = np.zeros(2, dtype=int)
        while np.any(counts < n_samples):
            X = rng.normal(0.0, 1.0, (pool_size, 2))
            latent = np.sin(X[:, 0]) ** 2 + np.cos(X[:, 1]) + np.sin(X[:, 0]) * np.cos(X[:, 1])
            y = (latent >= 0).astype(np.int64)
            for label in (0, 1):
                rows = X[y == label]
                pools[label].append(rows)
                counts[label] += len(rows)
        X = np.vstack([np.concatenate(rows) for rows in pools])
        return X, np.repeat([0, 1], counts)
    if dataset == "moons":
        return make_moons(n_samples=pool_size, noise=0.2, random_state=seed)
    if dataset == "circles":
        return make_circles(n_samples=pool_size, noise=0.12, factor=0.45, random_state=seed)
    if dataset == "blobs":
        return make_blobs(n_samples=pool_size, centers=[(-2.5, -1.5), (2.5, 2.0)],
                          cluster_std=[1.8, 1.2], random_state=seed)
    if dataset == "classification":
        return make_classification(n_samples=pool_size, n_features=2, n_redundant=0,
                                   n_clusters_per_class=2, class_sep=0.9,
                                   weights=[0.5, 0.5], flip_y=0.0, random_state=seed)
    raise ValueError(f"Unknown synthetic dataset: {dataset}")


def _clean_dataset(n_samples, dataset, minority_weight, seed, overlap, irrelevant_features, geometry_pool=None):
    # Fixed slots ensure one control cannot consume another control's randomness.
    geometry_seed, selection_seed, overlap_seed, corruption_seed, irrelevant_seed = seed.spawn(5)
    X_pool, y_pool = (_geometry_pool(dataset, n_samples, np.random.default_rng(geometry_seed))
                      if geometry_pool is None else geometry_pool)
    selection = np.random.default_rng(selection_seed)
    n_minority = int(round(n_samples * minority_weight))
    indices = np.concatenate([
        selection.choice(np.flatnonzero(y_pool == 0), n_samples - n_minority, replace=False),
        selection.choice(np.flatnonzero(y_pool == 1), n_minority, replace=False),
    ])
    selection.shuffle(indices)
    X, y = X_pool[indices], y_pool[indices]
    # Feature perturbations produce genuine class-distribution overlap; unlike
    # corruption they do not change the clean ground-truth label.
    if overlap:
        X = X + np.random.default_rng(overlap_seed).normal(0.0, overlap, X.shape)
    if irrelevant_features:
        irrelevant = np.random.default_rng(irrelevant_seed).normal(size=(irrelevant_features, n_samples)).T
        X = np.column_stack((X, irrelevant))
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)), corruption_seed


def corrupt_labels(y, noise_ratio, random_state=42, *, return_indices=False):
    """Flip a fraction of binary training labels using a dedicated RNG stream.

    Call only AFTER splitting clean real labels for inner fit/validation. The
    input is never modified; tensor device and array/tensor type are preserved.
    """
    if not 0 <= noise_ratio <= 1:
        raise ValueError("noise_ratio must be between zero and one.")
    values = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("corrupt_labels requires a one-dimensional binary label vector.")
    rng = np.random.default_rng(_seed_sequence(random_state))
    indices = np.sort(rng.choice(len(values), int(round(len(values) * noise_ratio)), replace=False))
    corrupted = y.clone() if isinstance(y, torch.Tensor) else values.copy()
    if isinstance(y, torch.Tensor):
        tensor_indices = torch.as_tensor(indices, device=y.device)
        corrupted[tensor_indices] = 1 - corrupted[tensor_indices]
    else:
        corrupted[indices] = 1 - corrupted[indices]
    return (corrupted, indices) if return_indices else corrupted


def create_synthetic_dataset(
    n_samples=1000,
    noise_ratio=0.3,
    dataset="nonlinear",
    minority_weight=0.1,
    random_state=42,
    *,
    overlap=0.0,
    irrelevant_features=0,
):
    """Return X/y tensors with independent prevalence, overlap and corruption.

    Clean class 1 prevalence is exactly round(n_samples * minority_weight).
    noise_ratio controls label flips, overlap controls extra Gaussian feature
    perturbation, and irrelevant_features adds independent standard normals.
    For experiments, use create_synthetic_split and corrupt inner fit only.
    """
    _validate_controls(n_samples, minority_weight, noise_ratio, overlap, irrelevant_features)
    X, clean_y, corruption_seed = _clean_dataset(
        n_samples, dataset, minority_weight, _seed_sequence(random_state), overlap, irrelevant_features
    )
    return X, corrupt_labels(clean_y, noise_ratio, corruption_seed)


def create_synthetic_split(
    n_samples=1000,
    dataset="nonlinear",
    minority_weight=0.1,
    random_state=42,
    *,
    test_fraction=0.2,
    overlap=0.0,
    irrelevant_features=0,
):
    """Generate separate clean train/test pools, not subsets of reused draws.

    Callers stratify clean training labels into inner fit and validation, then
    apply corrupt_labels only to inner fit. Validation and naturally imbalanced
    test targets remain clean.
    n_samples denotes the combined train/test count.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one.")
    n_test = int(round(n_samples * test_fraction))
    n_train = n_samples - n_test
    for size in (n_train, n_test):
        _validate_controls(size, minority_weight, 0, overlap, irrelevant_features)
    train_seed, test_seed, shared_geometry_seed = _seed_sequence(random_state).spawn(3)
    train_pool = test_pool = None
    if dataset == "classification":
        # make_classification samples its latent clusters/covariances too. A
        # shared latent model with disjoint candidate pools avoids train/test
        # distribution shift without ever repeating an observation.
        X_pool, y_pool = _geometry_pool(dataset, n_samples, np.random.default_rng(shared_geometry_seed))
        train_indices, test_indices = [], []
        for label in (0, 1):
            indices = np.flatnonzero(y_pool == label)
            boundary = int(round(len(indices) * n_train / n_samples))
            train_indices.extend(indices[:boundary])
            test_indices.extend(indices[boundary:])
        train_pool = X_pool[train_indices], y_pool[train_indices]
        test_pool = X_pool[test_indices], y_pool[test_indices]
    X_train, y_train, _ = _clean_dataset(n_train, dataset, minority_weight, train_seed, overlap,
                                         irrelevant_features, train_pool)
    X_test, y_test, _ = _clean_dataset(n_test, dataset, minority_weight, test_seed, overlap,
                                       irrelevant_features, test_pool)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_ids": np.asarray([f"train:{index}" for index in range(n_train)]),
        "test_ids": np.asarray([f"test:{index}" for index in range(n_test)]),
    }
