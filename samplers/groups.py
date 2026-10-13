"""Real-only subgroup support and subsampling co-membership reliability.

Groups describe eligible support, not proposal or relocation geometry. Stability
measures splitting/merging under real-row subsampling; it is not a noise score.
"""
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class GroupConfig:
    grouping: str = "kmeans"
    n_clusters: int = 8
    n_neighbors: int = 5
    repeats: int = 5
    sample_fraction: float = .8
    radius_quantile: float = .9

    def __post_init__(self):
        if self.grouping not in ("kmeans", "graph"):
            raise ValueError("grouping must be kmeans or graph")
        for name in ("n_clusters", "n_neighbors", "repeats"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, lower_open in (("sample_fraction", True), ("radius_quantile", False)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value <= 1
                    or (lower_open and value == 0)):
                interval = "(0, 1]" if lower_open else "[0, 1]"
                raise ValueError(f"{name} must be finite and in {interval}")


def _graph_labels(coordinates, config):
    """Sparse mutual kNN, with bounded-memory neighbor queries."""
    count = len(coordinates)
    if count == 1:
        return np.zeros(1, dtype=np.int64)
    k = min(config.n_neighbors, count - 1)
    # Explicit tree search avoids a dense all-pairs distance matrix even for
    # high-dimensional inputs. Deduplication makes the first neighbor self.
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coordinates)
    batch_size = max(1, min(512, 1_000_000 // (k + 1)))
    kth_distances = np.empty(count, dtype=float)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        distances, _ = neighbors.kneighbors(coordinates[start:stop])
        kth_distances[start:stop] = distances[:, -1]
    radius = np.quantile(kth_distances, config.radius_quantile)
    blocks = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        distances, indices = neighbors.kneighbors(coordinates[start:stop])
        retained = distances[:, 1:] <= radius
        row, column = np.nonzero(retained)
        blocks.append(csr_matrix((np.ones(len(row), dtype=bool), (row, indices[:, 1:][row, column])),
                                 shape=(stop - start, count)))
    directed = vstack(blocks, format="csr")
    mutual = directed.multiply(directed.T)
    return connected_components(mutual, directed=False, return_labels=True)[1]


def _fit_groups(rows, seed, config):
    """Deduplicate for geometry, then restore every real observation."""
    coordinates, inverse = np.unique(rows, axis=0, return_inverse=True)
    if config.grouping == "kmeans":
        unique_labels = KMeans(n_clusters=min(config.n_clusters, len(coordinates)),
                               n_init=10, random_state=seed).fit_predict(coordinates)
    else:
        unique_labels = _graph_labels(coordinates, config)
    row_labels = unique_labels[inverse]
    labels, first_indices = np.unique(row_labels, return_index=True)
    canonical = labels[np.argsort(first_indices)]
    remap = np.empty(int(unique_labels.max()) + 1, dtype=np.int64)
    remap[canonical] = np.arange(len(canonical))
    return remap[row_labels], np.bincount(remap[unique_labels], minlength=len(canonical))


def _co_membership_jaccard(base_labels, replica_labels, n_groups):
    """Count incident pairs algebraically, including replica-cluster outsiders.

    Both label arrays cover the same observed real rows. No pair sets or dense
    co-membership matrix are constructed.
    """
    observed = np.bincount(base_labels, minlength=n_groups)
    evaluable = observed >= 2
    ratios = np.zeros(n_groups, dtype=float)
    if not len(base_labels):
        return ratios, evaluable
    replica_counts = np.bincount(replica_labels)
    replica_count = len(replica_counts)
    cells, members = np.unique(base_labels * replica_count + replica_labels, return_counts=True)
    group_ids, replica_ids = np.divmod(cells, replica_count)
    intersection = np.zeros(n_groups, dtype=np.int64)
    incident = np.zeros(n_groups, dtype=np.int64)
    np.add.at(intersection, group_ids, members * (members - 1) // 2)
    # choose2(total) - choose2(total - members), without subtracting two
    # potentially large squared totals.
    np.add.at(incident, group_ids, members * (2 * replica_counts[replica_ids] - members - 1) // 2)
    base_pairs = observed * (observed - 1) // 2
    union = base_pairs + incident - intersection
    ratios[evaluable] = intersection[evaluable] / union[evaluable]
    return ratios, evaluable


def prepare_groups(X, y, target_labels, seed, config):
    """Fit real support groups and independently refit real-row subsamples.

    Subsampling is without replacement within each target class, using
    ceil(sample_fraction * real_support) rows. Duplicate coordinates contribute
    their original row multiplicity to support and stability, never extra
    coordinates. Groups with fewer than two distinct coordinates are explicit
    but cannot provide interpolative support.
    """
    if not isinstance(config, GroupConfig):
        raise TypeError("config must be a GroupConfig")
    X, y = np.asarray(X), np.asarray(y)
    if (X.ndim != 2 or X.shape[1] == 0 or y.ndim != 1 or len(X) != len(y)
            or not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all()):
        raise ValueError("Grouping requires finite real features and matching one-dimensional labels")
    target_labels = np.asarray(list(target_labels))
    rng = np.random.default_rng(seed)
    groups = []
    for label in np.unique(y):
        if not np.any(target_labels == label):
            continue
        source_indices = np.flatnonzero(y == label)
        rows = X[source_indices]
        base_labels, distinct_counts = _fit_groups(rows, seed, config)
        n_groups = len(distinct_counts)
        stability_sum = np.zeros(n_groups, dtype=float)
        evaluable_counts = np.zeros(n_groups, dtype=np.int64)
        sample_size = int(np.ceil(config.sample_fraction * len(rows)))
        for _ in range(config.repeats):
            sampled = np.sort(rng.choice(len(rows), size=sample_size, replace=False))
            repeat_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            replica_labels, _ = _fit_groups(rows[sampled], repeat_seed, config)
            ratios, evaluable = _co_membership_jaccard(base_labels[sampled], replica_labels, n_groups)
            stability_sum += ratios
            evaluable_counts += evaluable
        stability = np.divide(stability_sum, evaluable_counts, out=np.zeros(n_groups),
                              where=evaluable_counts > 0)
        for group_id in range(n_groups):
            indices = source_indices[base_labels == group_id]
            groups.append({"id": -1, "class": label.item(), "indices": indices,
                           "real_support": int(len(indices)),
                           "distinct_coordinates": int(distinct_counts[group_id]),
                           "stability": float(stability[group_id]),
                           "evaluable_repeats": int(evaluable_counts[group_id])})
    groups.sort(key=lambda group: int(group["indices"][0]))
    row_group_ids = np.full(len(y), -1, dtype=np.int64)
    for group_id, group in enumerate(groups):
        group["id"] = group_id
        row_group_ids[group["indices"]] = group_id
    return {"row_group_ids": row_group_ids, "groups": groups,
            "configuration": asdict(config), "synthetic_training_count": 0}
