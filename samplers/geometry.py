"""Frozen real-reference geometry for connected, categorical-compatible proposals.

Numeric coordinates are already training-preprocessed by the caller. Synthetic
points are queries only: they never update components, pools, or distance trees.
"""
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from samplers.groups import GroupConfig, _fit_groups


def _columns(values, width, name):
    columns = tuple(values)
    if (any(isinstance(c, (bool, np.bool_)) or not isinstance(c, Integral)
            or c < 0 or c >= width for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError(f"{name} must contain distinct valid column indices")
    return tuple(int(c) for c in columns)


def _onehot_valid(points, groups):
    valid = np.ones(len(points), dtype=bool)
    for group in groups:
        block = points[:, group]
        valid &= np.isin(block, (0, 1)).all(axis=1) & (block.sum(axis=1) == 1)
    return valid


def _nonnegative(value, name, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        domain = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {domain}")


class ConstraintGeometry:
    """Reusable geometry fitted exclusively to the supplied real source rows."""

    def __init__(self, X, y, categorical_columns=(), categorical_groups=(),
                 parameters=None, targets=None):
        source, labels = np.asarray(X), np.asarray(y)
        if (source.ndim != 2 or source.shape[1] == 0 or labels.ndim != 1
                or len(source) != len(labels) or not np.issubdtype(source.dtype, np.number)
                or np.iscomplexobj(source) or not np.isfinite(source).all()
                or len(np.unique(labels)) < 2
                or (np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels).all())):
            raise ValueError("Constraint geometry requires finite real features and matching labels with at least two classes")
        columns = _columns(categorical_columns, source.shape[1], "categorical_columns")
        blocks = tuple(_columns(block, source.shape[1], "categorical_groups")
                       for block in categorical_groups)
        flattened = tuple(c for block in blocks for c in block)
        if blocks:
            if any(not block for block in blocks) or len(set(flattened)) != len(flattened):
                raise ValueError("categorical_groups must be disjoint nonempty valid one-hot blocks")
            if columns and set(columns) != set(flattened):
                raise ValueError("categorical_columns must match the supplied one-hot blocks")
            if not _onehot_valid(source, blocks).all():
                raise ValueError("Source categorical_groups must be valid full one-hot blocks")
            columns = tuple(sorted(flattened))
        self.categorical_columns = columns
        self.categorical_groups = blocks
        categorical_set = set(columns)
        self.numeric_columns = tuple(c for c in range(source.shape[1]) if c not in categorical_set)
        parameters = dict(parameters or {})
        graph = GroupConfig(grouping="graph", n_neighbors=parameters.get("graph_neighbors", 5),
                            radius_quantile=parameters.get("graph_radius_quantile", .9))
        self.configuration = {
            "graph_neighbors": int(graph.n_neighbors),
            "graph_radius_quantile": float(graph.radius_quantile),
            "graph_space": "numeric_training_preprocessed",
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "categorical_groups": [list(block) for block in self.categorical_groups],
        }
        with np.errstate(over="ignore", invalid="ignore"):
            self.X = np.array(source, dtype=np.float32, copy=True)
        if not np.isfinite(self.X).all():
            raise ValueError("Source features must remain finite in Float32")
        self.y = labels.copy()
        self.X.setflags(write=False)
        self.y.setflags(write=False)
        self.real_row_ids = None
        self._classes, self._class_ids, counts = np.unique(self.y, return_inverse=True, return_counts=True)
        if targets is None:
            target_classes = set(self._classes.tolist())
        else:
            targets = dict(targets)
            if (set(targets) - set(self._classes.tolist())
                    or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral)
                           for n in targets.values())):
                raise ValueError("targets must contain integer final counts for existing real classes")
            target_classes = set()
            for label, count in zip(self._classes, counts):
                final = targets.get(label, count)
                if final < count:
                    raise ValueError("targets cannot remove real observations")
                if final > count:
                    target_classes.add(label.item())
        self._numeric = self.X[:, self.numeric_columns].astype(np.float64)
        self._trees = [cKDTree(self._numeric[self._class_ids == class_id])
                       for class_id in range(len(self._classes))] if self.numeric_columns else []
        self.groups = []
        for label in self._classes:
            if label.item() not in target_classes:
                continue
            indices = np.flatnonzero(self.y == label)
            component_ids, distinct_counts = _fit_groups(self._numeric[indices], 0, graph)
            for component_id, distinct_count in enumerate(distinct_counts):
                members = indices[component_ids == component_id]
                self.groups.append({"id": -1, "class": label.item(), "indices": members,
                                    "real_support": int(len(members)),
                                    "distinct_coordinates": int(distinct_count)})
        self.groups.sort(key=lambda group: int(group["indices"][0]))
        self.row_group_ids = np.full(len(self.y), -1, dtype=np.int64)
        self.row_pool_ids = np.full(len(self.y), -1, dtype=np.int64)
        self.pools = []
        for group_id, group in enumerate(self.groups):
            group["id"] = group_id
            indices = group["indices"]
            self.row_group_ids[indices] = group_id
            # Stable first-real-row ordering, including duplicate and singleton
            # pools; eligibility and distinct partner support belong to UGO.
            compatible = {}
            for index in indices:
                key = tuple(self.X[index, self.categorical_columns])
                compatible.setdefault(key, []).append(int(index))
            for members in compatible.values():
                pool_id = len(self.pools)
                pool = np.asarray(members, dtype=np.int64)
                self.pools.append(pool)
                self.row_pool_ids[pool] = pool_id
        _, self.seed_opposing_distances = self._reference_distances(
            self._numeric, np.arange(len(self.y)))

    def _reference_distances(self, numeric_points, seeds):
        """Exact tree queries; no dense candidate-by-reference distance matrix."""
        same = np.zeros(len(seeds), dtype=np.float64)
        opposing = np.zeros(len(seeds), dtype=np.float64)
        if not self.numeric_columns:
            return same, opposing
        for class_id in np.unique(self._class_ids[seeds]):
            positions = np.flatnonzero(self._class_ids[seeds] == class_id)
            for start in range(0, len(positions), 512):
                batch = positions[start:start + 512]
                points = numeric_points[batch]
                same[batch] = self._trees[class_id].query(points, k=1, eps=0)[0]
                nearest_other = np.full(len(batch), np.inf)
                for other_id, tree in enumerate(self._trees):
                    if other_id != class_id:
                        np.minimum(nearest_other, tree.query(points, k=1, eps=0)[0], out=nearest_other)
                opposing[batch] = nearest_other
        return same, opposing

    def _queries(self, points, seeds):
        seeds = np.asarray(seeds)
        if (seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer)
                or np.any(seeds < 0) or np.any(seeds >= len(self.y))):
            raise ValueError("seeds must be one-dimensional real source row indices")
        with np.errstate(over="ignore", invalid="ignore"):
            points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape != (len(seeds), self.X.shape[1]):
            raise ValueError("points must match the seed count and real feature width")
        return points, seeds

    def candidate_metrics(self, points, seeds, eta, rho, epsilon):
        """Check actual emitted Float32 coordinates using Float64 numeric norms.

        Boundaries are inclusive, with no tolerance or hidden constraint slack.
        Invalid candidate coordinates fail rather than entering a distance tree.
        """
        _nonnegative(eta, "eta")
        _nonnegative(rho, "rho")
        _nonnegative(epsilon, "epsilon", positive=True)
        points, seeds = self._queries(points, seeds)
        finite = np.isfinite(points).all(axis=1)
        categorical_valid = finite & _onehot_valid(points, self.categorical_groups)
        if self.categorical_columns:
            categorical_valid &= (points[:, self.categorical_columns]
                                  == self.X[seeds][:, self.categorical_columns]).all(axis=1)
        displacement = np.full(len(points), np.inf)
        same = np.full(len(points), np.inf)
        opposing = np.full(len(points), np.inf)
        numeric = points[finite][:, self.numeric_columns].astype(np.float64)
        displacement[finite] = np.linalg.norm(numeric - self._numeric[seeds[finite]], axis=1)
        same[finite], opposing[finite] = self._reference_distances(numeric, seeds[finite])
        seed_opposing = self.seed_opposing_distances[seeds]
        ratio = np.full(len(points), np.inf)
        ratio[finite] = same[finite] / (opposing[finite] + epsilon)
        displacement_pass = finite & (displacement <= eta * seed_opposing)
        ratio_pass = finite & (ratio <= rho)
        return {"displacement": displacement, "seed_opposing_distance": seed_opposing,
                "same_class_distance": same, "opposing_class_distance": opposing,
                "ratio": ratio, "categorical_valid": categorical_valid,
                "displacement_pass": displacement_pass, "ratio_pass": ratio_pass,
                "passes": displacement_pass & ratio_pass & categorical_valid}

    def nearest_in_pool(self, points, seeds, eligible):
        """Nearest eligible real row in the seed's component and exact tuple pool.

        Relocation may return the seed itself; distinct interpolation partners
        are selected separately. Unsupported scopes never fall back to a class.
        """
        points, seeds = self._queries(points, seeds)
        eligible = np.asarray(eligible)
        if eligible.dtype != np.bool_ or eligible.shape != (len(self.y),):
            raise ValueError("eligible must be a boolean mask over real source rows")
        nearest = np.full(len(seeds), -1, dtype=np.int64)
        valid = np.isfinite(points).all(axis=1) & _onehot_valid(points, self.categorical_groups)
        if self.categorical_columns:
            valid &= (points[:, self.categorical_columns]
                      == self.X[seeds][:, self.categorical_columns]).all(axis=1)
        pool_ids = self.row_pool_ids[seeds]
        for pool_id in np.unique(pool_ids[valid & (pool_ids >= 0)]):
            pool = self.pools[pool_id]
            members = pool[eligible[pool]]
            if not len(members):
                continue
            positions = np.flatnonzero(valid & (pool_ids == pool_id))
            if not self.numeric_columns:
                nearest[positions] = members[0]
                continue
            tree = cKDTree(self._numeric[members])
            for start in range(0, len(positions), 512):
                batch = positions[start:start + 512]
                numeric = points[batch][:, self.numeric_columns].astype(np.float64)
                _, indices = tree.query(numeric, k=1, eps=0)
                nearest[batch] = members[indices]
        return nearest

    def snapshot(self):
        """Detached, serialization-ready metadata; distance trees stay private."""
        row_ids = None if self.real_row_ids is None else np.asarray(self.real_row_ids)
        groups = []
        for group in self.groups:
            metadata = {**group, "indices": group["indices"].tolist()}
            if row_ids is not None:
                metadata["row_ids"] = row_ids[group["indices"]].tolist()
            groups.append(metadata)
        return {
            "geometry_configuration": {
                **self.configuration,
                "numeric_columns": list(self.numeric_columns),
                "categorical_columns": list(self.categorical_columns),
                "categorical_groups": [list(block) for block in self.categorical_groups],
            },
            "real_row_ids": None if row_ids is None else row_ids.tolist(),
            "row_group_ids": self.row_group_ids.tolist(),
            "groups": groups,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "categorical_groups": [list(block) for block in self.categorical_groups],
            "seed_opposing_distances": self.seed_opposing_distances.tolist(),
            "synthetic_training_count": 0,
        }


def prepare_constraint_geometry(X, y, categorical_columns=(), categorical_groups=(),
                                parameters=None, targets=None):
    """Prepare deterministic real-only graph scope and exact distance references."""
    return ConstraintGeometry(X, y, categorical_columns, categorical_groups, parameters, targets)
