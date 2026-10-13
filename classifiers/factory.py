"""Explicit classifier availability and equal, reproducible tuning opportunities."""
import inspect

import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = (
    "logistic_regression", "random_forest", "xgboost", "lightgbm", "knn",
    "svm_rbf", "mlp", "decision_tree", "adaboost", "balanced_random_forest",
)


def supports_sample_weight(name):
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}")
    return name != "knn"


def make_estimator(name, seed, parameters=None, device="auto", mlp_epochs=160):
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    defaults = {
        "logistic_regression": (LogisticRegression, dict(max_iter=1000, solver="lbfgs", random_state=seed)),
        "random_forest": (RandomForestClassifier, dict(n_estimators=200, random_state=seed, n_jobs=1)),
        "knn": (KNeighborsClassifier, dict(n_neighbors=5, weights="distance", n_jobs=1)),
        "svm_rbf": (SVC, dict(kernel="rbf", probability=True, random_state=seed)),
        "decision_tree": (DecisionTreeClassifier, dict(random_state=seed)),
        "adaboost": (AdaBoostClassifier, dict(random_state=seed)),
    }
    try:
        if name == "xgboost":
            from xgboost import XGBClassifier
            constructor, values = XGBClassifier, dict(n_estimators=200, max_depth=6, learning_rate=.05,
                subsample=.8, colsample_bytree=.8, eval_metric="logloss",
                tree_method="hist", device=selected, random_state=seed, n_jobs=1)
        elif name == "lightgbm":
            from lightgbm import LGBMClassifier
            # Official BAF baseline is CPU LightGBM. CPU-only sklearn controls
            # remain CPU even when the torch / XGBoost device is CUDA.
            constructor, values = LGBMClassifier, dict(n_estimators=200, random_state=seed, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
        elif name == "balanced_random_forest":
            from imblearn.ensemble import BalancedRandomForestClassifier
            constructor, values = BalancedRandomForestClassifier, dict(n_estimators=200, random_state=seed, n_jobs=1, sampling_strategy="all", replacement=True, bootstrap=False)
        elif name == "mlp":
            from experiments import MLPClassifierAdapter
            constructor, values = MLPClassifierAdapter, dict(epochs=mlp_epochs, random_state=seed, device=selected)
        else:
            constructor, values = defaults[name]
    except ImportError as exc:
        raise ImportError(f"Requested classifier {name!r} requires its installed dependency") from exc
    values.update(parameters or {})
    return constructor(**values)


_SPACES = {
    "logistic_regression": {"C": [.01, .1, 1., 10., 100.]},
    "random_forest": {"n_estimators": [100, 200, 400], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 3, 8], "max_features": ["sqrt", .8]},
    "balanced_random_forest": {"n_estimators": [100, 200, 400], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 3, 8], "max_features": ["sqrt", .8]},
    "xgboost": {"n_estimators": [100, 200, 400], "max_depth": [3, 6, 9], "learning_rate": [.01, .05, .1], "subsample": [.7, 1.], "colsample_bytree": [.7, 1.]},
    "lightgbm": {"n_estimators": [100, 200, 400], "num_leaves": [15, 31, 63], "learning_rate": [.01, .05, .1], "min_child_samples": [10, 20, 50]},
    "knn": {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"], "p": [1, 2]},
    "svm_rbf": {"C": [.1, 1., 10., 100.], "gamma": ["scale", .001, .01, .1]},
    "mlp": {"hidden_dim": [20, 64, 128], "lr": [.0003, .001, .003]},
    "decision_tree": {"max_depth": [None, 3, 6, 12], "min_samples_leaf": [1, 3, 8]},
    "adaboost": {"n_estimators": [50, 100, 200], "learning_rate": [.01, .1, 1.]},
}


def candidate_parameters(name, seed, budget, search_space=None):
    """Exactly budget candidates, default first; finite spaces cycle explicitly.

    Sampling with replacement avoids silently reducing a shared search budget
    when a finite classifier space is smaller. Repeated settings still consume
    the declared trial budget. An audited protocol space may override the common
    space (e.g. BAF); its distributions are passed to sklearn ParameterSampler.
    """
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}")
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer")
    space = _SPACES[name] if search_space is None else search_space
    rng = np.random.RandomState(seed)
    candidates = [{}]
    for _ in range(budget - 1):
        candidates.append(next(iter(ParameterSampler(space, n_iter=1, random_state=rng))))
    return candidates


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def fit_estimator(estimator, X, y, sample_weight=None, validation_data=None, epochs=None):
    """Clone, fit, and never reinterpret augmentation as validation."""
    from experiments import MLPClassifierAdapter
    model = clone(estimator)
    if isinstance(model, MLPClassifierAdapter):
        if epochs is not None:
            model.set_params(epochs=int(epochs))
        return model.fit(X, y, sample_weight=sample_weight, validation_data=validation_data)
    kwargs = {}
    if sample_weight is not None:
        if "sample_weight" not in inspect.signature(model.fit).parameters:
            raise TypeError(f"{type(model).__name__} does not support sample_weight")
        kwargs["sample_weight"] = _array(sample_weight)
    return model.fit(_array(X), _array(y), **kwargs)
