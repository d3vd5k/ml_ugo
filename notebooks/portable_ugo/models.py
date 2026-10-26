"""Native downstream learners; no experiment/repository runtime dependencies."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.svm import SVC


CLASSIFIERS = ("mlp", "logistic_regression", "xgboost", "random_forest", "svm")
_SPACES = {
    "logistic_regression": {"C": [.01, .1, 1., 10., 100.]},
    "random_forest": {"n_estimators": [100, 200, 400], "max_depth": [None, 6, 12],
                      "min_samples_leaf": [1, 3, 8], "max_features": ["sqrt", .8]},
    "xgboost": {"n_estimators": [100, 200, 400], "max_depth": [3, 6, 9],
                "learning_rate": [.01, .05, .1], "subsample": [.7, 1.],
                "colsample_bytree": [.7, 1.]},
    "svm": {"C": [.1, 1., 10., 100.], "gamma": ["scale", .001, .01, .1]},
    "mlp": {"hidden_dim": [20, 64, 128], "lr": [.0003, .001, .003]},
}


def as_array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def resolve_device(device):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("DEVICE must be 'auto', 'cpu', or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no CPU substitution")
    return "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"


class MLPClassifierAdapter:
    """Root two-hidden-layer ReLU/dropout(.25), full-batch RMSprop, fixed epochs.

    Prediction is chunked, including native MMA calls, to bound temporary VRAM.
    There is deliberately no validation argument or early stopping.
    """
    def __init__(self, *, hidden_dim=20, lr=.001, epochs=160, random_state=42,
                 device="cpu", prediction_batch_size=4096):
        self.hidden_dim, self.lr, self.epochs = hidden_dim, lr, epochs
        self.random_state, self.device = random_state, resolve_device(device)
        self.prediction_batch_size = prediction_batch_size
        self.backend_ = f"pytorch_{self.device}"

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        features = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.classes_ = np.unique(as_array(labels))
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError("MLP labels must be consecutive integers starting at zero")
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], self.hidden_dim), torch.nn.ReLU(),
            torch.nn.Dropout(.25), torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(), torch.nn.Dropout(.25),
            torch.nn.Linear(self.hidden_dim, len(self.classes_))).to(self.device)
        optimizer = torch.optim.RMSprop(self.model_.parameters(), lr=self.lr)
        self.model_.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(self.model_(features), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite MLP loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
        self.model_.eval()
        self.best_epoch_ = self.epochs
        self.training_count_ = len(labels)
        return self

    def predict_proba_tensor(self, X):
        chunks = []
        destination = X.device if isinstance(X, torch.Tensor) else torch.device("cpu")
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(X), self.prediction_batch_size):
                features = torch.as_tensor(X[start:start + self.prediction_batch_size],
                                           dtype=torch.float32, device=self.device)
                chunks.append(torch.softmax(self.model_(features), dim=1).to(destination))
        return (torch.cat(chunks) if chunks else
                torch.empty((0, len(self.classes_)), device=destination))

    def predict_proba(self, X):
        return self.predict_proba_tensor(X).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def candidate_parameters(name, seed, budget, smoke=False):
    """Exactly the declared number, default first; finite spaces sample with replacement."""
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}")
    rng = np.random.RandomState(seed)
    candidates = [{}]
    for _ in range(budget - 1):
        candidates.append(next(iter(ParameterSampler(_SPACES[name], n_iter=1, random_state=rng))))
    if smoke and name in ("random_forest", "xgboost"):
        candidates = [{**item, "n_estimators": 12} for item in candidates]
    return candidates


def make_estimator(name, seed, parameters, *, device, epochs, n_jobs):
    """CPU sklearn controls and genuinely selected Torch/XGBoost CPU or CUDA."""
    selected = resolve_device(device)
    if name == "mlp":
        return MLPClassifierAdapter(epochs=epochs, random_state=seed, device=selected, **parameters)
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed, **parameters)
        backend = "sklearn_cpu"
    elif name == "random_forest":
        values = dict(n_estimators=200, random_state=seed, n_jobs=n_jobs)
        model = RandomForestClassifier(**{**values, **parameters})
        backend = "sklearn_cpu"
    elif name == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=seed, **parameters)
        backend = "sklearn_cpu"
    elif name == "xgboost":
        from xgboost import XGBClassifier
        values = dict(n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8,
                      colsample_bytree=.8, eval_metric="logloss", tree_method="hist",
                      device=selected, random_state=seed, n_jobs=n_jobs)
        model = XGBClassifier(**{**values, **parameters})
        backend = f"xgboost_{selected}"
    else:
        raise ValueError(f"Unknown classifier: {name}")
    model.backend_ = backend
    return model


def fit_estimator(model, X, y):
    if isinstance(model, MLPClassifierAdapter):
        return model.fit(X, y)
    model.fit(as_array(X), as_array(y))
    if model.backend_ == "xgboost_cuda":
        import json
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not str(actual).startswith("cuda"):
            raise RuntimeError(f"XGBoost requested CUDA but fitted on {actual}; refusing silent fallback")
    return model
