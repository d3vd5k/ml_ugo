"""Real-only cross-fitted difficulty; no augmented-data or validation feedback."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import make_preprocessor
from experiments import _preprocessing_state


def prepare_oof_partitions(frame, y, row_ids, seed, n_splits=3):
    """Fit transforms separately inside each scoring fold of a real partition."""
    y, row_ids = np.asarray(y), np.asarray(row_ids)
    if not isinstance(frame, pd.DataFrame) or y.ndim != 1 or row_ids.ndim != 1:
        raise ValueError("OOF scoring needs raw features, real labels and row identities")
    if len(frame) != len(y) or len(row_ids) != len(y):
        raise ValueError("OOF rows, labels and identities must align")
    if pd.isna(row_ids).any() or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("OOF real-row identities must be unique and nonmissing")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or n_splits < 2 or counts.min() < n_splits:
        raise ValueError("Each real class must support every OOF fold")
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    partitions = []
    for fold, (train, score) in enumerate(folds.split(frame, y)):
        transform = make_preprocessor(frame.iloc[train])
        X_train = transform.fit_transform(frame.iloc[train])
        X_score = transform.transform(frame.iloc[score])
        partitions.append({"fold": fold, "train_indices": train, "score_indices": score,
                           "train_row_ids": row_ids[train], "score_row_ids": row_ids[score],
                           "X_train": X_train, "X_score": X_score,
                           "preprocessing": _preprocessing_state(transform)})
    return partitions


def oof_mma(partitions, y, algorithm, parameters, seed, settings):
    """Compute fixed seed difficulty with real-only, independently fitted models.

    Scores are computed before generation and reused across its rounds. No
    synthetic row can enter a scoring model, including a held-out descendant.
    Downstream UGO classifiers still retrain on their augmented training rows.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    probabilities = np.full((len(y), len(classes)), np.nan, dtype=np.float32)
    fold_ids = np.full(len(y), -1, dtype=np.int32)
    row_ids = np.empty(len(y), dtype=partitions[0]["score_row_ids"].dtype)
    audits = []
    for partition in partitions:
        train, score = partition["train_indices"], partition["score_indices"]
        if np.intersect1d(train, score).size or np.any(fold_ids[score] != -1):
            raise ValueError("OOF training and scoring rows must be disjoint with unique coverage")
        estimator = make_estimator(algorithm, seed, parameters=parameters,
                                   device=settings["device"], mlp_epochs=settings["mlp_epochs"])
        model = fit_estimator(estimator, partition["X_train"], y[train])
        if not np.array_equal(model.classes_, classes):
            raise ValueError("Every scoring model must contain all real classes")
        probabilities[score] = model.predict_proba(partition["X_score"])
        fold_ids[score] = partition["fold"]
        row_ids[score] = partition["score_row_ids"]
        audits.append({"fold": partition["fold"], "train_row_ids": partition["train_row_ids"],
                       "score_row_ids": partition["score_row_ids"],
                       "preprocessing": partition["preprocessing"], "model_seed": seed,
                       "train_count": len(train), "score_count": len(score),
                       "synthetic_training_count": 0})
    if np.any(fold_ids < 0) or not np.isfinite(probabilities).all():
        raise ValueError("Every real row must have a finite out-of-fold prediction")
    positions = np.searchsorted(classes, y)
    rows = np.arange(len(y))
    competing = probabilities.copy()
    competing[rows, positions] = -np.inf
    scores = competing.max(axis=1) - probabilities[rows, positions]
    audit = {"name": "oof_mma", "n_splits": len(partitions), "training_source": "real_only",
             "refresh": "fixed_before_generation", "synthetic_training_count": 0,
             "classes": classes, "row_ids": row_ids, "fold_ids": fold_ids,
             "scores": scores, "probabilities": probabilities, "folds": audits}
    return scores, audit
