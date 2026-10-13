"""Explicit LR-scored validation guidance, with unchanged native U2 anchors.

Guidance can overlap outer fitting/generation in this separately authorized
experiment. Tree scores predict LR-surrogate SGD benefit, NOT tree benefit.
"""
from time import perf_counter
from weakref import WeakKeyDictionary

from sklearn.linear_model import LogisticRegression

from classifiers.factory import fit_estimator, make_estimator
from samplers.learning_benefit import CandidatePoolSelector, model_state_hash
from samplers.ugo import score_seeds


class ValidationGuidedLearner:
    """Associate each fitted native checkpoint with its exact training prefix."""
    def __init__(self, algorithm, parameters, seed, surrogate_parameters, validation=None):
        if algorithm not in ("logistic_regression", "random_forest", "lightgbm"):
            raise ValueError("Validation guidance supports LR, RF and LightGBM")
        self.algorithm = algorithm
        self.estimator = make_estimator(algorithm, seed, parameters=parameters, device="cpu")
        self.surrogate_estimator = make_estimator("logistic_regression", seed,
                                                 parameters=surrogate_parameters, device="cpu")
        self.validation = validation
        self.counts = {key: 0 for key in ("operational", "parentdup", "paired", "initial")}
        self.seconds = {key: 0. for key in self.counts}
        self.surrogate_fits = 0
        self.surrogate_seconds = 0.
        self._prefixes = WeakKeyDictionary()
        self._scorers = WeakKeyDictionary()

    def fit(self, X, y, weights=None, kind="operational"):
        start = perf_counter()
        model = fit_estimator(self.estimator, X, y, sample_weight=weights,
                              validation_data=self.validation)
        self.counts[kind] += 1
        self.seconds[kind] += perf_counter() - start
        self._prefixes[model] = (X, y, weights)
        return model

    def scorer_for_model(self, model):
        if self.algorithm == "logistic_regression":
            if not isinstance(model, LogisticRegression):
                raise TypeError("Native LR scoring requires the actual fitted LR")
            return model
        if model not in self._prefixes:
            raise ValueError("No training prefix registered for this native checkpoint")
        if model not in self._scorers:
            X, y, weights = self._prefixes[model]
            start = perf_counter()
            self._scorers[model] = fit_estimator(self.surrogate_estimator, X, y,
                                                sample_weight=weights)
            self.surrogate_seconds += perf_counter() - start
            self.surrogate_fits += 1
        return self._scorers[model]


class ValidationGuidedSelector(CandidatePoolSelector):
    """Reuse exact LR utility/pool rules, preserving actual downstream difficulty."""
    def __init__(self, selector, guidance_X, guidance_y, seed, *, algorithm,
                 scorer_for_model, checkpoint_callback=None, **kwargs):
        if algorithm not in ("logistic_regression", "random_forest", "lightgbm"):
            raise ValueError("Validation guidance supports LR, RF and LightGBM")
        super().__init__(selector, guidance_X, guidance_y, seed, **kwargs)
        self.algorithm = algorithm
        self.scorer_for_model = scorer_for_model
        self.checkpoint_callback = checkpoint_callback
        self.scorer_kind = "native_logistic" if algorithm == "logistic_regression" else "explicit_lr_surrogate"

    def score_seeds(self, model, X, y):
        before = model_state_hash(model)
        start = perf_counter()
        scorer = self.scorer_for_model(model)
        surrogate_seconds = perf_counter() - start
        scorer_scores = super().score_seeds(scorer, X, y)
        if self.algorithm == "logistic_regression":
            if scorer is not model:
                raise ValueError("LR must use its actual downstream model, not a surrogate")
            native_scores = scorer_scores
        else:
            start = perf_counter()
            native_scores = score_seeds(model, X, y)
            native_anchor_seconds = perf_counter() - start
            self._checkpoint["native_anchor_scoring_seconds"] = native_anchor_seconds
        after = model_state_hash(model)
        if before != after:
            raise RuntimeError("Validation scoring mutated the native downstream model")
        self.downstream_model = model
        self._checkpoint.update(downstream_algorithm=self.algorithm, scorer_kind=self.scorer_kind,
            downstream_state_before_hash=before, downstream_state_after_hash=after,
            surrogate_fit_seconds=surrogate_seconds,
            one_step_claim="plain CE SGD on the LR scorer only; not native RF/LightGBM retraining")
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(model, scorer, self._checkpoint)
        return native_scores

    def select(self, real, candidates, cfg, allocation_record):
        result = super().select(real, candidates, cfg, allocation_record)
        after = model_state_hash(self.downstream_model)
        if after != self.rounds[-1]["downstream_state_before_hash"]:
            raise RuntimeError("Candidate scoring mutated the native downstream model")
        self.rounds[-1]["downstream_state_after_hash"] = after
        return result

    def summary(self):
        result = super().summary()
        result.update(downstream_algorithm=self.algorithm, scorer_kind=self.scorer_kind,
            guidance_role="original validation; training information for selection; overlaps outer fitting/generation",
            verification_claim="plain CE SGD on the LR scorer only; no native tree-gradient claim",
            surrogate_fit_seconds_total=sum(row["surrogate_fit_seconds"] for row in self.checkpoints))
        for destination, source in zip(result["rounds"], self.rounds):
            destination.update({key: source[key] for key in ("downstream_algorithm", "scorer_kind",
                "downstream_state_before_hash", "downstream_state_after_hash", "surrogate_fit_seconds")})
        return result
