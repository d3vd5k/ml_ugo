import argparse


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from classifiers.factory import make_estimator
from datasets.scripts.climb import load_climb_datasets
from experiments import RunArtifacts, run_fold, seed_everything


def make_estimators(random_state, xgb_device="auto", mlp_device="auto", mlp_epochs=160, algorithms=None):
    names = algorithms or ("logistic_regression", "random_forest", "knn", "svm_rbf", "mlp", "xgboost")
    return {name: make_estimator(name, random_state, device=xgb_device if name == "xgboost" else mlp_device,
                                  mlp_epochs=mlp_epochs) for name in names}


def benchmark_dataset(dataset, estimators, folds, ugo_max_iterations, random_state,
                      device="auto", validation_fraction=.15, artifacts=None):
    X, y = dataset["X"], dataset["y"]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []
    # One split sequence shared by every classifier and sampling strategy.
    for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y), start=1):
        results.extend(run_fold(dataset, estimators, train_indices, test_indices, fold,
                                ugo_max_iterations, random_state + fold, device,
                                validation_fraction, artifacts))
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark UGO on raw CLIMB tabular datasets.")
    parser.add_argument("--dataset", action="append", dest="datasets", help="CLIMB dataset name; repeat to select several")
    parser.add_argument("--target-type", choices=("binary", "multiclass", "all"), default="binary")
    parser.add_argument("--max-datasets", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--ugo-max-iterations", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-home", default=".cache/climb")
    parser.add_argument("--output-dir", default="results/runs")
    parser.add_argument("--random-state", "--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Tensor sampler device")
    parser.add_argument("--xgb-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--mlp-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--mlp-epochs", type=int, default=160)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--algorithm", action="append", choices=("logistic_regression", "random_forest", "knn", "svm_rbf", "mlp", "xgboost"), dest="algorithms")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    seed_everything(args.random_state)
    artifacts = RunArtifacts(args.output_dir, vars(args))
    all_results = []
    try:
        datasets = load_climb_datasets(dataset_names=args.datasets, target_type=args.target_type,
            max_datasets=args.max_datasets if not args.datasets else None, data_home=args.data_home)
        for dataset in datasets:
            if args.max_samples is not None and len(dataset["y"]) > args.max_samples:
                indices, _ = train_test_split(np.arange(len(dataset["y"])), train_size=args.max_samples,
                                             stratify=dataset["y"], random_state=args.random_state)
                dataset = {**dataset, "X": dataset["X"].iloc[indices].copy(),
                           "y": dataset["y"][indices], "row_ids": dataset["row_ids"][indices]}
            artifacts.dataset(dataset)
            print(f"Benchmarking {dataset['name']} ({len(dataset['y'])} raw rows)")
            estimators = make_estimators(args.random_state, args.xgb_device, args.mlp_device,
                                        args.mlp_epochs, args.algorithms)
            all_results.extend(benchmark_dataset(dataset, estimators, args.folds,
                args.ugo_max_iterations, args.random_state, args.device, args.validation_fraction, artifacts))
    except Exception as error:
        artifacts.finish(all_results, error=f"{type(error).__name__}: {error}")
        raise
    artifacts.finish(all_results)
    frame = pd.DataFrame(all_results)
    summary = frame.groupby(["algorithm", "strategy"])[["accuracy", "precision", "recall", "f1", "mcc", "pr_auc"]].mean()
    summary.to_csv(artifacts.path / "summary.csv")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Saved run to {artifacts.path}")


if __name__ == "__main__":
    main()
