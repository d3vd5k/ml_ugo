import argparse

import numpy as np
import pandas as pd
import torch

from datasets.scripts.synthetic import DATASETS, create_synthetic_split
from experiments import MLPClassifierAdapter, RunArtifacts, run_fold, seed_everything



def run_experiment(dataset, n_samples, noise_ratio, show_plots, minority_weight,
                   random_state=42, device="auto", epochs=160, max_iterations=20,
                   validation_fraction=.15, overlap=0., irrelevant_features=0,
                   test_fraction=.2, artifacts=None):
    split = create_synthetic_split(n_samples=n_samples, dataset=dataset,
        minority_weight=minority_weight,
        random_state=random_state, test_fraction=test_fraction,
        overlap=overlap, irrelevant_features=irrelevant_features)
    # Helper generates independent pools, not a split of a reused generator pool.
    train_count = len(split["y_train"])
    X = torch.cat((split["X_train"], split["X_test"])).cpu().numpy()
    y = torch.cat((split["y_train"], split["y_test"])).cpu().numpy()
    data = {"name": dataset, "X": pd.DataFrame(X), "y": y,
            "row_ids": np.concatenate((split["train_ids"], split["test_ids"])),
            "metadata": {"generator": dataset, "random_state": random_state,
                         "minority_weight": minority_weight, "overlap": overlap,
                         "irrelevant_features": irrelevant_features,
                         "label_corruption": noise_ratio, "test_fraction": test_fraction}}
    if artifacts is not None:
        artifacts.dataset(data)
    rows = run_fold(data, {"mlp": MLPClassifierAdapter(epochs=epochs, device=device,
                    random_state=random_state)}, np.arange(train_count),
                    np.arange(train_count, len(y)), 1, max_iterations,
                    random_state, device, validation_fraction, artifacts,
                    noise_ratio=noise_ratio, show_plots=show_plots)
    for row in rows:
        print(f"{dataset} {row['strategy']}: AP={row['pr_auc']:.4f}, F1={row['f1']:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run UGO with isolated real validation and test pools.")
    parser.add_argument("--dataset", choices=DATASETS, default="nonlinear")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--noise-ratio", type=float, default=.2, help="Training-only label corruption")
    parser.add_argument("--overlap", type=float, default=0., help="Geometric feature overlap (not label corruption)")
    parser.add_argument("--irrelevant-features", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--minority-weight", type=float, default=.1)
    parser.add_argument("--random-state", "--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--ugo-max-iterations", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--output-dir", default="results/runs")
    args = parser.parse_args()
    seed_everything(args.random_state)
    artifacts = RunArtifacts(args.output_dir, vars(args))
    rows = []
    try:
        for dataset in DATASETS if args.all_datasets else (args.dataset,):
            rows.extend(run_experiment(dataset, args.n_samples, args.noise_ratio,
                not args.no_plots, args.minority_weight, args.random_state, args.device,
                args.epochs, args.ugo_max_iterations, args.validation_fraction,
                args.overlap, args.irrelevant_features, args.test_fraction, artifacts))
    except Exception as error:
        artifacts.finish(rows, error=f"{type(error).__name__}: {error}")
        raise
    artifacts.finish(rows)
    print(f"Saved run to {artifacts.path}")


if __name__ == "__main__":
    main()
