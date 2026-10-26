"""Publication figures using the exact six-method implementation embedded in the notebook.

Run from the repository root: python -m reports.synthetic_six_methods
These fixed-seed, fixed-parameter training illustrations are not an F1 benchmark.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("moons", "linear", "circles")
LABELS = {"baseline": "No oversampling", "original_ugo": "Original UGO",
          "u2_flow_gate": "U2 + Flow gate", "smote": "SMOTE",
          "kmeans_smote": "KMeansSMOTE", "rwo": "RWO"}
TITLES = {"moons": "Moons", "linear": "Linear", "circles": "Circles"}
COLORS = {"majority": "#4477AA", "minority": "#CC6677", "synthetic": "#228833"}


@contextmanager
def notebook_runtime(notebook):
    """Load literal bundled sources only; never execute notebook experiment/install cells."""
    if any(name == "portable_ugo" or name.startswith("portable_ugo.") for name in sys.modules):
        raise RuntimeError("Run the figure generator in a fresh Python process.")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    sources = {}
    for cell in cells:
        if "bundled-source" not in cell.get("metadata", {}).get("tags", []):
            continue
        assignment, = ast.parse("".join(cell["source"])).body
        target, = assignment.targets
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != "BUNDLE_SOURCES":
            raise ValueError("Unexpected bundled-source cell")
        name, source = ast.literal_eval(target.slice), ast.literal_eval(assignment.value)
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Expected flat Python source modules")
        sources[name] = source
    if "sampling.py" not in sources or "u2.py" not in sources:
        raise ValueError("The notebook must contain the current six-method bundle")
    with TemporaryDirectory(prefix="ugo_qualitative_") as directory:
        package = Path(directory) / "portable_ugo"
        package.mkdir()
        for name, source in sources.items():
            (package / name).write_text(source, encoding="utf-8")
        sys.path.insert(0, directory)
        try:
            yield importlib.import_module("portable_ugo.benchmark")
        finally:
            sys.path.remove(directory)
            for name in list(sys.modules):
                if name == "portable_ugo" or name.startswith("portable_ugo."):
                    del sys.modules[name]


def make_dataset(name, seed):
    """Exactly 400 majority and 100 minority observations, with no label corruption."""
    if name == "moons":
        X, y = make_moons(n_samples=(400, 100), noise=.12, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=(400, 100), noise=.08, factor=.5, random_state=seed)
    elif name == "linear":
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
            n_redundant=0, n_repeated=0, n_clusters_per_class=1, weights=[.8, .2],
            class_sep=1.5, flip_y=0., random_state=seed)
    else:
        raise ValueError(name)
    if not np.array_equal(np.bincount(y), [400, 100]):
        raise ValueError("Unexpected real-data class counts")
    return pd.DataFrame(X, columns=["x1", "x2"]), np.asarray(y, dtype=np.int64)


def generate(benchmark, seed, device):
    points, counts, audits = [], [], {}
    parameters = {"group_alpha": .5, "f1_tau": .6}
    for dataset in DATASETS:
        frame, y = make_dataset(dataset, seed)
        row_ids = np.arange(len(y))
        preprocessor = benchmark.make_preprocessor(frame)
        X = preprocessor.fit_transform(frame).astype(np.float32)
        targets = {0: 400, 1: 400}
        fit, calibration = train_test_split(row_ids, test_size=.2, stratify=y, random_state=seed + 7919)
        core = benchmark.fit_flow_gmm(frame.iloc[fit], y[fit], row_ids[fit], seed,
            benchmark.FlowGMMConfig(epochs=60, device=device))
        gate = benchmark.FlowGate(core, preprocessor, calibration_row_ids=row_ids[calibration],
            validation_row_ids=np.empty(0, dtype=np.int64))
        gate.role = "qualitative_training_only"
        gate.calibration_class_counts = benchmark._counts(y[calibration])
        audits[dataset] = {"generator": {"moons": "make_moons(noise=0.12)",
            "linear": "make_classification(n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=1.5, flip_y=0)",
            "circles": "make_circles(noise=0.08, factor=0.5)"}[dataset],
            "raw_data_sha256": hashlib.sha256(frame.to_numpy().tobytes() + y.tobytes()).hexdigest(),
            "flow_fit_row_ids": fit.tolist(), "reserved_calibration_row_ids": calibration.tolist(),
            "methods": {}}
        scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        for method in benchmark.METHODS:
            fits = 0
            def model_fn(features, labels):
                nonlocal fits
                model = benchmark.make_estimator("mlp", seed, {}, device=device, epochs=160, n_jobs=4)
                fits += 1
                return benchmark.fit_estimator(model, features, labels)
            if method == "baseline":
                sampled_X, sampled_y = X, y
                audit = {"accepted_count": 0, "requested_count": 0, "stop_reason": "no_oversampling"}
            elif method in benchmark.SAMPLER_PARAMETERS:
                sampled_X, sampled_y, audit = benchmark.resample_control(method, X, y, targets, seed,
                    n_jobs=4, parameters=benchmark.SAMPLER_PARAMETERS[method])
            else:
                if method == "original_ugo":
                    sampled = benchmark.original_ugo(X, y, model_fn,
                        benchmark.OriginalUGOConfig(max_iterations=20, random_state=seed, device=device),
                        target_counts=targets)
                else:
                    sampled = benchmark.run_u2(X, y, model_fn, seed=seed, targets=targets,
                        parameters=parameters, max_iterations=20, device=device, flow_support=gate)
                sampled_X, sampled_y = benchmark.as_array(sampled.X), benchmark.as_array(sampled.y)
                audit = benchmark._audit(sampled, y, row_ids, targets, method != "original_ugo",
                    gate if method == "u2_flow_gate" else None, parameters["f1_tau"])
            np.testing.assert_array_equal(sampled_X[:len(y)], X)
            np.testing.assert_array_equal(sampled_y[:len(y)], y)
            synthetic = sampled_X[len(y):]
            synthetic_y = sampled_y[len(y):]
            if not np.isfinite(sampled_X).all() or not np.all(synthetic_y == 1) or len(synthetic) > 300:
                raise ValueError("Invalid synthetic geometry, labels or augmentation budget")
            # Plot the original sklearn coordinates, not PCA/t-SNE or standardized axes.
            raw_synthetic = scaler.inverse_transform(synthetic) if len(synthetic) else np.empty((0, 2))
            coordinates = np.concatenate((frame.to_numpy(), raw_synthetic))
            points.append(pd.DataFrame({"dataset": dataset, "method": method,
                "origin": ["real"] * len(y) + ["synthetic"] * len(synthetic),
                "class": np.concatenate((y, synthetic_y)), "point_index": np.arange(len(sampled_y)),
                "x1": coordinates[:, 0], "x2": coordinates[:, 1]}))
            counts.append({"dataset": dataset, "method": method,
                "real_majority": 400, "real_minority": 100,
                "requested_synthetic": audit["requested_count"], "actual_synthetic": len(synthetic),
                "final_minority": 100 + len(synthetic), "stop_reason": audit["stop_reason"],
                "classifier_fit_count": fits})
            audits[dataset]["methods"][method] = audit
            print(f"{dataset:7s} | {method:14s} | synthetic {len(synthetic):3d}/{audit['requested_count']:3d} | {audit['stop_reason']}", flush=True)
    return pd.concat(points, ignore_index=True), pd.DataFrame(counts), audits


def axis_limits(data):
    lower = data[["x1", "x2"]].min().to_numpy()
    upper = data[["x1", "x2"]].max().to_numpy()
    center = (lower + upper) / 2
    half = max(upper - lower) * .56
    return [(float(c - half), float(c + half)) for c in center]


def draw_panel(ax, data, count, limits, label, fontsize):
    real = data[data.origin == "real"]
    synthetic = data[data.origin == "synthetic"]
    for value, color, marker, size in [(0, COLORS["majority"], "o", 9), (1, COLORS["minority"], "^", 18)]:
        subset = real[real["class"] == value]
        ax.scatter(subset.x1, subset.x2, c=color, marker=marker, s=size,
            linewidths=0, alpha=.52 if value == 0 else .82, zorder=2 if value == 0 else 4)
    ax.scatter(synthetic.x1, synthetic.x2, c=COLORS["synthetic"], marker="x", s=15,
        linewidths=.65, alpha=.8, zorder=5)
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontsize=fontsize + 1, pad=7)
    generated, requested = int(count.actual_synthetic), int(count.requested_synthetic)
    annotation = "No synthetic points" if count.method == "baseline" else f"Synthetic: {generated}/{requested} requested"
    ax.text(.03, .97, annotation, transform=ax.transAxes, va="top", fontsize=fontsize - 1,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2}, zorder=6)
    ax.tick_params(labelsize=fontsize - 1, length=2.5, width=.6)
    ax.grid(color="#DDDDDD", linewidth=.5, alpha=.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(.6)
        spine.set_color("#888888")
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, fontsize):
    handles = [Line2D([], [], linestyle="", marker=marker, color=COLORS[key], markersize=size, label=label)
        for key, marker, size, label in [("majority", "o", 4, "Real majority (class 0)"),
            ("minority", "^", 5, "Real minority (class 1)"),
            ("synthetic", "x", 5, "Synthetic minority (class 1)")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
        ncol=3, frameon=False, fontsize=fontsize, handletextpad=.5, columnspacing=1.7)


def save_figure(fig, path, dpi):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + extension), dpi=dpi, facecolor="white")
    plt.close(fig)


def render(points, counts, output, dpi):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.transparent": False})
    limits = {dataset: axis_limits(points[points.dataset == dataset]) for dataset in DATASETS}
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8))
        fig.subplots_adjust(left=.08, right=.985, bottom=.13, top=.88, wspace=.35, hspace=.5)
        for index, (ax, method) in enumerate(zip(axes.flat, LABELS)):
            data = points[(points.dataset == dataset) & (points.method == method)]
            count = counts[(counts.dataset == dataset) & (counts.method == method)].iloc[0]
            draw_panel(ax, data, count, limits[dataset], f"({chr(97 + index)}) {LABELS[method]}", 9)
        fig.suptitle(f"{TITLES[dataset]}: qualitative oversampling comparison", fontsize=12, y=.98)
        fig.text(.5, .925, f"Same real data: 400 majority / 100 minority; fixed seed {points.attrs['seed']}",
            ha="center", fontsize=9)
        legend(fig, 8)
        save_figure(fig, output / f"{dataset}_comparison", dpi)
    fig, axes = plt.subplots(3, 6, figsize=(14.4, 8.4))
    fig.subplots_adjust(left=.045, right=.99, bottom=.09, top=.90, wspace=.38, hspace=.48)
    for row, dataset in enumerate(DATASETS):
        for column, method in enumerate(LABELS):
            data = points[(points.dataset == dataset) & (points.method == method)]
            count = counts[(counts.dataset == dataset) & (counts.method == method)].iloc[0]
            draw_panel(axes[row, column], data, count, limits[dataset],
                f"({chr(97 + row * 6 + column)}) {TITLES[dataset]} / {LABELS[method]}", 9)
    fig.suptitle("Synthetic geometry across six oversampling methods", fontsize=15, y=.98)
    fig.text(.5, .94, f"Identical real data within each row; seed {points.attrs['seed']}; 300 requested additions per oversampler; no rejected-point refill", ha="center", fontsize=10)
    legend(fig, 10)
    fig.savefig(output / "overview_preview.png", dpi=140, facecolor="white")
    save_figure(fig, output / "overview", dpi)
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/synthetic_six_methods")
    parser.add_argument("--notebook", type=Path, default=ROOT / "UGO_30_Datasets.ipynb")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Use at least 300 dpi for publication output")
    args.output.mkdir(parents=True, exist_ok=True)
    with notebook_runtime(args.notebook) as benchmark, threadpool_limits(limits=4):
        torch.set_num_threads(4)
        device = benchmark.resolve_device(args.device)
        points, counts, audits = generate(benchmark, args.seed, device)
        points.attrs["seed"] = args.seed
        points.to_csv(args.output / "points.csv", index=False, float_format="%.17g")
        counts.to_csv(args.output / "counts.csv", index=False)
        limits = render(points, counts, args.output, args.dpi)
        manifest = {"purpose": "Fixed-parameter, single-seed training geometry; not held-out performance evidence",
            "seed": args.seed, "device": device, "notebook_sha256": hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
            "method_order": list(LABELS), "scoring_learner": {"name": "mlp", "parameters": {}, "epochs": 160},
            "u2": {"group_alpha": .5, "f1_tau": .6, "max_iterations": 20, "rejected_refill": False},
            "flow": {"epochs": 60, "real_only": True, "reserved_calibration_fraction": .2, "calibration_used": False},
            "sampler_parameters": benchmark.SAMPLER_PARAMETERS, "coordinates": "original sklearn feature coordinates",
            "axis_limits_shared_across_all_methods_per_dataset": limits,
            "raster_dpi": args.dpi, "formats": ["png", "pdf", "svg"],
            "versions": {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "torch", "matplotlib")},
            "audits": audits}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=benchmark._json_default, allow_nan=False) + "\n")
    print(f"Saved publication figures and reproducible point data to {args.output}")


if __name__ == "__main__":
    main()
