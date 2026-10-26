"""Build the standalone notebook; users need only the emitted .ipynb file."""
import ast
import hashlib
import json
from pathlib import Path
import pprint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "UGO_30_Datasets.ipynb"

INSTALLER = '''# Run this FIRST in the notebook's Python kernel (Python 3.11 or newer).
# These are all pipeline dependencies. No repository checkout or local .py files are needed.
# On Linux/Windows, pip's standard PyTorch wheels support CUDA where available.
# For a specific CUDA wheel, set TORCH_INDEX_URL using https://pytorch.org/get-started/locally/.
import sys
import subprocess

if sys.version_info < (3, 11):
    raise RuntimeError("Use a Python 3.11+ kernel for SciPy's Welch ANOVA support.")
TORCH_INDEX_URL = None  # e.g. a CUDA wheel index from the official PyTorch selector
if TORCH_INDEX_URL:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torch>=2.6,<3",
                           "--index-url", TORCH_INDEX_URL])
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "numpy>=1.26,<3", "pandas>=2.2,<4", "scipy>=1.16,<2", "scikit-learn>=1.6,<2",
    "torch>=2.6,<3", "xgboost>=3,<4", "openml>=0.15,<0.16", "threadpoolctl>=3.1,<4",
    "imbalanced-learn>=0.14,<0.15", "ipython>=8,<10"])
print("Dependencies installed. If an already imported package was upgraded, restart the kernel before continuing.")
'''

INTRO = '''# Standalone six-method oversampling comparison

This notebook contains the full executable pipeline. **Only this notebook is needed**; it embeds its Python source and materializes a private temporary package. Internet is needed for the first dependency installation and uncached OpenML datasets, not to download pipeline code.

## Dataset source and selection

The catalogue contains **32 binary CLIMB/imbalanced-ensemble entries**, retaining the original dataset numbers 1–30. Datasets **10** (Diabetes-130-Hospitals) and **26** (Click_prediction_small) are too large for this run and are skipped by default. Two small raw OpenML datasets are appended: **31 = haberman (306 rows, 3 predictors)** and **32 = blood-transfusion-service-center (748 rows, 4 predictors)**. `DATASET_RANGE="1-32"` with `EXCLUDED_DATASETS=[10, 26]` selects **30 active datasets**. To run only the additions, use `"31-32"`; other inclusive ranges such as `"4-9"` still work without renumbering. Set `EXCLUDED_DATASETS=[]` only if you deliberately want to allow the large datasets again. These are CLIMB sources, not imbalanced-learn's separate 27-dataset collection. Original OpenML binary targets are preserved, never silently relabelled one-versus-rest, and globally preprocessed package arrays are not used.

## Compared methods and classifiers

- `baseline`: train on real data only, no oversampling or class weighting.
- `original_ugo`: the paper-derived original UGOSMOTE implementation, with its documented ambiguity resolutions and synthetic-parent behavior. This is not corrected UGO or a verified author-code reproduction.
- `u2_flow_gate`: real-only KMeans/stability allocation, capped group/anchor quotas, native modified-margin difficulty, same-class real-parent interpolation, and frozen compatibility-only FlowGMM rejection. The gate replaces post-neighbor detection **and relocation**, rejects without refill, and uses equal priors. No historical U1 composition, downweighting, F2, utility selection or learned generator is included.
- `smote`: standard [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), with a declared whole-category projection for mixed features (not silently substituted SMOTENC).
- `kmeans_smote`: [imbalanced-learn KMeansSMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.KMeansSMOTE.html), with the same category projection. Its prespecified cluster-balance threshold is 0.0 so rare-positive clusters are eligible; clusters still need enough minority neighbors. Infeasible cases are recorded as failures, never replaced with another sampler.
- `rwo`: Random Walk Oversampling, not random duplication: uniform real-class anchors plus independent numeric Gaussian steps `std(ddof=0) / sqrt(n_real_class)`. Nominal variables use class-conditional frequency/roulette sampling of whole one-hot categories. Zero-variance numeric features remain constant; no clipping or rounding is added. Numeric convention follows the [Python RWO reference](https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_rwo_sampling.py); nominal handling follows the [RWO mixed-feature documentation](https://github.com/cran/imbalance/blob/master/R/rwo.R). The independently written notebook implementation is not a bitwise reproduction of that R implementation, whose numeric scaling convention differs.

**MLP, logistic regression, XGBoost, random forest and RBF SVM** are included. PyTorch MLP, FlowGMM and dense neighbor distances use CUDA when available, as does XGBoost. Standard sklearn logistic regression, random forest and SVM are CPU estimators; the notebook reports this explicitly rather than claiming GPU execution. `DEVICE="cpu"` works without an NVIDIA GPU. Driver/toolkit installation is a system prerequisite, not something pip can supply. See [PyTorch installation](https://pytorch.org/get-started/locally/).

SMOTE and KMeansSMOTE use their actual CPU imbalanced-learn implementations; vectorized RWO also runs on CPU. Their downstream MLP/XGBoost classifier still uses CUDA when selected. Resampling uses only the current real training role; validation and test observations are never oversampled. Fixed sampler parameters are editable below and are shared across classifiers. New controls use one-shot resampling, not UGO's iterative retraining.

## Evaluation and portability contract

Split real data first. All imputation/scaling/encoding fit inside the corresponding training role. Minority-class **F1** selects native model settings, U2 alpha, gate tau and the final classifier decision threshold using real inner validation only. AP is secondary. Equal F1 thresholds prefer closest to 0.5, then higher. The gate threshold `tau` and classifier decision threshold are distinct. SVM's probability estimation uses its fitting data, not the outer test. MLP uses fixed epochs, not validation-driven early stopping. The minority label is identified from real outer training, including label 0.

FlowGMM is fitted on a stratified 80% of real inner-fit rows; the remaining 20% is reserved and unused. Validation and outer test never fit the detector. Its preprocessing/weights remain frozen during generation and outer replay; only the downstream classifier/preprocessor refit on full outer training after selection. Synthetic observations never become U2 support or parents. Original UGO intentionally preserves its original descendant behavior. Category-safe partner projection for original UGO is a declared mixed-data adapter deviation.

The common synthetic target allowance is 1,000 rows, clipped at real majority balance. Original/gated UGO also bound proposals and may stop below that target; one-shot SMOTE/RWO aim at the declared final class counts. KMeansSMOTE can round cluster allocations upward: only excess synthetic rows are deterministically trimmed and recorded, never refilled or padded. U2 real-support caps and gate rejection can produce substantial shortfalls. Do not interpret a common allowance as count-matched generation quality. `MAX_ROUNDS` applies to UGO, not the three one-shot controls. Fixed 0.5 F1, precision, recall, AP, balanced accuracy, MCC, counts, native fits, runtime and backend are also reported. Seeds are repeated-split descriptive evidence, not independent-fold significance.

**Run a smoke first:** set `SMOKE=True` and a small range. It uses only original outer-training rows, shorter epochs and smaller budgets; its results are not scientific evidence. For the full study, use `SMOKE=False` and `MAX_ROWS=None`. Changing `MAX_ROWS` is an explicit development subsample and is recorded; no automatic subsampling occurs. All 30 datasets × 5 classifiers × 3 seeds × 6 methods means **2,700 reported scientific cells**, plus native tuning and internal canonical-U2 fits. RBF SVM and exact neighbors can be expensive on large datasets. Run separate numbered ranges when appropriate.

Outputs go to a unique directory under `OUTPUT_DIR`: `comparison.csv` shows all six methods side by side with F1 and seed coverage; `summary.csv` includes every metric; `paired_deltas.csv` includes all 15 unordered method pairs on matching splits. Compact predictions, selections, dataset/split identities and explicit failures are also retained. No trained model checkpoints, augmented arrays or raw proposal-coordinate logs are saved. Missing cells remain visibly incomplete, not fake metrics, and their means must not be ranked as complete-study results. Keep seeds/splits and configuration identical when combining range runs, and never average duplicate cells twice.

Cell runtime excludes the shared FlowGMM fit, which is recorded once per dataset/seed in `manifest.json` under `shared_flow_fits`; per-cell shared-fit seconds are references to that same fit and must not be summed across classifiers. The three new control pools are generated once per dataset/seed/role/method and reused across native candidates/classifiers: only the first consumer's cell wall time includes pool creation. Audit `shared_generation_seconds` is a shared reference, not a new fit every time it appears. Use manifest total elapsed time alongside fit counts rather than treating all cell timings as independently repeated end-to-end costs.

**Verification:** appended datasets 31/32 completed **60 CUDA smoke cells outside the repository**, covering both additions across all five classifiers and six methods. Six predictive metrics per cell, all 60 validation-selected thresholds and all 150 matched method pairs were independently recomputed. Five range/exclusion regressions passed, and original IDs 1–30 were checked unchanged. Earlier core evidence includes 60 CPU and 60 CUDA smoke cells, numerical sampler regressions, U2 native/gated/all-rejected parity and the original 30 source-target checks. The unchanged dependency cell was reused rather than reinstalled. The full 2,700-cell default study, clean-machine dependency installation and Windows/macOS execution have not been exercised.
'''

CONFIGURATION = '''# EDIT THIS CELL. Numbering is 1-based and both endpoints are included.
DATASET_RANGE = "1-32"     # "31-32" runs only the two small additions; endpoints are inclusive
EXCLUDED_DATASETS = [10, 26]  # Keeps 30 active datasets without changing their original IDs
CLASSIFIERS = ["mlp", "logistic_regression", "xgboost", "random_forest", "svm"]
SEEDS = [42, 43, 44]
SMOKE = False              # True = short training-only integration check, NOT study results
DEVICE = "auto"            # "auto", "cpu", or "cuda" (explicit cuda requires working CUDA)
DATA_HOME = "ugo_data"      # Persistent raw OpenML cache; safe to reuse across ranges
OUTPUT_DIR = "ugo_results"
MAX_ROWS = None             # None uses every row; a number is an explicit development subset
N_CANDIDATES = 10
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
AUGMENTATION_CAP = 1000
MAX_ROUNDS = 20
MLP_EPOCHS = 160
FLOW_EPOCHS = 60
GROUP_ALPHAS = [0.0, 0.5, 1.0]
GATE_TAUS = [0.5, 0.6, 0.7, 0.8]
N_JOBS = 4
# Prespecified sampler settings: no automatic neighbor weakening or sampler fallback.
SAMPLER_PARAMETERS = {
    "smote": {"k_neighbors": 5},
    "kmeans_smote": {"k_neighbors": 2, "kmeans_estimator": 8,
                     "cluster_balance_threshold": 0.0, "density_exponent": "auto"},
    "rwo": {},
}

CONFIG = {key: globals()[key] for key in (
    "DATASET_RANGE", "EXCLUDED_DATASETS", "CLASSIFIERS", "SEEDS", "SMOKE", "DEVICE", "DATA_HOME", "OUTPUT_DIR",
    "MAX_ROWS", "N_CANDIDATES", "TEST_FRACTION", "VALIDATION_FRACTION", "AUGMENTATION_CAP",
    "MAX_ROUNDS", "MLP_EPOCHS", "FLOW_EPOCHS", "GROUP_ALPHAS", "GATE_TAUS", "N_JOBS",
    "SAMPLER_PARAMETERS")}
# Optional programmatic overrides are useful for automated notebook execution.
CONFIG.update(globals().get("NOTEBOOK_OVERRIDES", {}))
'''

BOOTSTRAP = '''# Materialize this notebook's embedded code, not code from the current directory.
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

BUNDLE_SHA256 = hashlib.sha256(json.dumps(BUNDLE_SOURCES, sort_keys=True).encode()).hexdigest()
BUNDLE_ROOT = Path(tempfile.mkdtemp(prefix="ugo_notebook_"))
PACKAGE_ROOT = BUNDLE_ROOT / "portable_ugo"
PACKAGE_ROOT.mkdir()
for relative, source in BUNDLE_SOURCES.items():
    destination = PACKAGE_ROOT / relative
    destination.write_text(source, encoding="utf-8")
# Rerunning cells replaces only this notebook's own package namespace.
for name in list(sys.modules):
    if name == "portable_ugo" or name.startswith("portable_ugo."):
        del sys.modules[name]
sys.path.insert(0, str(BUNDLE_ROOT))
importlib.invalidate_caches()
from portable_ugo.benchmark import run_benchmark, select_catalogue, show_catalogue
print("Bundled pipeline SHA256:", BUNDLE_SHA256)
print("Private runtime package:", PACKAGE_ROOT)
'''


class PortableImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("samplers."):
            node.module = "portable_ugo." + node.module.split(".", 1)[1]
        elif node.module == "datasets.scripts.climb":
            node.module = "portable_ugo.preprocessing"
        return node


def portable_source(path):
    original = path.read_text(encoding="utf-8")
    tree = PortableImports().visit(ast.parse(original))
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree) + "\n"
    return transformed, hashlib.sha256(original.encode()).hexdigest()


def preprocessing_source():
    tree = ast.parse((ROOT / "datasets/scripts/climb.py").read_text())
    names = {"_normalize_missing", "_dense_float32", "make_preprocessor", "categorical_columns"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("Required real-only preprocessing functions are missing")
    imports = '''import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
'''
    return imports + "\n\n" + "\n\n".join(ast.unparse(node) for node in functions) + "\n"


def cell(kind, source, tags=()):
    result = {"cell_type": kind, "metadata": {"tags": list(tags)}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def source_cell(name, source):
    quote = "'''" if "'''" not in source else '\"\"\"'
    if quote in source:
        raise ValueError(f"No readable triple-quote wrapper available for {name}")
    return cell("code", f"# Bundled module: {name}\nBUNDLE_SOURCES[{name!r}] = r{quote}{source}{quote}\n", ["bundled-source"])


def build():
    catalogue = json.loads((ROOT / "notebooks/portable_catalogue.json").read_text())
    if len(catalogue) != 32 or [row["number"] for row in catalogue] != list(range(1, 33)):
        raise RuntimeError("Catalogue must have exactly 32 sequential numbered entries")
    sources = {"__init__.py": '"""Self-contained UGO notebook runtime."""\n',
               "preprocessing.py": preprocessing_source()}
    provenance = {}
    for name in ("ugo", "ugo_original", "groups", "allocation", "flow_gmm"):
        sources[name + ".py"], provenance[name] = portable_source(ROOT / "samplers" / (name + ".py"))
    for name in ("u2", "models", "sampling", "benchmark"):
        path = ROOT / "notebooks/portable_ugo" / (name + ".py")
        if name == "models" and not path.exists():
            continue
        sources[name + ".py"] = path.read_text(encoding="utf-8")
    for name, source in sources.items():
        ast.parse(source, filename=name)
    cells = [cell("code", INSTALLER, ["dependencies"]), cell("markdown", INTRO),
             cell("code", CONFIGURATION, ["configuration"]),
             cell("markdown", "## Bundled implementation\n\nThese readable source cells are part of the notebook; run them in order. They require no checkout, remote source download or sidecar modules. The core sampler sources are vendored from the tested repository implementation; preprocessing retains training-role isolation.\n"),
             cell("code", "BUNDLE_SOURCES = {}\n")]
    for name, source in sources.items():
        cells.append(source_cell(name, source))
    cells.append(cell("code", BOOTSTRAP, ["bootstrap"]))
    cells.append(cell("code", "# Fixed, explicit CLIMB / raw OpenML catalogue.\nCATALOGUE = " + pprint.pformat(catalogue, width=110, sort_dicts=False) +
                      '\nfrom IPython.display import display\ndisplay(show_catalogue(CATALOGUE).assign(default_run=lambda rows: ~rows["number"].isin(CONFIG["EXCLUDED_DATASETS"])))\nSELECTED_DATASETS = select_catalogue(CATALOGUE, CONFIG["DATASET_RANGE"], excluded_numbers=CONFIG["EXCLUDED_DATASETS"])\nprint("Selected:", [(row["number"], row["name"]) for row in SELECTED_DATASETS])\nprint("Excluded dataset IDs:", CONFIG["EXCLUDED_DATASETS"])\n', ["catalogue"]))
    cells.append(cell("markdown", "## Execute the selected range\n\nThis runs actual model fitting. The selected validation threshold is frozen before each final outer refit. Both scientific and smoke outputs are labelled, and all failures are retained in the run directory. Do not treat a partial run or smoke result as a complete study.\n"))
    cells.append(cell("code", "RUN_RESULTS = run_benchmark(CONFIG, CATALOGUE)\n", ["benchmark"]))
    cells.append(cell("code", '''import pandas as pd
from portable_ugo.benchmark import METHODS

print("Results directory:", RUN_RESULTS["run_dir"])
labels = {"baseline": "No oversampling", "original_ugo": "Original UGO",
          "u2_flow_gate": "U2 + Flow gate", "smote": "SMOTE",
          "kmeans_smote": "KMeansSMOTE", "rwo": "RWO"}
columns = {f"{method}_minority_f1_mean": labels[method] for method in METHODS}
view = RUN_RESULTS["comparison"][["dataset_number", "dataset", "classifier", "status",
                                  *columns]].rename(columns=columns)
print("Mean minority F1. INCOMPLETE rows are not full comparisons.")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(view)
print("All F1 standard deviations and seed coverage:", Path(RUN_RESULTS["run_dir"]) / "comparison.csv")
print("All metrics:", Path(RUN_RESULTS["run_dir"]) / "summary.csv")
print("All 15 matched method pairs:", Path(RUN_RESULTS["run_dir"]) / "paired_deltas.csv")
''', ["results"]))
    for index, entry in enumerate(cells):
        entry["id"] = f"ugo-{index:03d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"},
                             "ugo_bundle": {"source_sha256": provenance, "catalogue": "32 binary CLIMB entries; 30 active by default; IDs 10/26 excluded; raw OpenML"}},
                "cells": cells}
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.name}: {len(cells)} cells, {OUTPUT.stat().st_size:,} bytes")
    return OUTPUT


if __name__ == "__main__":
    build()
