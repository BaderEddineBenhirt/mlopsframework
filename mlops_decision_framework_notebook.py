# %%
# ============================================================
# STRUCTURED COLAB NOTEBOOK
# HYBRID POLICY-DRIVEN MLOPS DECISION FRAMEWORK
# ============================================================
# It creates a GitHub-like experimental repository:
# - policies/*.yaml
# - outputs/*.json
# - reports/*.csv
# - README.md
# - requirements.txt
# - zip archive ready to download
#
# It evaluates:
# - multiple datasets
# - multiple drift scenarios
# - model metrics
# - decision gate
# - retraining advisor
# - baselines
# - sensitivity analysis
# - ablation study
# - decision-gate execution time
# ============================================================


# %%
# ============================================================
# 0. INSTALL AND IMPORT DEPENDENCIES
# ============================================================

import sys
import subprocess
import importlib.util

def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),
    ("joblib", "joblib"),
]:
    ensure_package(import_name, pip_name)

import os
import json
import yaml
import time
import shutil
import random
import platform
import warnings
import hashlib
import urllib.request
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)

from scipy.stats import wasserstein_distance
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)


# %%
# ============================================================
# 1. CREATE REPOSITORY-LIKE STRUCTURE
# ============================================================

BASE_DIR = Path(os.environ.get(
    "MLOPS_BASE_DIR",
    "/content/mlops-decision-framework",
))
if "mlops" not in BASE_DIR.name.lower():
    raise ValueError("MLOPS_BASE_DIR must name a dedicated MLOps output directory")
if BASE_DIR.exists():
    shutil.rmtree(BASE_DIR)

DOWNLOAD_CACHE = Path(os.environ.get(
    "MLOPS_DOWNLOAD_CACHE",
    "/content/mlops-validation-cache"
))
DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPLICATES = int(os.environ.get("MLOPS_BOOTSTRAP_REPLICATES", "1000"))
LEARNED_BASELINE_SEEDS = int(os.environ.get("MLOPS_LEARNED_BASELINE_SEEDS", "20"))
MONITORING_WINDOW_SIZE = int(os.environ.get("MLOPS_MONITORING_WINDOW_SIZE", "512"))
INCLUDE_SCANIA_APS = os.environ.get("MLOPS_INCLUDE_SCANIA_APS", "1") == "1"

for subdir in ["data", "policies", "models", "outputs", "reports", "src"]:
    (BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)

print("Created project:", BASE_DIR)


# %%
# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path

def save_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_float(x):
    return float(np.asarray(x).item()) if np.asarray(x).shape == () else float(x)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print("Downloading:", url)
        urllib.request.urlretrieve(url, destination)
    return destination


# %%
# ============================================================
# 3. CAPTURE ENVIRONMENT
# ============================================================

environment = {
    "execution_date_utc": datetime.utcnow().isoformat() + "Z",
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "pyyaml": yaml.__version__,
    "joblib": joblib.__version__,
}
save_json(environment, BASE_DIR / "outputs" / "environment.json")


# %%
# ============================================================
# 4. POLICY DESCRIPTORS
# ============================================================
# These are governance descriptors. They are not "AI".
# They externalize thresholds and decision weights so the gate is auditable.

decision_policy = {
    "policy_id": "decision-policy-v1",
    "description": "Policy for model promotion using performance, latency, cost, traceability, and drift.",
    "hard_constraints": {
        "min_accuracy": 0.85,
        "min_f1": 0.85,
        "max_mean_latency_ms": 10.0,
        "min_traceability_score": 0.80,
        "max_drift_score": None
    },
    "score_weights": {
        "f1": 0.40,
        "accuracy": 0.15,
        "latency": 0.15,
        "cost": 0.10,
        "drift_stability": 0.10,
        "traceability": 0.10
    },
    "promotion_threshold": 0.85,
    "tie_breaking": [
        "highest_decision_score",
        "lowest_mean_latency_ms",
        "smallest_model_size_mb"
    ],
    "weight_origin": "Initial policy assumption. It must be calibrated by domain expertise, sensitivity analysis, or historical deployment logs.",
    "drift_threshold_origin": "Assigned separately for each dataset from the stable validation partition."
}

monitoring_policy = {
    "policy_id": "monitoring-policy-v1",
    "description": "Monitoring and drift policy.",
    "drift_threshold": None,
    "drift_methods": {
        "numerical_features": "wasserstein_distance",
        "label_distribution": "total_variation_distance"
    },
    "supported_drift_types": [
        "stable",
        "covariate_drift",
        "label_shift",
        "sudden_drift",
        "progressive_drift",
        "recurring_drift"
    ]
}

retraining_policy = {
    "policy_id": "retraining-policy-v1",
    "description": "Retraining recommendation policy.",
    "strategy": "hybrid",
    "drift_threshold": None,
    "retraining_queue": "standard-training",
    "actions": {
        "stable": "no_retraining_required",
        "drifted": "retraining_recommended",
        "critical": "retraining_triggered"
    }
}

lifecycle_descriptor = {
    "descriptor_id": "lifecycle-descriptor-v1",
    "description": "Lifecycle descriptor for CI/CD/CT integration.",
    "stages": [
        "train_candidate_models",
        "evaluate_metrics",
        "generate_drift_report",
        "execute_decision_gate",
        "recommend_retraining",
        "export_decision_artifacts"
    ],
    "artifacts": [
        "metrics.json",
        "drift_report.json",
        "promotion_decision.json",
        "retraining_recommendation.json",
        "baseline_results.json",
        "sensitivity_results.json",
        "ablation_results.json",
        "execution_time.json",
        "drift_calibration.json",
        "learned_decision_training.json",
        "learned_decision_predictions.json",
    ]
}

save_yaml(decision_policy, BASE_DIR / "policies" / "decision_policy.yaml")
save_yaml(monitoring_policy, BASE_DIR / "policies" / "monitoring_policy.yaml")
save_yaml(retraining_policy, BASE_DIR / "policies" / "retraining_policy.yaml")
save_yaml(lifecycle_descriptor, BASE_DIR / "policies" / "lifecycle_descriptor.yaml")

decision_policy = load_yaml(BASE_DIR / "policies" / "decision_policy.yaml")
monitoring_policy = load_yaml(BASE_DIR / "policies" / "monitoring_policy.yaml")
retraining_policy = load_yaml(BASE_DIR / "policies" / "retraining_policy.yaml")


# %%
# ============================================================
# 5. DATASETS
# ============================================================

def dataset_breast_cancer():
    data = load_breast_cancer()
    return {
        "name": "wisconsin_breast_cancer",
        "domain": "small tabular binary classification",
        "source": "scikit-learn built-in dataset",
        "X": data.data.astype(float),
        "y": data.target.astype(int),
        "feature_names": list(data.feature_names),
        "class_names": [str(x) for x in data.target_names]
    }

def dataset_wine():
    data = load_wine()
    return {
        "name": "wine_multiclass",
        "domain": "tabular multiclass classification",
        "source": "scikit-learn built-in dataset",
        "X": data.data.astype(float),
        "y": data.target.astype(int),
        "feature_names": list(data.feature_names),
        "class_names": [str(x) for x in data.target_names]
    }

def dataset_synthetic_iot(random_state=42):
    X, y = make_classification(
        n_samples=2500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.65, 0.35],
        class_sep=1.2,
        flip_y=0.02,
        random_state=random_state,
    )
    return {
        "name": "synthetic_iot_sensor",
        "domain": "synthetic industrial/IoT-like binary classification",
        "source": "synthetic make_classification",
        "X": X.astype(float),
        "y": y.astype(int),
        "feature_names": [f"sensor_{i}" for i in range(X.shape[1])],
        "class_names": ["normal", "fault"]
    }

def dataset_synthetic_timeseries(random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = 3000
    n_features = 12
    t = np.arange(n_samples)
    X = np.stack([
        np.sin(t / (8 + i)) + 0.1 * rng.normal(size=n_samples)
        for i in range(n_features)
    ], axis=1)
    trend = np.linspace(0, 1, n_samples).reshape(-1, 1)
    X = X + 0.15 * trend
    risk = 0.4 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * rng.normal(size=n_samples)
    y = (risk > np.quantile(risk, 0.65)).astype(int)
    return {
        "name": "synthetic_timeseries_telemetry",
        "domain": "synthetic time-series-like telemetry classification",
        "source": "synthetic generated telemetry",
        "X": X.astype(float),
        "y": y.astype(int),
        "feature_names": [f"telemetry_{i}" for i in range(n_features)],
        "class_names": ["stable", "degraded"]
    }

SCANIA_APS_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00421/"
    "aps_failure_training_set.csv"
)
SCANIA_APS_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00421/"
    "aps_failure_test_set.csv"
)

def read_scania_aps_file(path):
    frame = pd.read_csv(path, skiprows=20, na_values="na")
    if "class" not in frame.columns:
        raise ValueError(f"Unexpected Scania APS schema in {path}")
    labels = frame.pop("class").map({"neg": 0, "pos": 1})
    if labels.isna().any():
        raise ValueError(f"Unexpected Scania APS class label in {path}")
    features = frame.apply(pd.to_numeric, errors="coerce")
    return features.to_numpy(dtype=float), labels.to_numpy(dtype=int), list(features.columns)

def dataset_scania_aps():
    train_path = download_file(
        SCANIA_APS_TRAIN_URL,
        DOWNLOAD_CACHE / "aps_failure_training_set.csv"
    )
    test_path = download_file(
        SCANIA_APS_TEST_URL,
        DOWNLOAD_CACHE / "aps_failure_test_set.csv"
    )
    X_train, y_train, feature_names = read_scania_aps_file(train_path)
    X_test, y_test, test_feature_names = read_scania_aps_file(test_path)

    if feature_names != test_feature_names:
        raise ValueError("Scania APS train and test feature schemas differ")

    usable = ~np.all(np.isnan(X_train), axis=0)
    X_train = X_train[:, usable]
    X_test = X_test[:, usable]
    feature_names = [name for name, keep in zip(feature_names, usable) if keep]

    return {
        "name": "scania_aps_failure",
        "domain": "real industrial predictive-maintenance classification",
        "source": "UCI APS Failure at Scania Trucks",
        "source_urls": [SCANIA_APS_TRAIN_URL, SCANIA_APS_TEST_URL],
        "source_sha256": {
            "training": file_sha256(train_path),
            "test": file_sha256(test_path),
        },
        "X": np.vstack([X_train, X_test]),
        "y": np.concatenate([y_train, y_test]),
        "feature_names": feature_names,
        "class_names": ["non_APS_failure", "APS_failure"],
        "predefined_split": {
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
        },
        "real_industrial": True,
        "class_imbalance": True,
        "export_full_csv": False,
    }

datasets = [
    dataset_breast_cancer(),
    dataset_wine(),
    dataset_synthetic_iot(),
    dataset_synthetic_timeseries(),
]

if INCLUDE_SCANIA_APS:
    datasets.append(dataset_scania_aps())

dataset_metadata = []
for ds in datasets:
    classes, counts = np.unique(ds["y"], return_counts=True)
    dataset_metadata.append({
        "name": ds["name"],
        "domain": ds["domain"],
        "source": ds["source"],
        "n_samples": int(ds["X"].shape[0]),
        "n_features": int(ds["X"].shape[1]),
        "class_distribution": {str(c): int(n) for c, n in zip(classes, counts)},
        "real_industrial": bool(ds.get("real_industrial", False)),
        "predefined_split": ds.get("predefined_split"),
        "source_urls": ds.get("source_urls", []),
        "source_sha256": ds.get("source_sha256", {}),
    })

save_json(dataset_metadata, BASE_DIR / "outputs" / "dataset_metadata.json")
pd.DataFrame(dataset_metadata).to_csv(BASE_DIR / "reports" / "dataset_metadata.csv", index=False)


# %%
# ============================================================
# 6. MODELS AND METRICS
# ============================================================

def build_models(seed=42, class_weight=None):
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=seed,
                class_weight=class_weight,
            ))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
                class_weight=class_weight,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]),
    }

def classification_metrics(y_true, y_pred, y_score=None):
    result = {
        "accuracy": to_float(accuracy_score(y_true, y_pred)),
        "precision": to_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": to_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": to_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": to_float(balanced_accuracy_score(y_true, y_pred)),
    }
    labels = np.unique(y_true)
    if len(labels) == 2:
        result["positive_class_recall"] = to_float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        )
        result["positive_class_f1"] = to_float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        )
        if y_score is not None:
            result["average_precision"] = to_float(
                average_precision_score(y_true, y_score)
            )
    return result

def positive_class_probability(model, X):
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(X)
    if probabilities.shape[1] != 2:
        return None
    return probabilities[:, 1]

def measure_latency_ms(model, X_sample, n_runs=30):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_sample)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return {
        "mean_latency_ms": to_float(np.mean(times)),
        "p95_latency_ms": to_float(np.percentile(times, 95))
    }

def model_size_mb(path):
    return to_float(Path(path).stat().st_size / (1024 * 1024))

def traceability_score(trace):
    fields = [
        "model_artifact_path",
        "dataset_identifier",
        "evaluation_metrics",
        "policy_identifiers",
        "decision_reasons"
    ]
    detail = {}
    score = 0.0
    for f in fields:
        ok = bool(trace.get(f))
        detail[f] = ok
        if ok:
            score += 0.20
    return round(score, 4), detail

def cost_proxy(mean_latency_ms, p95_latency_ms, size_mb):
    mean_cost = min(mean_latency_ms / 50.0, 1.0)
    p95_cost = min(p95_latency_ms / 100.0, 1.0)
    size_cost = min(size_mb / 500.0, 1.0)
    return to_float(0.40 * mean_cost + 0.30 * p95_cost + 0.30 * size_cost)


# %%
# ============================================================
# 7. DRIFT
# ============================================================

def total_variation_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    return to_float(0.5 * np.abs(p - q).sum())

def label_shift_score(y_ref, y_cur):
    labels = sorted(set(np.unique(y_ref)).union(set(np.unique(y_cur))))
    p = np.array([(y_ref == l).mean() for l in labels])
    q = np.array([(y_cur == l).mean() for l in labels])
    return total_variation_distance(p, q)

def stable_feature_scale(X_stable):
    scale = np.std(np.asarray(X_stable, dtype=float), axis=0)
    return np.where(scale > 1e-12, scale, np.nan)

def covariate_drift_score(X_ref, X_cur, feature_scale=None):
    if feature_scale is None:
        feature_scale = stable_feature_scale(X_ref)
    scores = []
    for j in range(X_ref.shape[1]):
        scale = feature_scale[j]
        if not np.isfinite(scale):
            continue
        normalized = wasserstein_distance(X_ref[:, j], X_cur[:, j]) / scale
        scores.append(min(normalized, 5.0))
    if not scores:
        return 0.0
    return to_float(np.mean(scores) / 5.0)

def sample_monitoring_window(X, y, size, seed):
    rng = np.random.default_rng(seed)
    size = min(int(size), len(y))
    if size <= 0:
        raise ValueError("Monitoring window must contain at least one sample")
    indices = rng.choice(len(y), size=size, replace=False)
    return X[indices], y[indices]

def sample_disjoint_monitoring_windows(X, y, size, seed):
    rng = np.random.default_rng(seed)
    size = min(int(size), len(y) // 2)
    if size <= 0:
        raise ValueError("Two monitoring windows require at least two samples")
    indices = rng.permutation(len(y))[: 2 * size]
    reference = indices[:size]
    current = indices[size:]
    return X[reference], y[reference], X[current], y[current]

def calibrate_drift_threshold(
    X_stable,
    y_stable,
    n_bootstrap=1000,
    quantile=0.95,
    window_size=512,
    seed=42,
    feature_scale=None,
):
    if len(y_stable) < 8:
        raise ValueError("At least eight stable observations are required for calibration")
    rng = np.random.default_rng(seed)
    window_size = min(int(window_size), max(4, len(y_stable) // 2))
    if feature_scale is None:
        feature_scale = stable_feature_scale(X_stable)
    scores = []

    for _ in range(int(n_bootstrap)):
        first = rng.choice(len(y_stable), size=window_size, replace=True)
        second = rng.choice(len(y_stable), size=window_size, replace=True)
        covariate = covariate_drift_score(
            X_stable[first],
            X_stable[second],
            feature_scale=feature_scale,
        )
        label = label_shift_score(y_stable[first], y_stable[second])
        scores.append(max(covariate, label))

    scores = np.asarray(scores, dtype=float)
    threshold = to_float(np.quantile(scores, quantile))
    ci_quantiles = []
    ci_replicates = max(200, min(1000, int(n_bootstrap)))
    for _ in range(ci_replicates):
        resampled = rng.choice(scores, size=len(scores), replace=True)
        ci_quantiles.append(np.quantile(resampled, quantile))

    return {
        "method": "bootstrap_stable_null_quantile",
        "quantile": to_float(quantile),
        "n_bootstrap": int(n_bootstrap),
        "window_size": int(window_size),
        "threshold": threshold,
        "threshold_ci95_low": to_float(np.quantile(ci_quantiles, 0.025)),
        "threshold_ci95_high": to_float(np.quantile(ci_quantiles, 0.975)),
        "stable_false_alarm_rate": to_float(np.mean(scores > threshold)),
        "stable_score_mean": to_float(np.mean(scores)),
        "stable_score_std": to_float(np.std(scores)),
        "active_drift_features": int(np.isfinite(feature_scale).sum()),
        "feature_normalization": "validation_partition_standard_deviation",
        "per_feature_normalized_distance_cap": 5.0,
    }

def generate_drift(
    X_ref,
    y_ref,
    scenario,
    seed=42,
    X_current_base=None,
    y_current_base=None,
    drift_threshold=None,
    feature_scale=None,
):
    rng = np.random.default_rng(seed)
    X_cur = X_ref.copy() if X_current_base is None else X_current_base.copy()
    y_cur = y_ref.copy() if y_current_base is None else y_current_base.copy()
    n, d = X_cur.shape
    k = min(d, max(5, int(np.ceil(0.10 * d))))
    if feature_scale is None:
        drift_scale = np.ones(k, dtype=float)
    else:
        drift_scale = np.asarray(feature_scale[:k], dtype=float)
        drift_scale = np.where(np.isfinite(drift_scale), drift_scale, 1.0)

    if scenario == "stable":
        pass

    elif scenario == "covariate_drift":
        X_cur[:, :k] += rng.normal(
            loc=1.5 * drift_scale,
            scale=0.8 * drift_scale,
            size=(n, k),
        )

    elif scenario == "label_shift":
        labels, counts = np.unique(y_cur, return_counts=True)
        if len(labels) >= 2:
            majority = labels[np.argmax(counts)]
            majority_idx = np.where(y_cur == majority)[0]
            other_idx = np.where(y_cur != majority)[0]
            selected_majority = rng.choice(
                majority_idx,
                size=max(1, int(0.8 * n)),
                replace=True,
            )
            selected_other = rng.choice(
                other_idx,
                size=max(1, n - len(selected_majority)),
                replace=True,
            )
            idx = np.concatenate([selected_majority, selected_other])
            rng.shuffle(idx)
            X_cur = X_cur[idx]
            y_cur = y_cur[idx]

    elif scenario == "sudden_drift":
        cut = n // 2
        X_cur[cut:, :k] += rng.normal(
            loc=3.0 * drift_scale,
            scale=1.0 * drift_scale,
            size=(n - cut, k),
        )

    elif scenario == "progressive_drift":
        intensity = np.linspace(0, 3.0, n).reshape(-1, 1)
        X_cur[:, :k] += intensity * rng.normal(
            loc=drift_scale,
            scale=0.3 * drift_scale,
            size=(n, k),
        )

    elif scenario == "recurring_drift":
        block = max(10, n // 4)
        for start in range(0, n, 2 * block):
            end = min(start + block, n)
            X_cur[start:end, :k] += rng.normal(
                loc=2.0 * drift_scale,
                scale=0.7 * drift_scale,
                size=(end - start, k),
            )

    else:
        raise ValueError(f"Unknown drift scenario: {scenario}")

    cov = covariate_drift_score(X_ref, X_cur, feature_scale=feature_scale)
    lab = label_shift_score(y_ref, y_cur)
    glob = max(cov, lab)

    if drift_threshold is None:
        drift_threshold = monitoring_policy["drift_threshold"]
    if drift_threshold is None:
        raise ValueError("A calibrated drift threshold is required")

    return {
        "scenario": scenario,
        "X_current": X_cur,
        "y_current": y_cur,
        "covariate_drift_score": cov,
        "label_shift_score": lab,
        "global_drift_score": glob,
        "drift_detected": bool(glob > drift_threshold),
        "drift_threshold": to_float(drift_threshold),
        "perturbed_feature_count": 0 if scenario in ["stable", "label_shift"] else int(k),
    }


# %%
# ============================================================
# 8. DECISION GATE
# ============================================================

def normalize_latency(latency, max_latency):
    return to_float(max(0.0, 1.0 - latency / max_latency))

def normalize_cost(cost):
    return to_float(max(0.0, 1.0 - cost))

def normalize_drift(drift, max_drift):
    return to_float(max(0.0, 1.0 - drift / max_drift))

def decision_score(metric, drift_score, policy):
    w = policy["score_weights"]
    hard = policy["hard_constraints"]
    return to_float(
        w["f1"] * metric["f1_score"]
        + w["accuracy"] * metric["accuracy"]
        + w["latency"] * normalize_latency(metric["mean_latency_ms"], hard["max_mean_latency_ms"])
        + w["cost"] * normalize_cost(metric["cost_proxy"])
        + w["drift_stability"] * normalize_drift(drift_score, hard["max_drift_score"])
        + w["traceability"] * metric["traceability_score"]
    )

def execute_decision_gate(candidate_metrics, drift_report, policy):
    start = time.perf_counter()
    hard = policy["hard_constraints"]
    ranked = []

    for m in candidate_metrics:
        drift = drift_report["global_drift_score"]
        score = decision_score(m, drift, policy)
        violations = []

        if m["accuracy"] < hard["min_accuracy"]:
            violations.append("accuracy_below_minimum")
        if m["f1_score"] < hard["min_f1"]:
            violations.append("f1_below_minimum")
        if m["mean_latency_ms"] > hard["max_mean_latency_ms"]:
            violations.append("latency_above_maximum")
        if m["traceability_score"] < hard["min_traceability_score"]:
            violations.append("traceability_below_minimum")
        if drift > hard["max_drift_score"]:
            violations.append("drift_above_maximum")

        if "drift_above_maximum" in violations:
            decision = "retrain"
        elif violations:
            decision = "reject"
        elif score >= policy["promotion_threshold"]:
            decision = "promote"
        else:
            decision = "keep_current"

        ranked.append({
            "dataset": m["dataset"],
            "scenario": drift_report["scenario"],
            "model_name": m["model_name"],
            "decision": decision,
            "decision_score": score,
            "violations": violations,
            "accuracy": m["accuracy"],
            "f1_score": m["f1_score"],
            "mean_latency_ms": m["mean_latency_ms"],
            "p95_latency_ms": m["p95_latency_ms"],
            "model_size_mb": m["model_size_mb"],
            "cost_proxy": m["cost_proxy"],
            "traceability_score": m["traceability_score"],
            "drift_score": drift
        })

    priority = {"promote": 3, "keep_current": 2, "retrain": 1, "reject": 0}
    ranked = sorted(
        ranked,
        key=lambda x: (
            priority[x["decision"]],
            x["decision_score"],
            -x["mean_latency_ms"],
            -x["model_size_mb"]
        ),
        reverse=True
    )

    selected = ranked[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "dataset": selected["dataset"],
        "scenario": drift_report["scenario"],
        "selected_model": selected["model_name"],
        "final_decision": selected["decision"],
        "selected_decision_score": selected["decision_score"],
        "ranking": ranked,
        "execution_time_ms": to_float(elapsed_ms),
        "policy_id": policy["policy_id"],
        "decision_reason": {
            "hard_constraints_first": True,
            "drift_has_priority_over_promotion": True,
            "tie_breaking": policy["tie_breaking"]
        }
    }

def retraining_advisor(decision, drift_report, policy):
    drift = drift_report["global_drift_score"]
    required = bool(drift > policy["drift_threshold"] or decision["final_decision"] == "retrain")
    action = policy["actions"]["drifted"] if required else policy["actions"]["stable"]
    return {
        "dataset": decision["dataset"],
        "scenario": decision["scenario"],
        "retraining_required": required,
        "action": action,
        "strategy": policy["strategy"],
        "retraining_queue": policy["retraining_queue"],
        "reason": {
            "drift_score": drift,
            "drift_threshold": policy["drift_threshold"],
            "decision_gate": decision["final_decision"]
        }
    }


# %%
# ============================================================
# 9. BASELINES, SENSITIVITY, ABLATION
# ============================================================

def baseline_accuracy_only(candidate_metrics, drift_report):
    best = max(candidate_metrics, key=lambda x: x["accuracy"])
    return {
        "strategy": "accuracy_only",
        "selected_model": best["model_name"],
        "decision": "promote",
        "drift_aware": False,
        "drift_score": drift_report["global_drift_score"]
    }

def baseline_f1_threshold(candidate_metrics, drift_report, threshold=0.85):
    eligible = [m for m in candidate_metrics if m["f1_score"] >= threshold]
    if eligible:
        best = max(eligible, key=lambda x: x["f1_score"])
        return {
            "strategy": "f1_threshold",
            "selected_model": best["model_name"],
            "decision": "promote",
            "drift_aware": False,
            "drift_score": drift_report["global_drift_score"]
        }
    return {
        "strategy": "f1_threshold",
        "selected_model": None,
        "decision": "reject",
        "drift_aware": False,
        "drift_score": drift_report["global_drift_score"]
    }

def baseline_fixed_retraining(candidate_metrics, drift_report):
    retrain = drift_report["scenario"] in ["progressive_drift", "recurring_drift"]
    best = max(candidate_metrics, key=lambda x: x["f1_score"])
    return {
        "strategy": "fixed_retraining_schedule",
        "selected_model": None if retrain else best["model_name"],
        "decision": "retrain" if retrain else "promote",
        "drift_aware": False,
        "drift_score": drift_report["global_drift_score"]
    }

def baseline_manual_policy(candidate_metrics, drift_report, drift_threshold):
    if drift_report["global_drift_score"] > drift_threshold:
        return {
            "strategy": "manual_policy",
            "selected_model": None,
            "decision": "retrain",
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"]
        }
    eligible = [m for m in candidate_metrics if m["f1_score"] >= 0.85 and m["mean_latency_ms"] <= 10.0]
    if eligible:
        best = max(eligible, key=lambda x: x["f1_score"])
        return {
            "strategy": "manual_policy",
            "selected_model": best["model_name"],
            "decision": "promote",
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"]
        }
    return {
        "strategy": "manual_policy",
        "selected_model": None,
        "decision": "reject",
        "drift_aware": True,
        "drift_score": drift_report["global_drift_score"]
    }

LEARNED_DECISION_FEATURES = [
    "accuracy",
    "f1_score",
    "latency_benefit",
    "cost_benefit",
    "traceability_score",
    "model_size_log",
    "drift_score",
    "normalized_drift",
]

def learned_decision_features(metric, drift_score, drift_threshold):
    return np.asarray([
        metric["accuracy"],
        metric["f1_score"],
        normalize_latency(metric["mean_latency_ms"], 10.0),
        normalize_cost(metric["cost_proxy"]),
        metric["traceability_score"],
        np.log1p(metric["model_size_mb"]),
        drift_score,
        drift_score / max(drift_threshold, 1e-8),
    ], dtype=float)

def evaluate_candidate_outcomes(models, candidate_metrics, X_current, y_current, policy):
    hard = policy["hard_constraints"]
    outcomes = {}
    for metric in candidate_metrics:
        model_name = metric["model_name"]
        model = models[model_name]
        prediction = model.predict(X_current)
        score = positive_class_probability(model, X_current)
        observed = classification_metrics(y_current, prediction, score)
        safe = bool(
            observed["accuracy"] >= hard["min_accuracy"]
            and observed["f1_score"] >= hard["min_f1"]
            and metric["mean_latency_ms"] <= hard["max_mean_latency_ms"]
            and metric["traceability_score"] >= hard["min_traceability_score"]
        )
        outcomes[model_name] = {
            "safe_to_promote": safe,
            "observed_accuracy": observed["accuracy"],
            "observed_f1": observed["f1_score"],
            "observed_balanced_accuracy": observed["balanced_accuracy"],
            "observed_positive_class_recall": observed.get("positive_class_recall"),
            "observed_average_precision": observed.get("average_precision"),
        }
    return outcomes

def build_learned_decision_training_rows(
    dataset_name,
    models,
    candidate_metrics,
    X_reference_pool,
    y_reference_pool,
    X_current_pool,
    y_current_pool,
    policy,
    scenarios,
    n_seeds,
    window_size,
    feature_scale,
):
    rows = []
    threshold = policy["hard_constraints"]["max_drift_score"]
    for seed in range(int(n_seeds)):
        X_reference, y_reference = sample_monitoring_window(
            X_reference_pool,
            y_reference_pool,
            window_size,
            10000 + seed,
        )
        X_current_base, y_current_base = sample_monitoring_window(
            X_current_pool,
            y_current_pool,
            window_size,
            20000 + seed,
        )
        for scenario in scenarios:
            drift = generate_drift(
                X_reference,
                y_reference,
                scenario,
                seed=30000 + seed,
                X_current_base=X_current_base,
                y_current_base=y_current_base,
                drift_threshold=threshold,
                feature_scale=feature_scale,
            )
            outcomes = evaluate_candidate_outcomes(
                models,
                candidate_metrics,
                drift["X_current"],
                drift["y_current"],
                policy,
            )
            for metric in candidate_metrics:
                model_name = metric["model_name"]
                features = learned_decision_features(
                    metric,
                    drift["global_drift_score"],
                    threshold,
                )
                rows.append({
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "scenario": scenario,
                    "model_name": model_name,
                    **dict(zip(LEARNED_DECISION_FEATURES, features)),
                    "safe_to_promote": int(outcomes[model_name]["safe_to_promote"]),
                    "observed_accuracy": outcomes[model_name]["observed_accuracy"],
                    "observed_f1": outcomes[model_name]["observed_f1"],
                })
    return rows

def train_learned_decision_models(training_rows, seed=42):
    frame = pd.DataFrame(training_rows)
    X = frame[LEARNED_DECISION_FEATURES].to_numpy(dtype=float)
    y = frame["safe_to_promote"].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        constant = int(y[0])
        return {
            "learned_logistic_regression": DummyClassifier(
                strategy="constant",
                constant=constant,
            ).fit(X, y),
            "learned_gradient_boosting": DummyClassifier(
                strategy="constant",
                constant=constant,
            ).fit(X, y),
        }

    return {
        "learned_logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]).fit(X, y),
        "learned_gradient_boosting": GradientBoostingClassifier(
            random_state=seed,
        ).fit(X, y),
    }

def safe_probability(classifier, X):
    probabilities = classifier.predict_proba(X)
    classes = np.asarray(classifier.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, int(np.where(classes == 1)[0][0])]

def learned_baseline_decisions(
    classifiers,
    candidate_metrics,
    drift_report,
    observed_outcomes,
):
    threshold = drift_report["drift_threshold"]
    X = np.vstack([
        learned_decision_features(metric, drift_report["global_drift_score"], threshold)
        for metric in candidate_metrics
    ])
    rows = []
    candidate_rows = []

    for strategy, classifier in classifiers.items():
        probabilities = safe_probability(classifier, X)
        predictions = (probabilities >= 0.5).astype(int)
        eligible = [
            index for index, prediction in enumerate(predictions)
            if prediction == 1
        ]
        if eligible:
            selected_index = max(eligible, key=lambda index: probabilities[index])
            selected_model = candidate_metrics[selected_index]["model_name"]
            decision = "promote"
        else:
            selected_model = None
            decision = "retrain" if drift_report["drift_detected"] else "reject"

        rows.append({
            "strategy": strategy,
            "selected_model": selected_model,
            "decision": decision,
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"],
        })

        for index, metric in enumerate(candidate_metrics):
            model_name = metric["model_name"]
            candidate_rows.append({
                "strategy": strategy,
                "dataset": metric["dataset"],
                "scenario": drift_report["scenario"],
                "model_name": model_name,
                "safe_probability": to_float(probabilities[index]),
                "predicted_safe": int(predictions[index]),
                "observed_safe": int(observed_outcomes[model_name]["safe_to_promote"]),
            })
    return rows, candidate_rows

def compare_baselines(
    candidate_metrics,
    drift_report,
    proposed,
    policy,
    observed_outcomes,
    learned_rows,
):
    rows = [
        baseline_accuracy_only(candidate_metrics, drift_report),
        baseline_f1_threshold(candidate_metrics, drift_report),
        baseline_fixed_retraining(candidate_metrics, drift_report),
        baseline_manual_policy(
            candidate_metrics,
            drift_report,
            policy["hard_constraints"]["max_drift_score"],
        ),
        *learned_rows,
        {
            "strategy": "proposed_decision_gate",
            "selected_model": proposed["selected_model"],
            "decision": proposed["final_decision"],
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"]
        }
    ]
    proposed_row = rows[-1]
    for r in rows:
        r["agreement_with_proposed"] = bool(
            r["decision"] == proposed_row["decision"]
            and r["selected_model"] == proposed_row["selected_model"]
        )
        r["false_promotion_under_drift"] = bool(
            r["decision"] == "promote"
            and drift_report["global_drift_score"] > policy["hard_constraints"]["max_drift_score"]
        )
        selected_model = r.get("selected_model")
        selected_outcome = observed_outcomes.get(selected_model, {})
        r["unsafe_promotion"] = bool(
            r["decision"] == "promote"
            and not selected_outcome.get("safe_to_promote", False)
        )
        r["missed_safe_promotion"] = bool(
            r["decision"] != "promote"
            and any(outcome["safe_to_promote"] for outcome in observed_outcomes.values())
        )
        r["retraining_triggered"] = bool(r["decision"] == "retrain")
    return rows

def renormalize_weights(policy):
    total = sum(policy["score_weights"].values())
    if total > 0:
        for k in policy["score_weights"]:
            policy["score_weights"][k] = policy["score_weights"][k] / total
    return policy

def run_ablation(candidate_metrics, drift_report, base_policy):
    result = {}
    variants = ["full", "no_drift", "no_latency", "no_cost", "no_traceability"]
    for variant in variants:
        p = json.loads(json.dumps(base_policy))
        if variant == "no_drift":
            p["score_weights"]["drift_stability"] = 0.0
            p["hard_constraints"]["max_drift_score"] = 999.0
        elif variant == "no_latency":
            p["score_weights"]["latency"] = 0.0
            p["hard_constraints"]["max_mean_latency_ms"] = 999.0
        elif variant == "no_cost":
            p["score_weights"]["cost"] = 0.0
        elif variant == "no_traceability":
            p["score_weights"]["traceability"] = 0.0
            p["hard_constraints"]["min_traceability_score"] = 0.0
        p = renormalize_weights(p)
        d = execute_decision_gate(candidate_metrics, drift_report, p)
        result[variant] = {
            "selected_model": d["selected_model"],
            "final_decision": d["final_decision"],
            "selected_decision_score": d["selected_decision_score"]
        }
    return result

def run_sensitivity(candidate_metrics, drift_report, base_policy):
    rows = []
    calibrated = base_policy["hard_constraints"]["max_drift_score"]
    threshold_variants = [
        (0.80 * calibrated, "calibrated_minus_20_percent"),
        (calibrated, "calibrated"),
        (1.20 * calibrated, "calibrated_plus_20_percent"),
        (0.15, "legacy_fixed_value"),
    ]

    seen_thresholds = set()
    for threshold, source in threshold_variants:
        rounded_threshold = round(float(threshold), 12)
        if rounded_threshold in seen_thresholds:
            continue
        seen_thresholds.add(rounded_threshold)
        p = json.loads(json.dumps(base_policy))
        p["hard_constraints"]["max_drift_score"] = threshold
        d = execute_decision_gate(candidate_metrics, drift_report, p)
        rows.append({
            "sensitivity_type": "drift_threshold",
            "parameter": "max_drift_score",
            "value": threshold,
            "threshold_source": source,
            "selected_model": d["selected_model"],
            "final_decision": d["final_decision"],
            "selected_decision_score": d["selected_decision_score"]
        })

    for weight in base_policy["score_weights"]:
        for factor in [0.90, 1.10]:
            p = json.loads(json.dumps(base_policy))
            p["score_weights"][weight] = p["score_weights"][weight] * factor
            p = renormalize_weights(p)
            d = execute_decision_gate(candidate_metrics, drift_report, p)
            rows.append({
                "sensitivity_type": "weight_variation",
                "parameter": weight,
                "value": factor,
                "selected_model": d["selected_model"],
                "final_decision": d["final_decision"],
                "selected_decision_score": d["selected_decision_score"]
            })
    return rows


# %%
# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================

all_metrics = []
all_drift_reports = []
all_decisions = []
all_retraining = []
all_baselines = []
all_ablation = []
all_sensitivity = []
all_execution_time = []
all_drift_calibration = []
all_learned_training = []
all_learned_candidate_predictions = []

drift_scenarios = [
    "stable",
    "covariate_drift",
    "label_shift",
    "sudden_drift",
    "progressive_drift",
    "recurring_drift"
]

for ds in datasets:
    print("\n============================================================")
    print("DATASET:", ds["name"])
    print("============================================================")

    X = ds["X"]
    y = ds["y"]

    if ds.get("predefined_split"):
        train_size = ds["predefined_split"]["train_size"]
        X_official_train = X[:train_size]
        y_official_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_official_train,
            y_official_train,
            test_size=0.20,
            stratify=y_official_train,
            random_state=42,
        )
        split_type = "official_test_with_training_validation_split"
    else:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X,
            y,
            test_size=0.40,
            stratify=y,
            random_state=42,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_tmp,
            y_tmp,
            test_size=0.50,
            stratify=y_tmp,
            random_state=42,
        )
        split_type = "stratified_60_20_20"

    class_weight = "balanced" if ds.get("class_imbalance") else None
    models = build_models(seed=42, class_weight=class_weight)
    current_metrics = []
    sample_weight = (
        compute_sample_weight(class_weight="balanced", y=y_train)
        if ds.get("class_imbalance")
        else None
    )

    for model_name, model in models.items():
        print("Training:", model_name)
        if model_name == "gradient_boosting" and sample_weight is not None:
            model.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        path = BASE_DIR / "models" / f"{ds['name']}__{model_name}.joblib"
        joblib.dump(model, path)

        y_valid_pred = model.predict(X_valid)
        y_valid_score = positive_class_probability(model, X_valid)
        perf = classification_metrics(y_valid, y_valid_pred, y_valid_score)
        y_test_pred = model.predict(X_test)
        y_test_score = positive_class_probability(model, X_test)
        heldout_perf = classification_metrics(y_test, y_test_pred, y_test_score)
        lat = measure_latency_ms(model, X_valid[:1], n_runs=30)
        size = model_size_mb(path)

        trace = {
            "model_artifact_path": str(path),
            "dataset_identifier": ds["name"],
            "evaluation_metrics": True,
            "policy_identifiers": decision_policy["policy_id"],
            "decision_reasons": True
        }
        tr_score, tr_detail = traceability_score(trace)

        c_proxy = cost_proxy(lat["mean_latency_ms"], lat["p95_latency_ms"], size)

        record = {
            "dataset": ds["name"],
            "domain": ds["domain"],
            "model_name": model_name,
            "model_artifact_path": str(path),
            "decision_metric_partition": "validation",
            "accuracy": perf["accuracy"],
            "precision": perf["precision"],
            "recall": perf["recall"],
            "f1_score": perf["f1_score"],
            "balanced_accuracy": perf["balanced_accuracy"],
            "positive_class_recall": perf.get("positive_class_recall"),
            "positive_class_f1": perf.get("positive_class_f1"),
            "average_precision": perf.get("average_precision"),
            "heldout_test_accuracy": heldout_perf["accuracy"],
            "heldout_test_f1_score": heldout_perf["f1_score"],
            "heldout_test_balanced_accuracy": heldout_perf["balanced_accuracy"],
            "heldout_test_positive_class_recall": heldout_perf.get("positive_class_recall"),
            "heldout_test_positive_class_f1": heldout_perf.get("positive_class_f1"),
            "heldout_test_average_precision": heldout_perf.get("average_precision"),
            "mean_latency_ms": lat["mean_latency_ms"],
            "p95_latency_ms": lat["p95_latency_ms"],
            "model_size_mb": size,
            "cost_proxy": c_proxy,
            "traceability_score": tr_score,
            "traceability_detail": tr_detail
        }

        current_metrics.append(record)
        all_metrics.append(record)

    drift_imputer = SimpleImputer(strategy="median")
    X_train_drift = drift_imputer.fit_transform(X_train)
    X_valid_drift = drift_imputer.transform(X_valid)
    X_test_drift = drift_imputer.transform(X_test)
    drift_feature_scale = stable_feature_scale(X_valid_drift)

    calibration = calibrate_drift_threshold(
        X_valid_drift,
        y_valid,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        quantile=0.95,
        window_size=MONITORING_WINDOW_SIZE,
        seed=42,
        feature_scale=drift_feature_scale,
    )
    calibration.update({
        "dataset": ds["name"],
        "split_type": split_type,
        "reference_partition": "validation",
    })
    all_drift_calibration.append(calibration)

    dataset_decision_policy = deepcopy(decision_policy)
    dataset_monitoring_policy = deepcopy(monitoring_policy)
    dataset_retraining_policy = deepcopy(retraining_policy)
    calibrated_threshold = calibration["threshold"]
    dataset_decision_policy["policy_id"] = f"decision-policy-v2-{ds['name']}"
    dataset_decision_policy["hard_constraints"]["max_drift_score"] = calibrated_threshold
    dataset_decision_policy["drift_threshold_origin"] = calibration["method"]
    dataset_monitoring_policy["policy_id"] = f"monitoring-policy-v2-{ds['name']}"
    dataset_monitoring_policy["drift_threshold"] = calibrated_threshold
    dataset_monitoring_policy["drift_threshold_origin"] = calibration["method"]
    dataset_retraining_policy["policy_id"] = f"retraining-policy-v2-{ds['name']}"
    dataset_retraining_policy["drift_threshold"] = calibrated_threshold
    dataset_retraining_policy["drift_threshold_origin"] = calibration["method"]

    save_yaml(
        dataset_decision_policy,
        BASE_DIR / "policies" / "calibrated" / f"{ds['name']}__decision_policy.yaml",
    )
    save_yaml(
        dataset_monitoring_policy,
        BASE_DIR / "policies" / "calibrated" / f"{ds['name']}__monitoring_policy.yaml",
    )
    save_yaml(
        dataset_retraining_policy,
        BASE_DIR / "policies" / "calibrated" / f"{ds['name']}__retraining_policy.yaml",
    )

    learned_training_rows = build_learned_decision_training_rows(
        ds["name"],
        models,
        current_metrics,
        X_train_drift,
        y_train,
        X_valid_drift,
        y_valid,
        dataset_decision_policy,
        drift_scenarios,
        LEARNED_BASELINE_SEEDS,
        MONITORING_WINDOW_SIZE,
        drift_feature_scale,
    )
    learned_classifiers = train_learned_decision_models(learned_training_rows, seed=42)
    all_learned_training.extend(learned_training_rows)

    reference_window_size = min(MONITORING_WINDOW_SIZE, len(y_test) // 2)
    X_reference, y_reference, X_current_base, y_current_base = (
        sample_disjoint_monitoring_windows(
            X_test_drift,
            y_test,
            reference_window_size,
            seed=40042,
        )
    )

    for scenario in drift_scenarios:
        drift_obj = generate_drift(
            X_reference,
            y_reference,
            scenario,
            seed=42,
            X_current_base=X_current_base,
            y_current_base=y_current_base,
            drift_threshold=calibrated_threshold,
            feature_scale=drift_feature_scale,
        )
        drift_report = {
            "dataset": ds["name"],
            "scenario": scenario,
            "covariate_drift_score": drift_obj["covariate_drift_score"],
            "label_shift_score": drift_obj["label_shift_score"],
            "global_drift_score": drift_obj["global_drift_score"],
            "drift_detected": drift_obj["drift_detected"],
            "drift_threshold": calibrated_threshold,
            "perturbed_feature_count": drift_obj["perturbed_feature_count"],
            "threshold_origin": calibration["method"],
            "method": dataset_monitoring_policy["drift_methods"],
        }
        all_drift_reports.append(drift_report)

        observed_outcomes = evaluate_candidate_outcomes(
            models,
            current_metrics,
            drift_obj["X_current"],
            drift_obj["y_current"],
            dataset_decision_policy,
        )
        learned_rows, learned_candidate_rows = learned_baseline_decisions(
            learned_classifiers,
            current_metrics,
            drift_report,
            observed_outcomes,
        )
        all_learned_candidate_predictions.extend(learned_candidate_rows)

        decision = execute_decision_gate(
            current_metrics,
            drift_report,
            dataset_decision_policy,
        )
        retraining = retraining_advisor(
            decision,
            drift_report,
            dataset_retraining_policy,
        )
        baselines = compare_baselines(
            current_metrics,
            drift_report,
            decision,
            dataset_decision_policy,
            observed_outcomes,
            learned_rows,
        )
        ablation = run_ablation(current_metrics, drift_report, dataset_decision_policy)
        sensitivity = run_sensitivity(current_metrics, drift_report, dataset_decision_policy)

        all_decisions.append(decision)
        all_retraining.append(retraining)

        for b in baselines:
            all_baselines.append({"dataset": ds["name"], "scenario": scenario, **b})

        all_ablation.append({"dataset": ds["name"], "scenario": scenario, "variants": ablation})

        for s in sensitivity:
            all_sensitivity.append({"dataset": ds["name"], "scenario": scenario, **s})

        all_execution_time.append({
            "dataset": ds["name"],
            "scenario": scenario,
            "decision_gate_execution_time_ms": decision["execution_time_ms"]
        })

# ============================================================
# SAVE DATASETS INTO data/
# ============================================================

for ds in datasets:
    X = ds["X"]
    y = ds["y"]

    if ds.get("export_full_csv", True):
        df = pd.DataFrame(X, columns=ds["feature_names"])
        df["target"] = y
        dataset_path = BASE_DIR / "data" / f"{ds['name']}.csv"
        df.to_csv(dataset_path, index=False)
    else:
        save_json({
            "dataset": ds["name"],
            "source": ds["source"],
            "source_urls": ds.get("source_urls", []),
            "source_sha256": ds.get("source_sha256", {}),
            "download_note": "Downloaded from UCI at execution time and excluded from the archive.",
        }, BASE_DIR / "data" / f"{ds['name']}__manifest.json")

print("Datasets saved in data/:")
for file in sorted((BASE_DIR / "data").glob("*.csv")):
    print("-", file)


# %%
# ============================================================
# 11. EXPORT JSON ARTIFACTS
# ============================================================

save_json(all_metrics, BASE_DIR / "outputs" / "metrics.json")
save_json(all_drift_reports, BASE_DIR / "outputs" / "drift_report.json")
save_json(all_decisions, BASE_DIR / "outputs" / "promotion_decision.json")
save_json(all_retraining, BASE_DIR / "outputs" / "retraining_recommendation.json")
save_json(all_baselines, BASE_DIR / "outputs" / "baseline_results.json")
save_json(all_sensitivity, BASE_DIR / "outputs" / "sensitivity_results.json")
save_json(all_ablation, BASE_DIR / "outputs" / "ablation_results.json")
save_json(all_execution_time, BASE_DIR / "outputs" / "execution_time.json")
save_json(all_drift_calibration, BASE_DIR / "outputs" / "drift_calibration.json")
save_json(all_learned_training, BASE_DIR / "outputs" / "learned_decision_training.json")
save_json(
    all_learned_candidate_predictions,
    BASE_DIR / "outputs" / "learned_decision_predictions.json",
)


# %%
# ============================================================
# 12. EXPORT CSV REPORTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
drift_df = pd.DataFrame(all_drift_reports)
decision_df = pd.DataFrame([
    {
        "dataset": d["dataset"],
        "scenario": d["scenario"],
        "selected_model": d["selected_model"],
        "final_decision": d["final_decision"],
        "selected_decision_score": d["selected_decision_score"],
        "execution_time_ms": d["execution_time_ms"]
    }
    for d in all_decisions
])
retraining_df = pd.DataFrame(all_retraining)
baseline_df = pd.DataFrame(all_baselines)
sensitivity_df = pd.DataFrame(all_sensitivity)
execution_time_df = pd.DataFrame(all_execution_time)
drift_calibration_df = pd.DataFrame(all_drift_calibration)
learned_training_df = pd.DataFrame(all_learned_training)
learned_candidate_df = pd.DataFrame(all_learned_candidate_predictions)

metrics_df.to_csv(BASE_DIR / "reports" / "model_metrics_table.csv", index=False)
drift_df.to_csv(BASE_DIR / "reports" / "drift_report_table.csv", index=False)
decision_df.to_csv(BASE_DIR / "reports" / "promotion_decision_table.csv", index=False)
retraining_df.to_csv(BASE_DIR / "reports" / "retraining_recommendation_table.csv", index=False)
baseline_df.to_csv(BASE_DIR / "reports" / "baseline_table.csv", index=False)
sensitivity_df.to_csv(BASE_DIR / "reports" / "sensitivity_table.csv", index=False)
execution_time_df.to_csv(BASE_DIR / "reports" / "execution_time_table.csv", index=False)
drift_calibration_df.to_csv(BASE_DIR / "reports" / "drift_calibration_table.csv", index=False)
learned_training_df.to_csv(BASE_DIR / "reports" / "learned_decision_training.csv", index=False)
learned_candidate_df.to_csv(
    BASE_DIR / "reports" / "learned_decision_predictions.csv",
    index=False,
)

baseline_summary = baseline_df.groupby("strategy").agg(
    total_cases=("strategy", "count"),
    false_promotions_under_drift=("false_promotion_under_drift", "sum"),
    unsafe_promotions=("unsafe_promotion", "sum"),
    missed_safe_promotions=("missed_safe_promotion", "sum"),
    retraining_triggers=("retraining_triggered", "sum"),
    agreement_with_proposed=("agreement_with_proposed", "mean")
).reset_index()
baseline_summary.to_csv(BASE_DIR / "reports" / "baseline_summary_table.csv", index=False)

baseline_summary_by_dataset = baseline_df.groupby(["dataset", "strategy"]).agg(
    total_cases=("strategy", "count"),
    false_promotions_under_drift=("false_promotion_under_drift", "sum"),
    unsafe_promotions=("unsafe_promotion", "sum"),
    missed_safe_promotions=("missed_safe_promotion", "sum"),
    retraining_triggers=("retraining_triggered", "sum"),
    agreement_with_proposed=("agreement_with_proposed", "mean"),
).reset_index()
baseline_summary_by_dataset.to_csv(
    BASE_DIR / "reports" / "baseline_summary_by_dataset_table.csv",
    index=False,
)

def summarize_learned_predictions(group):
    observed = group["observed_safe"].to_numpy(dtype=int)
    predicted = group["predicted_safe"].to_numpy(dtype=int)
    probability = group["safe_probability"].to_numpy(dtype=float)
    result = {
        "candidate_cases": int(len(group)),
        "decision_accuracy": to_float(accuracy_score(observed, predicted)),
        "decision_macro_f1": to_float(
            f1_score(observed, predicted, average="macro", zero_division=0)
        ),
    }
    result["decision_roc_auc"] = (
        to_float(roc_auc_score(observed, probability))
        if len(np.unique(observed)) == 2
        else np.nan
    )
    return pd.Series(result)

learned_summary_rows = [
    {
        "dataset": dataset,
        "strategy": strategy,
        **summarize_learned_predictions(group).to_dict(),
    }
    for (dataset, strategy), group in learned_candidate_df.groupby(["dataset", "strategy"])
]
learned_summary_rows.extend([
    {
        "dataset": "overall",
        "strategy": strategy,
        **summarize_learned_predictions(group).to_dict(),
    }
    for strategy, group in learned_candidate_df.groupby("strategy")
])
learned_decision_summary = pd.DataFrame(learned_summary_rows)
learned_decision_summary.to_csv(
    BASE_DIR / "reports" / "learned_decision_summary_table.csv",
    index=False,
)

execution_summary = execution_time_df.groupby("dataset").agg(
    mean_execution_time_ms=("decision_gate_execution_time_ms", "mean"),
    p95_execution_time_ms=("decision_gate_execution_time_ms", lambda x: np.percentile(x, 95))
).reset_index()
execution_summary.to_csv(BASE_DIR / "reports" / "execution_time_summary_table.csv", index=False)


# %%
# ============================================================
# 13. GENERATE README AND REQUIREMENTS
# ============================================================

readme = """# Reproducible MLOps Decision Framework Experiment

This repository contains the experimental artifacts for a hybrid policy-driven and multi-criteria MLOps decision framework.

## Objective

The framework evaluates candidate machine learning models and produces model promotion, rejection, retention, or retraining recommendations using model metrics, latency, model size, cost proxy, traceability score, drift reports, and generated YAML policy descriptors.

The proposed gate remains transparent and auditable. Learned logistic-regression and gradient-boosting decision policies are included as comparative baselines.

## Structure

```text
mlops-decision-framework/
├── data/
├── models/
├── outputs/
├── policies/
│   └── calibrated/
├── reports/
└── src/
```

## Generated Policy Files

- `policies/decision_policy.yaml`
- `policies/monitoring_policy.yaml`
- `policies/retraining_policy.yaml`
- `policies/lifecycle_descriptor.yaml`
- `policies/calibrated/<dataset>__decision_policy.yaml`
- `policies/calibrated/<dataset>__monitoring_policy.yaml`
- `policies/calibrated/<dataset>__retraining_policy.yaml`

The three top-level policy templates leave the drift threshold unset. The executable policies under `policies/calibrated/` contain the threshold estimated for each dataset.

## Generated JSON Outputs

- `outputs/environment.json`
- `outputs/dataset_metadata.json`
- `outputs/metrics.json`
- `outputs/drift_report.json`
- `outputs/promotion_decision.json`
- `outputs/retraining_recommendation.json`
- `outputs/baseline_results.json`
- `outputs/sensitivity_results.json`
- `outputs/ablation_results.json`
- `outputs/execution_time.json`
- `outputs/drift_calibration.json`
- `outputs/learned_decision_training.json`
- `outputs/learned_decision_predictions.json`

## Generated CSV Reports

- `reports/dataset_metadata.csv`
- `reports/model_metrics_table.csv`
- `reports/drift_report_table.csv`
- `reports/promotion_decision_table.csv`
- `reports/retraining_recommendation_table.csv`
- `reports/baseline_table.csv`
- `reports/baseline_summary_table.csv`
- `reports/baseline_summary_by_dataset_table.csv`
- `reports/sensitivity_table.csv`
- `reports/execution_time_table.csv`
- `reports/execution_time_summary_table.csv`
- `reports/drift_calibration_table.csv`
- `reports/learned_decision_training.csv`
- `reports/learned_decision_predictions.csv`
- `reports/learned_decision_summary_table.csv`

## Datasets

The default experiment uses five datasets:

1. Wisconsin Breast Cancer.
2. Wine multiclass dataset.
3. Synthetic industrial/IoT-like dataset.
4. Synthetic time-series-like telemetry dataset.
5. UCI APS Failure at Scania Trucks, a real industrial predictive-maintenance dataset with its official training and test partitions.

The Scania source files are downloaded directly from UCI. Their URLs and SHA-256 digests are recorded in the dataset metadata and data manifest. Missing sensor values are imputed from the training partition only. Class weighting is used because APS failures are rare.

Set `MLOPS_INCLUDE_SCANIA_APS=0` only for a reduced smoke test. Results intended for the paper must use the default value.

## Drift-Threshold Calibration

The operational drift threshold is not fixed at 0.15. For each dataset, two bootstrap monitoring windows are drawn from the stable validation partition. The threshold is the empirical 95th percentile of the resulting null drift-score distribution. The experiment records a 95 percent bootstrap confidence interval, the stable false-alarm rate, the window size, and the number of bootstrap replicates.

The historical value 0.15 appears only in sensitivity analysis, alongside the calibrated threshold and plus or minus 20 percent variants.

## Drift Scenarios

The following scenarios are evaluated:

- stable,
- covariate drift,
- label shift,
- sudden drift,
- progressive drift,
- recurring drift.

## Baselines

The proposed gate is compared against:

- accuracy-only,
- F1-threshold,
- fixed retraining schedule,
- manual policy,
- learned logistic-regression policy,
- learned gradient-boosting policy.

The learned policies are trained on validation-window examples. Their target is whether a candidate satisfies the performance and operational constraints on the realized future window. The target is not copied from the proposed gate. Final comparison uses the held-out test partition and reports unsafe promotions, missed safe promotions, candidate-level decision accuracy, macro F1, and ROC AUC when both target classes are present.

## Reproducibility

Run the Python or Colab playbook from top to bottom. The environment and package versions are stored in `outputs/environment.json`.

The main controls are `MLOPS_BOOTSTRAP_REPLICATES`, `MLOPS_LEARNED_BASELINE_SEEDS`, and `MLOPS_MONITORING_WINDOW_SIZE`. Their defaults are 1000, 20, and 512.

## Scientific Limitation

The Scania case improves external validity but does not constitute a live production deployment. Drift is still injected under controlled scenarios, and registry metadata are simulated. A future evaluation should connect the gate to MLflow Model Registry or TFX and replay time-stamped production monitoring and deployment decisions.
"""
with open(BASE_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)

requirements = f"""numpy=={np.__version__}
pandas=={pd.__version__}
scipy=={scipy.__version__}
scikit-learn=={sklearn.__version__}
PyYAML=={yaml.__version__}
joblib=={joblib.__version__}
"""
with open(BASE_DIR / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

for name in [
    "decision_gate.py",
    "drift.py",
    "drift_calibration.py",
    "baselines.py",
    "learned_baselines.py",
    "sensitivity.py",
    "ablation.py",
]:
    with open(BASE_DIR / "src" / name, "w", encoding="utf-8") as f:
        f.write("# Source logic is implemented in the Colab playbook and exported as reproducible artifacts.\n")


# %%
# ============================================================
# 14. CREATE ZIP ARCHIVE
# ============================================================

zip_base = str(BASE_DIR)
zip_path = f"{zip_base}.zip"
if Path(zip_path).exists():
    Path(zip_path).unlink()

shutil.make_archive(
    zip_base,
    "zip",
    root_dir=str(BASE_DIR.parent),
    base_dir=BASE_DIR.name,
)


# %%
# ============================================================
# 15. DISPLAY FINAL TABLES
# ============================================================

print("\n==================== DATASETS ====================")
display(pd.DataFrame(dataset_metadata))

print("\n==================== MODEL METRICS ====================")
display(metrics_df[[
    "dataset", "model_name", "accuracy", "precision", "recall", "f1_score",
    "mean_latency_ms", "p95_latency_ms", "model_size_mb", "cost_proxy", "traceability_score"
]].sort_values(["dataset", "f1_score"], ascending=[True, False]))

print("\n==================== DRIFT REPORTS ====================")
display(drift_df.sort_values(["dataset", "scenario"]))

print("\n==================== DECISIONS ====================")
display(decision_df.sort_values(["dataset", "scenario"]))

print("\n==================== BASELINE SUMMARY ====================")
display(baseline_summary)

print("\n==================== BASELINE SUMMARY BY DATASET ====================")
display(baseline_summary_by_dataset.sort_values(["dataset", "strategy"]))

print("\n==================== DRIFT CALIBRATION ====================")
display(drift_calibration_df.sort_values("dataset"))

print("\n==================== LEARNED DECISION SUMMARY ====================")
display(learned_decision_summary.sort_values("strategy"))

print("\n==================== EXECUTION TIME SUMMARY ====================")
display(execution_summary)

print("\n============================================================")
print("PLAYBOOK COMPLETED SUCCESSFULLY")
print("Project folder:", BASE_DIR)
print("ZIP archive:", zip_path)
print("To download in Colab, run:")
print("from google.colab import files")
print(f"files.download('{zip_path}')")
print("============================================================")

# Optional automatic download in Colab.
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    pass

