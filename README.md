# MLOps Model-Promotion Decision Framework

This repository contains the reproducible experiment for a hybrid policy-driven and multi-criteria MLOps gate. The gate recommends model promotion, rejection, current-model retention, or retraining from model-quality, latency, cost, traceability, and drift evidence.

## Extended validation

The experiment includes an extended evaluation of external validity, drift-threshold calibration, learned decision baselines, and held-out monitoring outcomes.

- It adds [UCI APS Failure at Scania Trucks](https://archive.ics.uci.edu/ml/datasets/aps%2Bfailure%2Bat%2Bscania%2Btrucks), a real industrial predictive-maintenance dataset with 60,000 official training observations and 16,000 official test observations.
- It estimates one drift threshold per dataset from the 95th percentile of a bootstrap stable-null distribution. The report includes the 95 percent confidence interval and stable false-alarm rate.
- It compares the proposed gate with accuracy-only, F1-threshold, fixed-schedule, manual-policy, learned logistic-regression, and learned gradient-boosting decision strategies.
- It separates validation metrics used by the gate from outcomes measured on a held-out future test window.

The former fixed drift value `0.15` is retained only as a sensitivity-analysis reference. It is not used as the operational threshold.

## Run the experiment

```bash
python -m pip install -r requirements.txt
python mlops_decision_framework_notebook.py
```

The default output directory is `/content/mlops-decision-framework`, which is suitable for Google Colab. A local run can select another dedicated directory whose name contains `mlops`.

```bash
MLOPS_BASE_DIR=/tmp/mlops-decision-framework \
python mlops_decision_framework_notebook.py
```

The Scania files are downloaded directly from UCI and cached outside the generated archive. The source URLs and SHA-256 digests are written to the dataset metadata.

## Reproducibility controls

| Variable | Default | Purpose |
|---|---:|---|
| `MLOPS_BOOTSTRAP_REPLICATES` | `1000` | Stable-null bootstrap repetitions |
| `MLOPS_LEARNED_BASELINE_SEEDS` | `20` | Monitoring-window seeds for learned decision baselines |
| `MLOPS_MONITORING_WINDOW_SIZE` | `512` | Maximum observations per monitoring window |
| `MLOPS_INCLUDE_SCANIA_APS` | `1` | Include the real industrial case study |

Setting `MLOPS_INCLUDE_SCANIA_APS=0` is intended only for a reduced smoke test. Paper results must use the default configuration.

## Main generated reports

- `drift_calibration_table.csv` contains dataset-specific thresholds, confidence intervals, and stable false-alarm rates.
- `baseline_summary_table.csv` compares decision strategies across all scenarios.
- `baseline_summary_by_dataset_table.csv` provides the same comparison for each dataset.
- `learned_decision_summary_table.csv` reports candidate-level decision accuracy, macro F1, and ROC AUC.
- `model_metrics_table.csv` distinguishes validation metrics from held-out test metrics.

The exact full-run tables are committed under `results/validation/`. See [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) for the protocol, values, interpretation, and remaining limitations.

## Remaining scope limitation

The Scania case improves external validity but is not a live registry deployment. Drift scenarios remain controlled interventions and registry metadata remain simulated. A production study should connect the gate to MLflow Model Registry or TFX and replay time-stamped monitoring and deployment decisions.
