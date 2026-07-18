# Reviewer 2 Validation Report

This report records the full default run used to answer Reviewer 2. It was generated with five datasets, six monitoring scenarios, 1,000 stable-null bootstrap repetitions, 20 learned-baseline seeds, and monitoring windows of at most 512 observations.

## External-validity extension

The evaluation now includes UCI APS Failure at Scania Trucks. The official training file contains 60,000 observations and the official test file contains 16,000 observations. The combined dataset has 170 sensor variables, 74,625 negative cases, and 1,375 APS-failure cases.

The official test partition is preserved. The official training partition is divided into training and validation subsets. Missing values are imputed from the training partition only. Class weighting is applied to the three Scania candidate models.

For logistic regression on Scania, validation accuracy is 0.971750 and weighted F1 is 0.977633. Held-out test accuracy is 0.975625 and weighted F1 is 0.979144. The held-out positive-class recall is 0.906667, positive-class F1 is 0.635514, and average precision is 0.796860.

## Drift-threshold calibration

The operational threshold is the empirical 95th percentile of a stable-null bootstrap distribution estimated separately for each dataset. The 95 percent confidence interval is obtained by bootstrapping that quantile. Stable-null false-alarm rates range from 0.040 to 0.046.

| Dataset | Threshold | 95 percent CI | Null false-alarm rate | Window |
|---|---:|---:|---:|---:|
| Wisconsin Breast Cancer | 0.192982 | [0.175439, 0.192982] | 0.041 | 57 |
| Wine multiclass | 0.388889 | [0.333333, 0.388889] | 0.044 | 18 |
| Synthetic IoT sensor | 0.088000 | [0.080000, 0.092000] | 0.045 | 250 |
| Synthetic telemetry | 0.080000 | [0.073333, 0.083333] | 0.046 | 300 |
| Scania APS failure | 0.017578 | [0.016764, 0.017622] | 0.040 | 512 |

The historical threshold `0.15` is not used operationally. It remains only as a sensitivity-analysis reference.

Two of the five held-out stable scenarios exceeded their calibrated threshold. These occurred for synthetic telemetry and Scania. This result is retained rather than hidden because it measures threshold transfer from validation to unseen monitoring windows. It shows that a nominal 5 percent validation false-alarm target does not guarantee zero alarms on a small set of held-out windows.

## Decision-strategy comparison

The learned decision targets are derived from realized candidate performance on future windows. They are not copied from the proposed gate. Candidate-level evaluation is performed on held-out test windows.

| Strategy | Candidate accuracy | Macro F1 | ROC AUC |
|---|---:|---:|---:|
| Learned gradient boosting | 0.822222 | 0.796495 | 0.918598 |
| Learned logistic regression | 0.755556 | 0.745370 | 0.872809 |

The operational comparison covers 30 dataset-scenario cases.

| Strategy | False promotions under drift | Unsafe promotions | Missed safe promotions | Retraining triggers |
|---|---:|---:|---:|---:|
| Accuracy only | 18 | 24 | 0 | 0 |
| F1 threshold | 18 | 24 | 0 | 0 |
| Fixed schedule | 13 | 15 | 4 | 10 |
| Learned gradient boosting | 11 | 3 | 4 | 7 |
| Learned logistic regression | 17 | 12 | 1 | 1 |
| Manual policy | 0 | 7 | 11 | 18 |
| Proposed decision gate | 0 | 7 | 11 | 18 |

The proposed gate never promotes when the calibrated drift threshold is exceeded. It is also conservative, with 11 missed safe promotions. Seven unsafe promotions remain in scenarios whose aggregate drift score does not exceed the calibrated threshold. The learned gradient-boosting baseline achieves the best candidate-level prediction metrics and only three unsafe promotions, but it produces 11 promotions in cases where measured drift exceeds the calibrated threshold. These results support the hard safety constraint while showing the cost of that conservatism.

## Runtime

Mean gate execution time ranges from 0.020855 ms to 0.029661 ms across the five datasets. This timing covers the gate only. It excludes model training, inference, data loading, and drift-score computation.

## Reproduction evidence

The `results/reviewer2` directory contains the exact dataset metadata, model metrics, threshold calibration, drift reports, promotion decisions, baseline summaries, learned-policy summaries, execution-time summary, and environment capture from this run.

The experiment still does not constitute a live production deployment. Scania supplies real industrial observations, while the monitored shifts and registry metadata remain controlled or simulated. The next external-validity step is a time-stamped replay against an MLflow or TFX registry and production monitoring history.
