# Precursor-to-Surge Price Classification

A machine learning pipeline that mines Coinbase limit-order-book history for short, quiet/negative price stretches (**precursors**) that are immediately followed by a continuous upward move (a **surge**), then trains classifiers to predict — from the precursor's own order-book statistics alone — whether the surge will clear a fixed price target.

> ### ⚠️ PROTOTYPE / PROOF-OF-CONCEPT — NOT VALIDATED FOR LIVE TRADING
> This project demonstrates a feature-engineering and modeling approach. It has **not** been evaluated with a leakage-free, chronological protocol and **must not** be used to size trades, estimate returns, or inform any capital-at-risk decision. See [Known issues](#known-issues-do-not-ignore) and [Before any live use](#before-any-live-use) below.

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Files in this repo](#files-in-this-repo)
- [Top-line results](#top-line-results-held-out-split)
- [Known issues (do not ignore)](#known-issues-do-not-ignore)
- [Before any live use](#before-any-live-use)
- [Full report](#full-report)

## What this project does

Over a 13-month window, 303,127 fifteen-second Coinbase order-book snapshots are mined into 5,730 precursor→surge event pairs. Each pair is scored by how close the precursor's own price peak came to the subsequent surge's peak, then reduced to a binary label: did the surge clear a fixed target (`surge_targets_met_pct > 0.74`)? Only 103 of 5,730 pairs (1.8%) are positive — a severe class imbalance that shapes every modeling decision.

The goal is a reusable early-warning signal, not a finished trading model.

## Pipeline at a glance

1. **Ingest** — 303,127 order-book snapshots (15-second polling) combined from 300+ daily CSVs covering Aug 11, 2022 – Sep 3, 2023.
2. **Mine sequences** — 10-row rolling momentum windows compared against a dataset-wide mean-change threshold split the tape into alternating precursor/surge episodes.
3. **Engineer features** — for each episode: `length` (duration), `sum_change` (cumulative price change), and buy/ask capitalization & volume momentum during the precursor.
4. **Pair & label** — each precursor is paired with its subsequent surge; the pair is labeled `1` if the surge clears the precursor's prior peak by more than 0.74%, else `0`.
5. **Model** — ADASYN oversampling (after comparing several `imbalanced-learn` resamplers) feeding 7 classifiers: Logistic Regression, Bernoulli Naive Bayes, K-Nearest Neighbors, Balanced Bagging, Balanced Random Forest, RUSBoost, and a soft-voting ensemble.

## Files in this repo

| File | Purpose |
|---|---|
| `1_data_getter_bin_pipeline.ipynb` | Ingests raw limit-order-book data, mines precursor/surge episodes, engineers features, writes `binned_pipeline.csv` / `binary_binned_pipeline.csv`. |
| `2_classifiers_bin_pipeline_data.ipynb` | Loads the binary-labeled CSV, resamples, trains & compares 7 classifiers, produces SHAP and learning-curve diagnostics. |
| `ML_Project_Summary_Report.pdf` | Full project report: data, preprocessing, EDA, modeling, results, limitations & recommendations. |
| `README.md` | This document. |

## Top-line results (held-out split)

| Model | Score | Type |
|---|---|---|
| K-Nearest Neighbors (k=3) | 93.1% accuracy | Best single model |
| Balanced Random Forest | 83.6% balanced accuracy | Best imbalance-aware ensemble |
| Voting Ensemble | 82.0% accuracy | Most balanced precision/recall |
| RUSBoost | 70.9% balanced accuracy | Imbalance-aware ensemble |
| Balanced Bagging | 68.2% balanced accuracy | Imbalance-aware ensemble |
| Logistic Regression | 68.1% accuracy | Linear baseline |
| Bernoulli Naive Bayes | 64.9% accuracy | Linear baseline |

Strongest predictors: precursor **length** and **sum_change** (~44% combined feature importance), followed by order-book capital/volume momentum. See the full report for the corrected feature-importance chart and all confusion matrices.

## Known issues (do not ignore)

- **Oversampling before the split.** ADASYN is applied to the full dataset *before* `train_test_split` — synthetic minority points can end up on both sides of the split, inflating every accuracy figure above.
- **Timestamp used as a raw feature under a random split.** The feature set includes the raw epoch-millisecond `time` value, and the split is random rather than chronological. It ranks as the 3rd-most-important feature in a re-derived check — most likely calendar memorization, not genuine signal.
- **Test-set scaler fit independently of train.** `StandardScaler().fit_transform()` is called separately on train and test instead of fitting once on train and transforming test.
- **Cross-validation run on the test set.** The reported 10-fold CV scores for the ensemble models are computed on the held-out test split, not on training folds.
- **A very small, synthetic-heavy minority class.** Only 103 real positive examples exist in 13 months of data; most minority rows after ADASYN are synthetic interpolations.
- **SHAP feature-name mismatch (cosmetic).** The notebook's SHAP summary plot is generated with a mismatched number of feature labels; corrected in the full report.

## Before any live use

Re-run the modeling notebook with:
- train/test split performed **before** any resampling,
- a chronological (walk-forward) split instead of a random one,
- the timestamp feature removed or re-encoded (e.g. cyclical hour-of-day/day-of-week),
- the scaler fit once on train and applied to test,
- cross-validation confined to training folds only.

Full detail and rationale are in Sections 6–7 of `ML_Project_Summary_Report.pdf`.

## Full report

For complete methodology, every chart, all confusion matrices, learning curves, and the full recommendations list, see **[`ML_Project_Summary_Report.pdf`](./ML_Project_Summary_Report.pdf)**.

---
