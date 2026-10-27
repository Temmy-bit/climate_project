# Climate-Sensitive Mortality Prediction

This repository contains a forecasting pipeline for predicting whether a mortality record is climate-sensitive using tabular demographic, geographic, temporal, and climate features. The project combines several machine learning models, including LightGBM, CatBoost, ExtraTrees, and a PyTorch MLP, then blends their predictions for a final submission.

The code is structured around a recurring pattern:

- load train/test data and climate metadata
- engineer time and location-aware features
- evaluate with grouped cross-validation to avoid leakage across similar locations
- train final models on the full training set
- export predictions in the required competition-style output format

---

## Project purpose

The goal is to estimate a binary target, `is_climate_sensitive`, using historical records and climate-related signals.

The project is tuned for a competition setting where:

- the training and test sets are not perfectly independent by location
- the target is sensitive to time and geography
- a robust ensemble and grouped validation strategy are more reliable than a single model

The prediction pipeline outputs two target columns:

- `TargetRAUC`: a continuous probability-like score scaled to the $[0,1]$ range
- `TargetF1`: a binary decision from a threshold at $0.5$

---

## Repository structure

```text
climate_project/
├── climate_auc.py          # baseline ROC-AUC model using LightGBM + ExtraTrees
├── climate_nn.py           # PyTorch MLP with engineered age/time features
├── climate_blend.py        # blended ensemble of tree and neural models
├── climate_external.py     # downloads external weather data and creates lagged features
├── data/
│   ├── Train (1).csv
│   ├── Test (1).csv
│   └── climate_features.csv
├── notebooks/
├── src/
│   └── climate_project/
├── pyproject.toml
├── README.md
├── submission_rauc.csv
├── submission_rauc_nn.csv
├── submission_blend.csv
└── catboost_info/
```

---

## Data requirements

The project expects the following datasets in the `data/` folder:

- `Train (1).csv`
- `Test (1).csv`
- `climate_features.csv`

The feature files include fields such as:

- `ID`
- `age`
- `gender`
- `zone`
- `latitude`
- `longitude`
- `deathdate`
- `location`
- time and climate-related columns used for modeling

In addition, the external weather pipeline can generate extra features using Open-Meteo weather data when `climate_external.py` is run.

---

## Environment setup

This project uses Python 3.12+ and includes dependencies in `pyproject.toml`.

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Or install the required libraries directly:

```bash
pip install catboost lightgbm numpy pandas scikit-learn scipy torch
```

---

## Quick start

### 1) Train the ROC-AUC baseline

```bash
python climate_auc.py --data-dir data --out submission_rauc.csv
```

This script trains a grouped cross-validation pipeline and exports predictions for the competition output.

### 2) Train the neural network baseline

```bash
python climate_nn.py --data-dir data --out submission_rauc_nn.csv
```

This applies a PyTorch MLP with Fourier-based age/time features and a grouped validation loop.

### 3) Train the blended ensemble

```bash
python climate_blend.py --data-dir data --out submission_blend.csv
```

This blends:

- LightGBM
- CatBoost
- ExtraTrees
- MLP

and produces an ensemble submission with `TargetF1` and `TargetRAUC` columns.

---

## Optional external weather features

The project also includes a weather enrichment workflow that downloads daily climate observations from Open-Meteo and converts them into lagged features summarizing recent temperature, rainfall, dry periods, and radiation conditions.

### Fetch weather data

```bash
python climate_external.py fetch --data-dir data --daily daily_weather.csv
```

### Build external features

```bash
python climate_external.py build --data-dir data --daily daily_weather.csv --out external_features.csv
```

### Use the external features in the final ensemble

```bash
python climate_blend.py --data-dir data --external external_features.csv --out submission_blend.csv
```

> This step requires internet access and is optional for the basic workflow.

---

## Modeling approach

### AUC-focused tree pipeline (`climate_auc.py`)

This script creates a strong baseline by combining:

- time features computed from `deathdate`
- encoded demographic geography fields
- climate summaries from the climate feature table
- grouped stratified cross-validation by `location`

It uses:

- `LGBMClassifier`
- `ExtraTreesClassifier`
- a rank-average prediction strategy for final output calibration

### Neural network pipeline (`climate_nn.py`)

This script builds a compact MLP that uses:

- transformed age features
- Fourier features for nonlinear time and age patterns
- seasonality features derived from month and time drift
- location-aware interactions with age bins and time

The approach is designed to capture smooth but nonlinear risk patterns that tree ensembles may not fully express.

### Blended ensemble (`climate_blend.py`)

The final model combines several distinct signal sources:

- tree-based classifiers for structured demographic + climate effects
- a neural model for nonlinear age/time interactions
- target-encoding features such as location-based infant shares and grouped effects
- optional external weather covariates

The ensemble blends probabilities and ranks using a configurable set of weights and then rescales the final rank predictions to the $[0,1]$ range.

---

## Validation strategy

A key part of the code is the use of `StratifiedGroupKFold` to evaluate models by grouped location rather than by random row split.

This matters because the dataset contains repeated or correlated observations by location, and ordinary random validation can overestimate model performance.

The scripts print grouped AUC metrics such as:

- mean AUC
- standard deviation across folds
- model-specific performance comparisons

---

## Output format

The submission files are written as CSVs with at least the following columns:

```csv
ID,TargetF1,TargetRAUC
```

Where:

- `ID` matches the test rows
- `TargetF1` is a binary thresholded prediction
- `TargetRAUC` is a calibrated probability-like score

The exact output names are:

- `submission_rauc.csv`
- `submission_rauc_nn.csv`
- `submission_blend.csv`

---

## Notes and practical tips

- Run the scripts from the project root so the relative data paths resolve correctly.
- If you are using the external weather pipeline, ensure the environment has internet access.
- For best results, compare the baseline models first before relying on the ensemble.
- The system is designed for tabular competition data and is not a general-purpose production service.

---

## Typical workflow

```bash
# basic modeling
python climate_auc.py --data-dir data --out submission_rauc.csv
python climate_nn.py --data-dir data --out submission_rauc_nn.csv
python climate_blend.py --data-dir data --out submission_blend.csv

# optional external weather enhancement
python climate_external.py fetch --data-dir data --daily daily_weather.csv
python climate_external.py build --data-dir data --daily daily_weather.csv --out external_features.csv
python climate_blend.py --data-dir data --external external_features.csv --out submission_blend.csv
```

---

## License and project status

This repository is a research and competition-style machine learning project. It is intended for experimentation, model benchmarking, and submission generation. The scripts are designed to be easy to run locally or in a notebook environment.

If you want to improve the project further, the most promising next steps are:

1. compare feature ablations from the core and weather models
2. tune the ensemble weights using grouped validation
3. quantify the gain of external climate features over the base models
4. add a reproducible training script and experiment tracking

---

## Summary

This repository demonstrates a complete ensemble-based forecasting workflow for climate-sensitive mortality classification. It combines classical gradient boosting, an MLP, target encoding, and optional external climate signals to build a robust predictive system.
