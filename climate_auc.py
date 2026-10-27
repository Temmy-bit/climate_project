"""Climate-sensitive mortality: ROC-AUC pipeline.

Usage:
    python climate_auc.py --data-dir /path/to/csvs [--out submission_rauc.csv] [--skip-cv]

Expects Train_1.csv, Test_1.csv, climate_features.csv in --data-dir.
Requires: pandas numpy scikit-learn lightgbm scipy
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

CORE = ["age", "t", "gender", "zone", "latitude", "longitude", "month"]
CORE_PLUS_WEATHER = CORE + ["avg_temperature", "max_temperature", "min_temperature", "precipitation"]


# ---------------------------------------------------------------- data
def load(data_dir):
    tr = pd.read_csv(f"{data_dir}/Train (1).csv")
    te = pd.read_csv(f"{data_dir}/Test (1).csv")
    cf = pd.read_csv(f"{data_dir}/climate_features.csv").drop(columns="deathdate")
    return tr, te, cf


def build_features(tr, te, cf):
    """Stack train+test so encodings are identical; return (train_df, test_df, y, groups)."""
    male_token = tr.gender.iloc[0]  # consistent binary encoding across both sets
    y = tr.is_climate_sensitive.values
    groups = tr.location.values
    al = pd.concat([tr.drop(columns="is_climate_sensitive"), te], ignore_index=True)
    al = al.merge(cf, on="ID", how="left")
    al["deathdate"] = pd.to_datetime(al.deathdate)
    al["month"] = al.deathdate.dt.month
    al["t"] = (al.deathdate - pd.Timestamp("2007-01-01")).dt.days / 365.25  # continuous time -> captures label drift
    al["gender"] = (al.gender == male_token).astype(int)
    al["zone"] = (al.zone == "Rural").astype(int)
    n = len(tr)
    return al.iloc[:n].reset_index(drop=True), al.iloc[n:].reset_index(drop=True), y, groups


# ---------------------------------------------------------------- models
def make_lgb(seed=0, n_estimators=150):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.02, num_leaves=8, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
        random_state=seed, verbose=-1)


def make_et(seed=0):
    return ExtraTreesClassifier(500, min_samples_leaf=10, max_features=0.7, n_jobs=-1, random_state=seed)


# ---------------------------------------------------------------- validation
def grouped_cv(train, y, groups, cols, make_model, reps=3, folds=5):
    """Location-grouped CV (test locations are ~97% unseen). Returns mean, std of OOF AUC over reps."""
    scores = []
    for r in range(reps):
        oof = np.zeros(len(train))
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=r)
        for a, b in splitter.split(train, y, groups):
            m = make_model()
            m.fit(train.loc[a, cols], y[a])
            oof[b] = m.predict_proba(train.loc[b, cols])[:, 1]
        scores.append(roc_auc_score(y, oof))
    return float(np.mean(scores)), float(np.std(scores))


def run_cv(train, y, groups):
    experiments = {
        "age only": ["age"],
        "age + t": ["age", "t"],
        "core (LGB)": CORE,
        "core + weather (LGB)": CORE_PLUS_WEATHER,
    }
    for name, cols in experiments.items():
        m, s = grouped_cv(train, y, groups, cols, make_lgb)
        print(f"{name:24s} grouped AUC {m:.4f} +/- {s:.4f}")
    m, s = grouped_cv(train, y, groups, CORE, make_et)
    print(f"{'core (ExtraTrees)':24s} grouped AUC {m:.4f} +/- {s:.4f}")


# ---------------------------------------------------------------- final fit
def fit_predict(train, test, y, seeds=5):
    """Rank-average of LightGBM + ExtraTrees over several seeds, scaled to [0, 1]."""
    preds = []
    for s in range(seeds):
        for make in (lambda s=s: make_lgb(s), lambda s=s: make_et(s)):
            m = make()
            m.fit(train[CORE], y)
            preds.append(rankdata(m.predict_proba(test[CORE])[:, 1]) / len(test))
    p = np.mean(preds, axis=0)
    return (p - p.min()) / (p.max() - p.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--out", default="submission_rauc.csv")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    tr, te, cf = load(args.data_dir)
    train, test, y, groups = build_features(tr, te, cf)
    print(f"train {train.shape}, test {test.shape}, positive rate {y.mean():.3f}, "
          f"unseen test locations {(~te.location.isin(tr.location)).mean():.1%}")

    if not args.skip_cv:
        run_cv(train, y, groups)

    sub = pd.DataFrame({"ID": te.ID, "TargetRAUC": fit_predict(train, test, y), "TargetF1": (fit_predict(train, test, y)>= 0.5).astype(int)})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
