"""Final blend: LightGBM + CatBoost + ExtraTrees + PyTorch MLP.

Usage:
    python climate_blend.py --data-dir /path/to/csvs [--out submission_blend.csv] [--skip-cv]

Needs climate_auc.py and climate_nn.py in the same folder, plus: catboost lightgbm torch scikit-learn scipy pandas numpy.
Outputs ID, TargetF1 (weighted prob >= 0.5), TargetRAUC (weighted rank blend in [0,1]).
"""
import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import climate_auc as T
import climate_nn as N

warnings.filterwarnings("ignore")
SPW = float(os.environ.get("SPW", 1.4))  # class weight: shifts the effective 0.5 cut-off toward F1-optimal
LOGIT_SHIFT = np.log(SPW)
EXTRA = ["locyr_infant_share"]  # label-free: share of infants among deaths in same location-year (train+test)
TE_SETS = {
    "ageb_yr": ["ageb", "yr"], "ageb_mon": ["ageb", "month"], "ageb_zone": ["ageb", "zone"],
    "ageb_gender": ["ageb", "gender"], "ageb_yr2_zone": ["ageb", "yr2", "zone"], "cell": ["cell"],
}
TE_NAMES = ["te_" + k for k in TE_SETS]
MODELS = ["LGB", "CB", "ET", "MLP"]


# ---------------------------------------------------------------- features
def prepare(data_dir):
    tr, te, cf = T.load(data_dir)
    train, test, y, groups = T.build_features(tr, te, cf)
    al = pd.concat([train, test], ignore_index=True)
    al["yr"] = al.deathdate.dt.year
    al["yr2"] = al.yr // 2
    al["infant"] = (al.age < 1).astype(int)
    al["locyr_infant_share"] = al.groupby(["location", "yr"]).infant.transform("mean")
    al["ageb"] = np.select([al.age < 1, al.age <= 5, al.age < 15, al.age < 60], [0, 1, 2, 3], 4)
    al["cell"] = al.latitude.round(2).astype(str) + "_" + al.longitude.round(2).astype(str)
    n = len(train)
    return al.iloc[:n].reset_index(drop=True), al.iloc[n:].reset_index(drop=True), y, groups, te.ID.values


def _te_one(dfit, yfit, dap, cols, m=20):
    key = (lambda d: d[cols].astype(str).agg("|".join, axis=1)) if len(cols) > 1 else (lambda d: d[cols[0]].astype(str))
    s = pd.DataFrame({"k": key(dfit).values, "y": yfit}).groupby("k").y.agg(["sum", "count"])
    k = key(dap).values
    S = np.nan_to_num(s["sum"].reindex(k).values)
    C = np.nan_to_num(s["count"].reindex(k).values)
    return (S + m * yfit.mean()) / (C + m)


def add_te(dtr, ytr, gtr, dap):
    """Target-mean encodings. Train rows: inner location-grouped OOF; apply rows: fit on all train rows."""
    Etr = np.zeros((len(dtr), len(TE_SETS)))
    Eap = np.zeros((len(dap), len(TE_SETS)))
    for j, cols in enumerate(TE_SETS.values()):
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(dtr, ytr, gtr):
            Etr[b, j] = _te_one(dtr.iloc[a], ytr[a], dtr.iloc[b], cols)
        Eap[:, j] = _te_one(dtr, ytr, dap, cols)
    return (pd.concat([dtr.reset_index(drop=True), pd.DataFrame(Etr, columns=TE_NAMES)], axis=1),
            pd.concat([dap.reset_index(drop=True), pd.DataFrame(Eap, columns=TE_NAMES)], axis=1))


# ---------------------------------------------------------------- models (each returns P(y=1) for the apply set)
TREE_COLS = T.CORE + EXTRA + TE_NAMES
ET_COLS = T.CORE + EXTRA

_orig_matrix = N.make_matrix


def _matrix_with_extra(df, stats=None):
    X, st = _orig_matrix(df, stats)
    return np.concatenate([X, ((df.locyr_infant_share.values - 0.35) / 0.3)[:, None].astype(np.float32)], axis=1), st


N.make_matrix = _matrix_with_extra


def fit_predict_all(dtr, ytr, dap, seeds_tree=3, seeds_nn=3, epochs=30):
    import lightgbm as lgb
    out = {}
    out["LGB"] = np.mean([lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5, scale_pos_weight=SPW, random_state=s, verbose=-1)
        .fit(dtr[TREE_COLS], ytr).predict_proba(dap[TREE_COLS])[:, 1] for s in range(seeds_tree)], axis=0)
    out["CB"] = np.mean([CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=5, l2_leaf_reg=8, scale_pos_weight=SPW, random_seed=s, verbose=0)
        .fit(dtr[TREE_COLS], ytr).predict_proba(dap[TREE_COLS])[:, 1] for s in range(seeds_tree)], axis=0)
    et = np.mean([ExtraTreesClassifier(500, min_samples_leaf=10, max_features=0.7, n_jobs=-1, random_state=s)
                  .fit(dtr[ET_COLS], ytr).predict_proba(dap[ET_COLS])[:, 1] for s in range(seeds_tree)], axis=0)
    out["ET"] = 1 / (1 + np.exp(-(np.log(np.clip(et, 1e-4, 1 - 1e-4) / (1 - np.clip(et, 1e-4, 1 - 1e-4))) + LOGIT_SHIFT)))
    Xtr, st = N.make_matrix(dtr)
    Xap = N.make_matrix(dap, st)[0]
    nn_p = np.mean([N.fit_predict_one(Xtr, ytr.astype(np.float32), [Xap], seed=s, epochs=epochs)[0] for s in range(seeds_nn)], axis=0)
    out["MLP"] = 1 / (1 + np.exp(-(np.log(np.clip(nn_p, 1e-4, 1 - 1e-4) / (1 - np.clip(nn_p, 1e-4, 1 - 1e-4))) + LOGIT_SHIFT)))
    return out


# ---------------------------------------------------------------- blending
def blend(P, w):
    prob = sum(w[k] * P[k] for k in MODELS)
    rank = sum(w[k] * rankdata(P[k]) / len(P[k]) for k in MODELS)
    return prob, rank


def weight_grid(step=0.1):
    pts = np.arange(0, 1 + 1e-9, step)
    for c in itertools.product(pts, repeat=len(MODELS) - 1):
        if sum(c) <= 1 + 1e-9:
            yield dict(zip(MODELS, list(c) + [1 - sum(c)]))


def run_cv(train, y, groups, reps=2):
    oofs = []
    for r in range(reps):
        oof = {k: np.zeros(len(train)) for k in MODELS}
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(train, y, groups):
            dtr, dva = add_te(train.iloc[a], y[a], groups[a], train.iloc[b])
            for k, v in fit_predict_all(dtr, y[a], dva).items():
                oof[k][b] = v
        oofs.append(oof)
        print(f"rep {r} done", flush=True)
    return oofs


def score(oofs, y, w):
    s = []
    for P in oofs:
        prob, rank = blend(P, w)
        auc, f1 = roc_auc_score(y, rank), f1_score(y, prob >= 0.5)
        s.append((auc, f1, 0.6 * f1 + 0.4 * auc))
    return np.mean(s, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--out", default="submission_blend.csv")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--external", default=None,
                    help="external_features.csv from climate_external.py (added to LightGBM/CatBoost only)")
    args = ap.parse_args()
    torch.set_num_threads(4)

    train, test, y, groups, test_ids = prepare(args.data_dir)
    if args.external:
        global TREE_COLS
        ext = pd.read_csv(args.external)
        ext_cols = [c for c in ext.columns if c != "ID"]
        n_tr = len(train)
        train = train.merge(ext, on="ID", how="left")
        test = test.merge(ext, on="ID", how="left")
        assert len(train) == n_tr, "external merge changed train row count"
        TREE_COLS = T.CORE + EXTRA + TE_NAMES + ext_cols
        print(f"external features: {len(ext_cols)} columns, train NaN rows {train[ext_cols].isna().any(axis=1).mean():.1%}")
    weights = {"LGB": 0.3, "CB": 0.2, "ET": 0.2, "MLP": 0.3}
    if not args.skip_cv:
        oofs = run_cv(train, y, groups)
        print("--- single models (grouped OOF): AUC / F1@0.5 / combined")
        for k in MODELS:
            print(f"{k:4s}", np.round(score(oofs, y, {m: float(m == k) for m in MODELS}), 4))
        print("equal blend  ", np.round(score(oofs, y, {m: 0.25 for m in MODELS}), 4))
        best = max(weight_grid(), key=lambda w: score(oofs, y, w)[2])
        print("best weights ", {k: round(v, 2) for k, v in best.items()}, np.round(score(oofs, y, best), 4))
        # shrink the searched weights toward equal to limit overfitting to OOF
        weights = {k: 0.5 * best[k] + 0.5 * 0.25 for k in MODELS}
        print("final weights (50% shrink to equal)", {k: round(v, 3) for k, v in weights.items()},
              np.round(score(oofs, y, weights), 4))
        print("OOF predicted positive rate", round(float(np.mean([blend(P, weights)[0] >= 0.5 for P in oofs])), 3),
              "| true", round(float(y.mean()), 3), "| SPW", SPW)

    dtr, dte = add_te(train, y, groups, test)
    P = fit_predict_all(dtr, y, dte, seeds_tree=5, seeds_nn=5)
    prob, rank = blend(P, weights)
    rauc = (rank - rank.min()) / (rank.max() - rank.min())
    sub = pd.DataFrame({"ID": test_ids, "TargetF1": (prob >= 0.5).astype(int), "TargetRAUC": rauc})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows), predicted positive rate {sub.TargetF1.mean():.3f}")


if __name__ == "__main__":
    main()
