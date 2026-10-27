"""PyTorch MLP for climate-sensitive mortality (ROC-AUC).

Usage:
    python climate_nn.py --data-dir /path/to/csvs [--out submission_rauc_nn.csv] [--epochs 40] [--skip-cv]

Expects Train_1.csv, Test_1.csv, climate_features.csv in --data-dir.
Requires: pandas numpy scikit-learn scipy torch
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def load(data_dir):
    tr = pd.read_csv(f"{data_dir}/Train (1).csv")
    te = pd.read_csv(f"{data_dir}/Test (1).csv")
    cf = pd.read_csv(f"{data_dir}/climate_features.csv").drop(columns="deathdate")
    return tr, te, cf


def build_frame(tr, te, cf):
    male_token = tr.gender.iloc[0]
    y = tr.is_climate_sensitive.values.astype(np.float32)
    groups = tr.location.values
    al = pd.concat([tr.drop(columns="is_climate_sensitive"), te], ignore_index=True).merge(cf, on="ID", how="left")
    al["deathdate"] = pd.to_datetime(al.deathdate)
    al["t"] = (al.deathdate - pd.Timestamp("2007-01-01")).dt.days / 365.25
    al["month"] = al.deathdate.dt.month
    al["gender"] = (al.gender == male_token).astype(int)
    al["zone"] = (al.zone == "Rural").astype(int)
    n = len(tr)
    return al.iloc[:n].reset_index(drop=True), al.iloc[n:].reset_index(drop=True), y, groups


def fourier(x, lo, hi, n_freq):
    """Deterministic Fourier features: lets an MLP represent sharp, non-linear age/time effects."""
    z = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
    ks = np.arange(1, n_freq + 1)
    ang = 2 * np.pi * z[:, None] * ks[None, :] / 2.0
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def make_matrix(df, stats=None):
    """Numeric design matrix. `stats` (mean/std of continuous cols) must come from the training fold only."""
    cont = df[["latitude", "longitude"]].values.astype(np.float64)
    if stats is None:
        stats = (cont.mean(0), cont.std(0) + 1e-9)
    cont = (cont - stats[0]) / stats[1]
    age = df.age.values.astype(np.float64)
    parts = [
        cont,
        df[["gender", "zone"]].values.astype(np.float64),
        np.log1p(age)[:, None] / 3.0,
        (age[:, None] / 100.0),
        fourier(np.log1p(age), 0, np.log1p(110), 8),   # age: infant/child cliff, elderly decline
        fourier(df.t.values, 0, 16, 6),                # time drift
        fourier(df.month.values, 1, 13, 1),            # seasonality (1 harmonic)
        # age-group x time (drift differs by group)
        ((age < 1)[:, None] * df.t.values[:, None] / 16.0),
        (((age >= 1) & (age <= 5))[:, None] * df.t.values[:, None] / 16.0),
        ((age >= 60)[:, None] * df.t.values[:, None] / 16.0),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32), stats


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, d_in, hidden=64, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(drop),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_predict_one(Xtr, ytr, Xte_list, seed, epochs=40, lr=2e-3, wd=1e-2, hidden=64, drop=0.3, bs=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xtr.shape[1], hidden, drop)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(Xtr) / bs)))
    loss_fn = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(Xtr)
    Y = torch.from_numpy(ytr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        return [torch.sigmoid(model(torch.from_numpy(x))).numpy() for x in Xte_list]


def ensemble_predict(df_tr, ytr, df_te_list, n_seeds, **kw):
    Xtr, stats = make_matrix(df_tr)
    Xtes = [make_matrix(d, stats)[0] for d in df_te_list]
    outs = [[] for _ in Xtes]
    for s in range(n_seeds):
        for k, p in enumerate(fit_predict_one(Xtr, ytr, Xtes, seed=s, **kw)):
            outs[k].append(rankdata(p) / len(p))
    return [np.mean(o, axis=0) for o in outs]


# ---------------------------------------------------------------- validation
def grouped_cv(train, y, groups, n_seeds=3, reps=2, folds=5, **kw):
    scores = []
    for r in range(reps):
        oof = np.zeros(len(train))
        for a, b in StratifiedGroupKFold(folds, shuffle=True, random_state=r).split(train, y, groups):
            oof[b] = ensemble_predict(train.iloc[a], y[a], [train.iloc[b]], n_seeds, **kw)[0]
        scores.append(roc_auc_score(y, oof))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--out", default="submission_rauc_nn.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    tr, te, cf = load(args.data_dir)
    train, test, y, groups = build_frame(tr, te, cf)
    print(f"train {train.shape}, test {test.shape}, pos rate {y.mean():.3f}")

    if not args.skip_cv:
        m, s = grouped_cv(train, y, groups, epochs=args.epochs)
        print(f"MLP grouped AUC {m:.4f} +/- {s:.4f}")

    p = ensemble_predict(train, y, [test], args.seeds, epochs=args.epochs)[0]
    p = (p - p.min()) / (p.max() - p.min())
    pd.DataFrame({"ID": te.ID, "TargetRAUC": p, "TargetF1": (p >= 0.5).astype(int)}).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(p)} rows)")


if __name__ == "__main__":
    main()
