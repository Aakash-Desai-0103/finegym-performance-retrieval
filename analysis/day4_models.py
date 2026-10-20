"""Day 4: stronger score models on the frozen VideoMAE features.

Compares, on the same split and metrics as Day 3:
  ridge     linear regression (Day 3 reference)
  mlp       small neural network (ensemble of 5 seeds, early stopping)
  pairwise  ranking model trained only on PAIRS OF CLIPS FROM THE SAME VAULT ("which one has the
            higher E?"), i.e. learns exactly the same-action comparison the project is about

Targets: E (main) and Total (for the FineGrade comparison).
Metrics: SRCC overall, SRCC within the same vault (codes with >=5 test clips), 95% bootstrap CIs
over test VIDEOS.

Usage:
    python analysis/day4_models.py --feats features/videomae_k400_all.npz \
        --index features/videomae_k400_index.csv --out analysis
Writes analysis/day4_results.csv and prints the table. Runs on a laptop CPU in a few minutes.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="videomae_k400_all.npz")
ap.add_argument("--index", default="videomae_k400_index.csv")
ap.add_argument("--out", default=".")
ap.add_argument("--boot", type=int, default=500)
args = ap.parse_args()

F = np.load(args.feats, allow_pickle=True)
m = pd.read_csv(args.index, dtype={"vault_code": str})
assert (F["ids"] == m.clip_id.values).all(), "features and index are not aligned"
W = F["whole"].astype(np.float32)
S = F["stages"].astype(np.float32)
hs = F["has_stages"].astype(bool)
code = m.vault_code.fillna("NA").values
vid = m.video_id.values
tr = (m.split == "train").values
te = (m.split == "test").values
rng = np.random.default_rng(0)

stage_cat = np.where(hs[:, None], S.reshape(len(S), -1), np.tile(W, 3))  # no stages -> whole x3
FEATS = {
    "landing": np.where(hs[:, None], S[:, 2], W),
    "stages (3)": stage_cat,
    "whole+stages": np.hstack([W, stage_cat]),
}


def within_srcc(p, y, c):
    rs, ws = [], []
    for v in np.unique(c):
        k = c == v
        if k.sum() >= 5 and np.std(y[k]) > 0 and np.std(p[k]) > 0:
            rs.append(spearmanr(p[k], y[k])[0])
            ws.append(k.sum())
    return np.average(rs, weights=ws) if rs else np.nan


def evaluate(p, y):
    yt, ct, vt = y[te], code[te], vid[te]
    s, w = spearmanr(p, yt)[0], within_srcc(p, yt, ct)
    tv = np.unique(vt)
    bs, bw = [], []
    for _ in range(args.boot):
        pick = rng.choice(tv, len(tv))
        idx = np.concatenate([np.where(vt == v)[0] for v in pick])
        bs.append(spearmanr(p[idx], yt[idx])[0])
        bw.append(within_srcc(p[idx], yt[idx], ct[idx]))
    return s, np.nanpercentile(bs, [2.5, 97.5]), w, np.nanpercentile(bw, [2.5, 97.5])


def prep(X, n_pca=None):
    sc = StandardScaler().fit(X[tr])
    Z = sc.transform(X)
    if n_pca:
        Z = PCA(n_pca, random_state=0).fit(Z[tr]).transform(Z)
        Z = StandardScaler().fit(Z[tr]).transform(Z)
    return Z


def ridge(X, y):
    Z = prep(X)
    r = RidgeCV(alphas=np.logspace(0, 5, 21)).fit(Z[tr], y[tr])
    return r.predict(Z[te])


def mlp(X, y):
    # Settings chosen by 5-fold cross-validation over TRAIN videos (see notes): standardised target,
    # strong weight decay, no early stopping (a random clip-level validation split leaks broadcast
    # identity between train and validation and made the network fail on new competitions).
    Z = prep(X, n_pca=128)
    mu, sd = y[tr].mean(), y[tr].std()
    preds = []
    for seed in range(5):
        net = MLPRegressor(hidden_layer_sizes=(64,), alpha=10.0, max_iter=300, random_state=seed)
        net.fit(Z[tr], (y[tr] - mu) / sd)
        preds.append(net.predict(Z[te]) * sd + mu)
    return np.mean(preds, 0)


def pairwise(X, y, min_gap=0.1):
    """Linear ranker from same-vault train pairs: P(i beats j) = sigmoid(w . (x_i - x_j))."""
    Z = prep(X, n_pca=256)
    idx = np.where(tr)[0]
    A, B = [], []
    for c in np.unique(code[idx]):
        k = idx[code[idx] == c]
        for a in range(len(k)):
            for b in range(a + 1, len(k)):
                i, j = k[a], k[b]
                if vid[i] != vid[j] and abs(y[i] - y[j]) >= min_gap:
                    A.append(i); B.append(j)
    A, B = np.array(A), np.array(B)
    D = Z[A] - Z[B]
    lab = (y[A] > y[B]).astype(int)
    D = np.vstack([D, -D]); lab = np.concatenate([lab, 1 - lab])  # symmetric
    clf = LogisticRegression(C=0.01, fit_intercept=False, max_iter=5000).fit(D, lab)
    pairwise.n_pairs = len(A)
    return Z[te] @ clf.coef_[0]


rows = []
for target in ["E", "Total"]:
    y = m[target].values.astype(float)
    for fname, X in FEATS.items():
        for mname, fn in [("ridge", ridge), ("mlp", mlp), ("pairwise", pairwise)]:
            if target == "Total" and mname == "pairwise":
                continue  # within-vault Total ranking == E ranking (D is constant); skip
            p = fn(X, y)
            s, sci, w, wci = evaluate(p, y)
            rows.append([target, fname, mname, s, *sci, w, *wci])
            print(f"{target:5s} {fname:13s} {mname:9s} SRCC {s:.3f} [{sci[0]:.2f},{sci[1]:.2f}]  "
                  f"within-vault {w:.3f} [{wci[0]:.2f},{wci[1]:.2f}]", flush=True)
print(f"pairwise model trained on {pairwise.n_pairs} same-vault train pairs (|dE| >= 0.1)")

R = pd.DataFrame(rows, columns=["target", "features", "model", "SRCC", "SRCC_lo", "SRCC_hi",
                                "within_vault_SRCC", "wv_lo", "wv_hi"])
R.round(3).to_csv(os.path.join(args.out, "day4_results.csv"), index=False)
print("\nwritten", os.path.join(args.out, "day4_results.csv"))
