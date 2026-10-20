"""Day 4 follow-up: is the within-vault E ranking real execution signal, or a competition-level shortcut?

Hypothesis: regression models learn the absolute E level of each competition (elite meets score
higher, and the features recognise the broadcast very well). Within one vault code, test clips come
from several competitions, so a model could rank them partly by WHICH COMPETITION they are from.

This script evaluates E predictions three ways:
  overall               Spearman over all test clips
  within vault          Spearman inside each vault code (as before), across competitions
  within vault + video  Spearman inside each (vault code, source video) group: same vault, same
                        competition, so the competition level is held constant and only
                        execution differences remain

plus a diagnostic model:
  <model> video-avg     each clip's prediction replaced by the mean prediction of its video, i.e. a
                        model that only knows "which competition". Its within-vault score shows how
                        much ranking is possible from competition level alone.

Usage:
    python analysis/day4_within_video.py --feats features/videomae_k400_all.npz \
        --index features/videomae_k400_index.csv --out analysis
    # optional: pairwise trained only on pairs from different videos
    python analysis/day4_within_video.py ... --pairs cross

What to look for:
  * Does ridge's 'within vault + video' drop far below its 'within vault'?  -> shortcut
  * Does pairwise hold up better than ridge on 'within vault + video'?       -> pairwise more honest
  * How high is 'video-avg' on 'within vault'?                               -> size of the shortcut
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
ap.add_argument("--pairs", choices=["all", "cross"], default="all",
                help="pairwise training pairs: all same-vault pairs, or only pairs from different videos")
ap.add_argument("--min-group", type=int, default=3, help="min clips per (vault, video) group")
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
y = m.E.values.astype(float)
rng = np.random.default_rng(0)

stage_cat = np.where(hs[:, None], S.reshape(len(S), -1), np.tile(W, 3))
FEATS = {"landing": np.where(hs[:, None], S[:, 2], W), "stages (3)": stage_cat}

# ---- grouped Spearman ------------------------------------------------------------------------
def grouped_srcc(p, yy, groups, min_n):
    rs, ws = [], []
    for g in np.unique(groups):
        k = groups == g
        # unique-value checks, not np.std: averaged predictions that are "equal" can differ by
        # ~1e-16 after floating-point rounding, which makes spearmanr return nan
        if k.sum() >= min_n and len(np.unique(yy[k])) > 1 and len(np.unique(np.round(p[k], 9))) > 1:
            r = spearmanr(p[k], yy[k])[0]
            if np.isfinite(r):
                rs.append(r)
                ws.append(k.sum())
    return (np.average(rs, weights=ws) if rs else np.nan), len(rs), int(sum(ws))


g_vault = code[te]
g_vault_video = np.array([f"{c}|{v}" for c, v in zip(code[te], vid[te])])


def evaluate(p):
    yt, vt = y[te], vid[te]
    s = spearmanr(p, yt)[0] if len(np.unique(np.round(p, 9))) > 1 else np.nan
    wv, nv, cv = grouped_srcc(p, yt, g_vault, 5)
    wvv, nvv, cvv = grouped_srcc(p, yt, g_vault_video, args.min_group)
    tv = np.unique(vt)
    b1, b2 = [], []
    for _ in range(args.boot):
        pick = rng.choice(tv, len(tv))
        idx = np.concatenate([np.where(vt == v)[0] for v in pick])
        b1.append(grouped_srcc(p[idx], yt[idx], g_vault[idx], 5)[0])
        b2.append(grouped_srcc(p[idx], yt[idx], g_vault_video[idx], args.min_group)[0])
    ci = lambda b: np.nanpercentile(b, [2.5, 97.5])
    return dict(SRCC=s, within_vault=wv, wv_ci=ci(b1), within_vault_video=wvv, wvv_ci=ci(b2),
                n_vault_groups=nv, n_vault_clips=cv, n_vv_groups=nvv, n_vv_clips=cvv)


# ---- models (same settings as day4_models.py) ------------------------------------------------
def prep(X, n_pca=None):
    Z = StandardScaler().fit(X[tr]).transform(X)
    if n_pca:
        Z = PCA(n_pca, random_state=0).fit(Z[tr]).transform(Z)
        Z = StandardScaler().fit(Z[tr]).transform(Z)
    return Z


def ridge(X):
    Z = prep(X)
    return RidgeCV(alphas=np.logspace(0, 5, 21)).fit(Z[tr], y[tr]).predict(Z[te])


def mlp(X):
    Z = prep(X, n_pca=128)
    mu, sd = y[tr].mean(), y[tr].std()
    preds = [MLPRegressor(hidden_layer_sizes=(64,), alpha=10.0, max_iter=300, random_state=s)
             .fit(Z[tr], (y[tr] - mu) / sd).predict(Z[te]) * sd + mu for s in range(5)]
    return np.mean(preds, 0)


def pairwise(X, min_gap=0.1):
    Z = prep(X, n_pca=256)
    idx = np.where(tr)[0]
    A, B = [], []
    for c in np.unique(code[idx]):
        k = idx[code[idx] == c]
        for a in range(len(k)):
            for b in range(a + 1, len(k)):
                i, j = k[a], k[b]
                if abs(y[i] - y[j]) < min_gap:
                    continue
                if args.pairs == "cross" and vid[i] == vid[j]:
                    continue
                A.append(i); B.append(j)
    A, B = np.array(A), np.array(B)
    D = Z[A] - Z[B]
    lab = (y[A] > y[B]).astype(int)
    clf = LogisticRegression(C=0.01, fit_intercept=False, max_iter=5000).fit(
        np.vstack([D, -D]), np.concatenate([lab, 1 - lab]))
    pairwise.n_pairs = len(A)
    return Z[te] @ clf.coef_[0]


def video_avg(p):
    """Replace each test prediction by its video's mean prediction (competition level only)."""
    out = p.copy()
    for v in np.unique(vid[te]):
        k = vid[te] == v
        out[k] = p[k].mean()
    return out


# ---- run -------------------------------------------------------------------------------------
rows = []
for fname, X in FEATS.items():
    preds = {"ridge": ridge(X), "mlp": mlp(X), "pairwise": pairwise(X)}
    preds["ridge video-avg"] = video_avg(preds["ridge"])
    for mname, p in preds.items():
        r = evaluate(p)
        rows.append({"features": fname, "model": mname, **{k: v for k, v in r.items()
                                                           if not k.endswith("_ci")},
                     "wv_lo": r["wv_ci"][0], "wv_hi": r["wv_ci"][1],
                     "wvv_lo": r["wvv_ci"][0], "wvv_hi": r["wvv_ci"][1]})
        print(f"{fname:11s} {mname:16s} overall {r['SRCC']:.3f} | within vault {r['within_vault']:.3f} "
              f"[{r['wv_ci'][0]:.2f},{r['wv_ci'][1]:.2f}] | within vault+video "
              f"{r['within_vault_video']:.3f} [{r['wvv_ci'][0]:.2f},{r['wvv_ci'][1]:.2f}]", flush=True)

r0 = rows[0]
print(f"\nwithin vault: {r0['n_vault_groups']} groups / {r0['n_vault_clips']} clips (>=5 per group)")
print(f"within vault+video: {r0['n_vv_groups']} groups / {r0['n_vv_clips']} clips "
      f"(>={args.min_group} per group)")
print(f"pairwise trained on {pairwise.n_pairs} pairs (--pairs {args.pairs})")
out = os.path.join(args.out, f"day4_within_video_{args.pairs}.csv")
pd.DataFrame(rows).round(3).to_csv(out, index=False)
print("written", out)