"""Day 5: evaluate zero-shot VLM execution scores against the truth and against our Day 3/4 model.

Compares, on EXACTLY the same test clips the VLM scored:
  VLM zero-shot    E parsed from the VLM's answer
  ridge (3 stages) our frozen-VideoMAE stage model, trained on all train clips (Day 3/4 setup)

Metrics: parse rate; spread of predictions (does the VLM give everyone the same score?);
Spearman overall, within the same vault, within the same vault AND competition; mean absolute error
and bias. 95% CIs by bootstrap over test videos.

Usage (download capstone/vlm/vlm_scores.csv from Drive into a local vlm/ folder first):
    python analysis/day5_eval_vlm.py --vlm vlm/vlm_scores.csv \
        --feats features/videomae_k400_all.npz --index features/videomae_k400_index.csv --out analysis
Writes analysis/day5_vlm_eval.csv and prints a few VLM answers for qualitative inspection.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--vlm", default="vlm/vlm_scores.csv")
ap.add_argument("--feats", default="features/videomae_k400_all.npz")
ap.add_argument("--index", default="features/videomae_k400_index.csv")
ap.add_argument("--out", default=".")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()

# ---- VLM results -----------------------------------------------------------------------------
v = pd.read_csv(args.vlm, dtype={"vault_code": str})
v = v.drop_duplicates("clip_id", keep="last")
n_all = len(v)
parsed = v.E_vlm.notna()
print(f"VLM model: {v.model.iloc[0]}")
print(f"clips scored: {n_all} | E parsed: {parsed.sum()} ({100 * parsed.mean():.0f}%) | "
      f"errors: {(v.error.fillna('') != '').sum()}")

# ---- our model on the same clips --------------------------------------------------------------
F = np.load(args.feats, allow_pickle=True)
m = pd.read_csv(args.index, dtype={"vault_code": str})
assert (F["ids"] == m.clip_id.values).all(), "features and index are not aligned"
W, S, hs = F["whole"].astype(np.float32), F["stages"].astype(np.float32), F["has_stages"].astype(bool)
X = np.where(hs[:, None], S.reshape(len(S), -1), np.tile(W, 3))
tr = (m.split == "train").values
sc = StandardScaler().fit(X[tr])
ridge = RidgeCV(alphas=np.logspace(0, 5, 21)).fit(sc.transform(X[tr]), m.E.values[tr])
pos = {c: i for i, c in enumerate(m.clip_id)}
missing = [c for c in v.clip_id if c not in pos]
if missing:
    print(f"warning: {len(missing)} VLM clips not in the feature index; they are dropped")
v = v[v.clip_id.isin(pos)].reset_index(drop=True)
v["E_ridge"] = ridge.predict(sc.transform(X[[pos[c] for c in v.clip_id]]))

# ---- metrics ----------------------------------------------------------------------------------
def grouped(p, y, g, min_n):
    rs, ws = [], []
    for k in np.unique(g):
        i = g == k
        if i.sum() >= min_n and len(np.unique(y[i])) > 1 and len(np.unique(np.round(p[i], 9))) > 1:
            r = spearmanr(p[i], y[i])[0]
            if np.isfinite(r):
                rs.append(r); ws.append(i.sum())
    return np.average(rs, weights=ws) if rs else np.nan


def srcc(p, y):
    return spearmanr(p, y)[0] if len(np.unique(np.round(p, 9))) > 1 else np.nan


def metrics(d, col):
    p, y = d[col].values.astype(float), d.E_true.values.astype(float)
    g1 = d.vault_code.values
    g2 = (d.vault_code + "|" + d.video_id).values
    return {"overall": srcc(p, y), "within_vault": grouped(p, y, g1, 5),
            "within_vault_video": grouped(p, y, g2, 2),
            "MAE": np.mean(np.abs(p - y)), "bias": np.mean(p - y),
            "pred_std": np.std(p), "distinct": len(np.unique(np.round(p, 2)))}


rng = np.random.default_rng(0)
rows = []
for name, col, d in [("VLM zero-shot", "E_vlm", v[v.E_vlm.notna()]),
                     ("ridge 3 stages (ours)", "E_ridge", v[v.E_vlm.notna()]),
                     ("ridge 3 stages (ours), all scored clips", "E_ridge", v)]:
    d = d.reset_index(drop=True)
    r = metrics(d, col)
    vids = d.video_id.unique()
    boots = {k: [] for k in ["overall", "within_vault", "within_vault_video"]}
    for _ in range(args.boot):
        pick = rng.choice(vids, len(vids))
        b = pd.concat([d[d.video_id == x] for x in pick], ignore_index=True)
        mb = metrics(b, col)
        for k in boots:
            boots[k].append(mb[k])
    ci = {k: np.nanpercentile(boots[k], [2.5, 97.5]) for k in boots}
    rows.append({"model": name, "n_clips": len(d), **r,
                 **{f"{k}_lo": ci[k][0] for k in ci}, **{f"{k}_hi": ci[k][1] for k in ci}})

y = v.E_true
print(f"\ntrue E on these clips: mean {y.mean():.2f}, std {y.std():.3f}, range {y.min():.2f}-{y.max():.2f}")
print(f"{'model':42s} {'n':>3s}  {'overall':>16s}  {'within vault':>16s}  {'vault+video':>16s}  "
      f"{'MAE':>5s} {'bias':>6s} {'pred std':>8s} {'distinct':>8s}")
for r in rows:
    f = lambda k: f"{r[k]:.3f} [{r[k + '_lo']:.2f},{r[k + '_hi']:.2f}]"
    print(f"{r['model']:42s} {r['n_clips']:3d}  {f('overall'):>16s}  {f('within_vault'):>16s}  "
          f"{f('within_vault_video'):>16s}  {r['MAE']:.3f} {r['bias']:+.3f} {r['pred_std']:8.3f} "
          f"{r['distinct']:8d}")
pd.DataFrame(rows).round(3).to_csv(os.path.join(args.out, "day5_vlm_eval.csv"), index=False)

print("\nExample VLM answers (true E in brackets):")
for _, r in v.sample(min(4, len(v)), random_state=1).iterrows():
    print(f"--- {r.clip_id} [true E {r.E_true}]\n{str(r.response).strip()}")
print("\nwritten", os.path.join(args.out, "day5_vlm_eval.csv"))
