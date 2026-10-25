"""Week 2, Sun: is the pose good enough to build symbols on? Which model?

Reads the pilot outputs (pose/<model>/summary.csv for each model) and your filled-in
pose/review_checklist.csv, and reports:

  A. model comparison on automatic quality metrics, per stage
  B. your visual review: how often the gymnast was right and each stage usable (primary model)
  C. which automatic metrics separate good from bad stages (so the full run can flag bad clips
     without looking at every sheet)
  D. first symbols against the judges' E (60 clips, no training, so only a direction check)
  E. GO / NO-GO per stage

Usage (from the repo root, after downloading My Drive/capstone/pose/ into ./pose):
    python pose/pose_quality.py --pose-dir pose
Writes <pose-dir>/pilot_report_<version>.txt.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--pose-dir", default="pose")
ap.add_argument("--version", default="v3", help="gymnast-selection version (summary_<v>.csv)")
ap.add_argument("--review", default=None, help="default: <pose-dir>/review_checklist_<version>.csv")
ap.add_argument("--primary", default=None, help="model folder you reviewed (default: from checklist)")
args = ap.parse_args()
P = Path(args.pose_dir)
STAGES = ["approach", "flight", "landing"]
QCOLS = ["coverage", "lower_conf", "lower_visible", "jitter", "limb_cv"]
SYMS = ["flight_knee_min", "flight_knee_mean", "flight_hip_mean", "flight_leg_sep_max",
        "landing_knee_min", "landing_hip_min", "landing_feet_range", "landing_balance_range"]

SUMF = f"summary_{args.version}.csv"
models = {d.name: pd.read_csv(d / SUMF, dtype={"vault_code": str})
          for d in sorted(P.iterdir()) if (d / SUMF).exists()}
assert models, f"no <model>/{SUMF} under {P}"

rev_path = Path(args.review) if args.review else P / f"review_checklist_{args.version}.csv"
rev = pd.read_csv(rev_path, dtype=str) if rev_path.exists() else None
primary = args.primary or (rev.sheet.iloc[0].split("/")[0] if rev is not None else list(models)[0])

buf = io.StringIO()
with redirect_stdout(buf):
    # ---------------------------------------------------------------- A. models
    print("A. AUTOMATIC QUALITY PER MODEL (pilot clips)")
    print("   coverage = frames with the gymnast; usable = coverage>=0.8 and limb_cv<=0.25 "
          "and lower_visible>=0.6\n")
    for name, S in models.items():
        t = {}
        for st in STAGES:
            usable = ((S[f"{st}_coverage"] >= 0.8) & (S[f"{st}_limb_cv"] <= 0.25) &
                      (S[f"{st}_lower_visible"] >= 0.6))
            t[st] = {"coverage (median)": S[f"{st}_coverage"].median(),
                     "share from rotated pass": S.get(f"{st}_from_rotated", pd.Series(dtype=float)).mean(),
                     "lower-body conf (mean)": S[f"{st}_lower_conf"].mean(),
                     "lower visible (mean)": S[f"{st}_lower_visible"].mean(),
                     "jitter (median)": S[f"{st}_jitter"].median(),
                     "limb_cv (median)": S[f"{st}_limb_cv"].median(),
                     "USABLE clips": usable.mean()}
        print(f"--- {name} ({len(S)} clips, gymnast height/frame {S.whole_box_h.median():.2f})"
              f"{'   <- reviewed' if name == primary else ''}")
        print(pd.DataFrame(t).round(3).to_string(), "\n")

    # ---------------------------------------------------------------- B. review
    S = models[primary]
    verdict = {}
    if rev is None or rev[["gymnast_right", "flight_ok"]].replace("", np.nan).dropna().empty:
        print("B. VISUAL REVIEW: checklist not filled in yet; skipping B, C and the verdict.\n")
    else:
        norm = lambda x: {"y": 1.0, "yes": 1.0, "p": 0.5, "partial": 0.5, "n": 0.0, "no": 0.0}.get(
            str(x).strip().lower(), np.nan)
        R = rev.copy()
        for c in ["gymnast_right"] + [f"{s}_ok" for s in STAGES]:
            R[c] = R[c].map(norm)
        done = R.gymnast_right.notna()
        print(f"B. VISUAL REVIEW ({done.sum()} clips reviewed, model {primary})")
        print(f"   gymnast picked correctly: {R.gymnast_right[done].mean():.0%}")
        for st in STAGES:
            v = R[f"{st}_ok"][done]
            print(f"   {st:9s} ok {(v == 1).mean():.0%} | partial {(v == 0.5).mean():.0%} | "
                  f"bad {(v == 0).mean():.0%}")
            verdict[st] = (v == 1).mean() + 0.5 * (v == 0.5).mean()
        verdict["gymnast"] = R.gymnast_right[done].mean()
        notes = R.loc[done & R.notes.notna() & (R.notes.str.strip() != ""), ["clip_id", "notes"]]
        if len(notes):
            print("   your notes:\n" + "\n".join(f"     {a}: {b}" for a, b in notes.values))
        print()

        # ------------------------------------------------------------ C. metrics vs review
        print("C. DO THE AUTOMATIC METRICS AGREE WITH YOUR EYES? (mean metric for stages you marked "
              "ok vs bad)")
        M = S.merge(R[["clip_id"] + [f"{s}_ok" for s in STAGES]], on="clip_id")
        for st in STAGES:
            lab = M[f"{st}_ok"]
            rows = {}
            for q in QCOLS:
                col = M[f"{st}_{q}"]
                rows[q] = {"ok": col[lab == 1].mean(), "partial": col[lab == 0.5].mean(),
                           "bad": col[lab == 0].mean()}
            print(f"--- {st} (ok {int((lab == 1).sum())}, partial {int((lab == 0.5).sum())}, "
                  f"bad {int((lab == 0).sum())})")
            print(pd.DataFrame(rows).T.round(3).to_string())
        print("   A metric is a useful automatic flag if 'ok' and 'bad' differ clearly.\n")

    # ---------------------------------------------------------------- D. symbols vs E
    print(f"D. FIRST SYMBOLS vs JUDGES' E ({primary}, {len(S)} clips; no training, direction only)")
    print("   expected sign: straighter knees / smaller leg separation / less landing movement -> "
          "higher E\n")
    exp = {"flight_knee_min": +1, "flight_knee_mean": +1, "flight_hip_mean": 0, "flight_leg_sep_max": -1,
           "landing_knee_min": +1, "landing_hip_min": +1, "landing_feet_range": -1,
           "landing_balance_range": -1}
    rows = []
    for s in SYMS:
        if s not in S:
            continue
        ok = S[s].notna()
        r = spearmanr(S.loc[ok, s], S.loc[ok, "E"])[0] if ok.sum() > 5 else np.nan
        ws, ns = [], []
        for _, g in S[ok].groupby("vault_code"):
            if len(g) >= 4 and g.E.nunique() > 1 and g[s].nunique() > 1:
                ws.append(spearmanr(g[s], g.E)[0]); ns.append(len(g))
        wv = np.average(ws, weights=ns) if ws else np.nan
        e = exp[s]
        agree = "" if e == 0 or not np.isfinite(r) else ("as expected" if np.sign(r) == e else "OPPOSITE")
        rows.append(dict(symbol=s, n=int(ok.sum()), median=S[s].median(), SRCC_E=r,
                         within_vault=wv, n_vault_clips=sum(ns), sign=agree))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("   With ~60 clips, |SRCC| < 0.25 is indistinguishable from noise. Look for the sign and for "
          "the landing symbols,\n   which carried the most signal in the video features.\n")

    # ---------------------------------------------------------------- E. verdict
    print("E. VERDICT")
    if not verdict:
        print("   fill in the checklist and re-run for a verdict")
    else:
        need = {"gymnast": 0.90, "approach": 0.6, "flight": 0.6, "landing": 0.8}
        for k, thr in need.items():
            v = verdict[k]
            print(f"   {k:9s} {v:.0%} (need {thr:.0%}) -> {'GO' if v >= thr else 'NO-GO'}")
        if verdict["gymnast"] < 0.9:
            print("   -> gymnast picking needs fixing before anything else (send me the bad clip ids)")
        elif verdict["landing"] >= 0.8 and verdict["flight"] < 0.6:
            print("   -> build landing symbols first; flight needs a better pose model or crops")
        elif verdict["landing"] >= 0.8 and verdict["flight"] >= 0.6:
            print("   -> GO for the full run (Monday) with this model")
        else:
            print("   -> pose is not good enough yet; try the other model's sheets on the bad clips")

text = buf.getvalue()
print(text)
out = P / f"pilot_report_{args.version}.txt"
out.write_text(text, encoding="utf-8")
print("written", out)
