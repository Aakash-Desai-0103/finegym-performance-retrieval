"""Week 2, Tue: judge-style symbols per vault + an explainable rule-based execution score.

Input: the output of keyframes.py (keyframes.csv + pose_clean/<clip>.npz).

1. SYMBOLS: measurements a judge looks at, per phase (lengths in leg lengths, angles in degrees,
   180 = straight):
     second flight [table, touchdown):
       sf_knee_min      most bent knees               (bent knees)
       sf_hip_min       most closed hip angle         (piked / broken body line)
       sf_leg_sep_max   widest gap between the ankles (legs apart)
       sf_duration_s    time in the air               (height / amplitude)
     landing [touchdown, touchdown + 1.5 s):
       ld_knee_min      deepest squat                 (deep landing)
       ld_hip_min       trunk folded forward          (trunk movement)
       ld_feet_range    change in the gap between the feet (steps)
       ld_step_events   number of separate foot movements (steps)
       ld_balance_range hips moving over the feet     (balance)
       ld_arm_range     wrists moving vs shoulders    (arm swings)

2. FAULTS: each symbol is compared with the SAME VAULT performed by other gymnasts in the TRAIN
   split (vaults differ: a tucked vault bends the knees on purpose). fault = how far the clip is
   on the bad side of the typical value for its vault, in standard deviations, minus a 0.5 SD
   tolerance, clipped at 0. A missing symbol gives no fault (and is reported).

3. SCORES (all explainable as a list of deductions):
     rules_fixed   E = baseline - sum(w_i * fault_i), w_i = starting values in the spirit of the
                   FIG Code of Points (0.1 small, 0.3 medium per SD), NOT fitted to E
     rules_fitted  same form, w_i >= 0 fitted on train E (non-negative least squares), so each
                   symbol can only ever deduct, never add
     ridge_symbols unconstrained linear model on the symbols (not explainable as deductions;
                   reference for how much the symbols carry)
   Evaluated like Days 3-4: SRCC overall, within the same vault, within vault + competition,
   95% CIs by bootstrapping test videos.

Usage:
    python pose/symbols.py --kf data/keyframes --out data/symbols
Writes symbols.csv, explanations.csv, symbols_report.txt, rules_model.json into --out.
"""
import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posekit as pk  # noqa: E402

# symbol: (phase, bad direction: -1 = smaller is worse, +1 = larger is worse, fixed weight per SD,
#          plain-language fault)
SYMBOLS = {
    "sf_knee_min":      ("second flight", -1, 0.10, "knees bent"),
    "sf_hip_min":       ("second flight", -1, 0.10, "body line broken (hips piked)"),
    "sf_leg_sep_max":   ("second flight", +1, 0.10, "legs apart"),
    "sf_duration_s":    ("second flight", -1, 0.10, "low / short second flight"),
    "ld_knee_min":      ("landing", -1, 0.10, "deep squat on landing"),
    "ld_hip_min":       ("landing", -1, 0.10, "trunk folded forward on landing"),
    "ld_feet_range":    ("landing", +1, 0.30, "steps on landing"),
    "ld_step_events":   ("landing", +1, 0.10, "several foot movements on landing"),
    "ld_balance_range": ("landing", +1, 0.10, "balance (hips moving over feet)"),
    "ld_arm_range":     ("landing", +1, 0.10, "arm swings on landing"),
}
TOL = 0.5          # SDs of tolerance before a fault counts
MIN_REF = 5        # min train clips of a vault to use vault-specific reference values


# ------------------------------------------------------------------------------ symbols
LANDING_S = 1.5     # landing window: touchdown + 1.5 s (excludes saluting / walking off the mat)
MIN_FRAMES = 3      # a symbol needs at least this many good frames


def plausible(k, L):
    """Frames whose thigh and shin lengths are within [0.4, 1.8] x the clip's median
    (rejects swapped / garbage keypoints; generous because of foreshortening in flight)."""
    p = k[..., :2]
    seg = lambda a, b: np.nanmean(np.stack([np.linalg.norm(p[:, pk.KP[f"{s}_{a}"]] - p[:, pk.KP[f"{s}_{b}"]], axis=-1)
                                            for s in "lr"]), axis=0)
    ok = np.ones(len(k), bool)
    for a, b in [("hip", "knee"), ("knee", "ank")]:
        x = seg(a, b)
        med = np.nanmedian(x)
        ok &= np.isfinite(x) & (x > 0.4 * med) & (x < 1.8 * med)
    return ok


def phase_symbols(k, valid, keys, fr, fps):
    """v2: robust percentiles instead of min/max, implausible frames removed, landing window
    limited to 1.5 s, steps counted as sustained foot movements."""
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    out = {}
    if not keys:
        return out
    a, z = fr["landing"]
    t0, td = keys["table"], keys["touchdown"]
    L = pk.body_scale(k)
    good = valid & plausible(k, L)
    ang = pk.joint_angles(k)
    p = k[..., :2]
    sf = np.arange(t0, td)
    sf = sf[good[sf]] if len(sf) else sf
    ld_all = np.arange(td, min(z, td + int(round(LANDING_S * fps))))
    ld = ld_all[good[ld_all]] if len(ld_all) else ld_all

    def q(x, idx, pct):
        v = x[idx] if len(idx) else np.array([])
        v = v[np.isfinite(v)]
        return float(np.percentile(v, pct)) if len(v) >= MIN_FRAMES else np.nan

    def spread(x, idx):
        lo, hi = q(x, idx, 5), q(x, idx, 95)
        return hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan

    out["sf_knee_min"] = q(ang["knee"], sf, 10)
    out["sf_hip_min"] = q(ang["hip"], sf, 10)
    sep = np.linalg.norm(p[:, pk.KP["l_ank"]] - p[:, pk.KP["r_ank"]], axis=-1) / L
    out["sf_leg_sep_max"] = q(sep, sf, 90)
    out["sf_duration_s"] = (td - t0) / fps
    out["ld_knee_min"] = q(ang["knee"], ld, 10)
    out["ld_hip_min"] = q(ang["hip"], ld, 10)
    feet = (p[:, pk.KP["l_ank"], 0] - p[:, pk.KP["r_ank"], 0]) / L      # camera-pan invariant
    out["ld_feet_range"] = spread(feet, ld)
    if len(ld) >= MIN_FRAMES:
        f = pd.Series(np.where(good, feet, np.nan)[ld_all]).interpolate(limit_area="inside")
        f = f.rolling(5, center=True, min_periods=1).median().values
        v = np.abs(np.diff(f)) * fps                                     # leg lengths per second
        moving = np.nan_to_num(v) > 0.6
        runs, n = 0, 0
        for m in moving:                                                 # sustained movements only
            n = n + 1 if m else 0
            runs += n == 3
        out["ld_step_events"] = float(runs)
    else:
        out["ld_step_events"] = np.nan
    hipx = np.nanmean(p[:, [pk.KP["l_hip"], pk.KP["r_hip"]], 0], axis=1)
    ankx = np.nanmean(p[:, [pk.KP["l_ank"], pk.KP["r_ank"]], 0], axis=1)
    out["ld_balance_range"] = spread((hipx - ankx) / L, ld)
    wr = np.nanmean(p[:, [pk.KP["l_wri"], pk.KP["r_wri"]], 1], axis=1)
    sh = np.nanmean(p[:, [pk.KP["l_sho"], pk.KP["r_sho"]], 1], axis=1)
    out["ld_arm_range"] = spread((wr - sh) / L, ld)
    out["sf_valid_frames"] = int(len(sf)); out["ld_valid_frames"] = int(len(ld))
    return out


# ------------------------------------------------------------------------------ faults
def reference(df, train):
    """Per-vault (and global fallback) median and robust SD of each symbol, train clips only."""
    ref = {}
    tr = df[train]
    for s in SYMBOLS:
        g = {"__all__": (tr[s].median(), 1.4826 * (tr[s] - tr[s].median()).abs().median() + 1e-6)}
        for code, d in tr.groupby("vault_code"):
            x = d[s].dropna()
            if len(x) >= MIN_REF:
                g[code] = (x.median(), 1.4826 * (x - x.median()).abs().median() + 1e-6)
        ref[s] = g
    return ref


def faults(df, ref):
    F = pd.DataFrame(index=df.index)
    for s, (_, sign, _, _) in SYMBOLS.items():
        med = df.vault_code.map(lambda c: ref[s].get(c, ref[s]["__all__"])[0]).astype(float)
        sd = df.vault_code.map(lambda c: ref[s].get(c, ref[s]["__all__"])[1]).astype(float)
        z = sign * (df[s] - med) / sd
        F[s] = np.clip(z - TOL, 0, 3).fillna(0.0)
    return F


def load_model(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    m["ref"] = {s: {c: tuple(v) for c, v in g.items()} for s, g in m["ref"].items()}
    return m


def deductions(row, model):
    """Per-symbol deduction (fitted rules) for one clip (a symbols.csv row) -> {symbol: points}."""
    out = {}
    for j, s in enumerate(model["symbols"]):
        v = row.get(s, np.nan)
        if pd.isna(v):
            continue
        med, sdv = model["ref"][s].get(str(row["vault_code"]), model["ref"][s]["__all__"])
        z = SYMBOLS[s][1] * (v - med) / sdv
        out[s] = model["w_fit"][j] * float(np.clip(z - model["tol"], 0, 3))
    return out


# ------------------------------------------------------------------------------ evaluation
def grouped(p, y, g, min_n):
    rs, ws = [], []
    for k in np.unique(g):
        i = g == k
        if i.sum() >= min_n and len(np.unique(y[i])) > 1 and len(np.unique(np.round(p[i], 9))) > 1:
            r = spearmanr(p[i], y[i])[0]
            if np.isfinite(r):
                rs.append(r); ws.append(i.sum())
    return np.average(rs, weights=ws) if rs else np.nan


def evaluate(p, d, boot=1000, seed=0):
    y = d.E.values.astype(float)
    g1 = d.vault_code.values
    g2 = (d.vault_code + "|" + d.video_id).values
    f = lambda pp, yy, a, b: (spearmanr(pp, yy)[0] if len(np.unique(np.round(pp, 9))) > 1 else np.nan,
                              grouped(pp, yy, a, 5), grouped(pp, yy, b, 3))
    point = f(p, y, g1, g2)
    rng = np.random.default_rng(seed)
    vids = d.video_id.values
    uv = np.unique(vids)
    B = []
    for _ in range(boot):
        idx = np.concatenate([np.where(vids == v)[0] for v in rng.choice(uv, len(uv))])
        B.append(f(p[idx], y[idx], g1[idx], g2[idx]))
    B = np.array(B, float)
    lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)
    return point, lo, hi


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", default="data/keyframes")
    ap.add_argument("--out", default="data/symbols")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    kf, out = Path(args.kf), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    K = pd.read_csv(kf / "keyframes.csv", dtype={"vault_code": str})
    K = K[K.has_stages & K.E.notna() & K.split.isin(["train", "test"])].reset_index(drop=True)
    rows = []
    for r in K.itertuples():
        z = np.load(kf / "pose_clean" / f"{r.clip_id}.npz")
        fr = {s: tuple(v) for s, v in json.loads(str(z["stage_frames"])).items()}
        rows.append(dict(clip_id=r.clip_id, **phase_symbols(z["kpts"], z["valid"], json.loads(str(z["keys"])),
                                                           fr, float(z["fps"]))))
    D = K[["clip_id", "video_id", "vault_code", "E", "split", "m_touchdown"]].merge(pd.DataFrame(rows), on="clip_id")
    D["vault_code"] = D.vault_code.fillna("NA")
    D.to_csv(out / "symbols.csv", index=False)
    train, test = (D.split == "train").values, (D.split == "test").values

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"clips: {len(D)} (train {train.sum()}, test {test.sum()}) | touchdown found by pose: "
              f"{D.m_touchdown.astype(str).str.startswith('pose').mean():.0%}\n")

        # ---- 1. symbols: coverage and direction on TRAIN (within the same vault)
        print("1. SYMBOLS (train): coverage, and Spearman with E within the same vault")
        print("   expected sign: a fault direction of -1 means smaller values should go with LOWER E\n")
        rows = []
        for s, (ph, sign, w, name) in SYMBOLS.items():
            ok = train & D[s].notna().values
            wv = grouped(D.loc[ok, s].values.astype(float), D.loc[ok, "E"].values.astype(float),
                         D.loc[ok, "vault_code"].values, 5)
            rows.append(dict(symbol=s, phase=ph, fault=name, coverage=D.loc[train, s].notna().mean(),
                             median=D.loc[train, s].median(), within_vault_SRCC=wv,
                             expected=("+" if sign < 0 else "-"),
                             agrees=("yes" if np.isfinite(wv) and np.sign(wv) == -sign else "no")))
        T1 = pd.DataFrame(rows)
        print(T1.round(3).to_string(index=False), "\n")

        # ---- 2. faults and scores
        ref = reference(D, train)
        F = faults(D, ref)
        base = D.loc[train, "E"].max()                        # fixed rules start from the best train E
        w_fixed = np.array([SYMBOLS[s][2] for s in SYMBOLS])
        pred = {"rules_fixed": base - F.values @ w_fixed}
        # fitted rules: E = c - F w, w >= 0  ->  NNLS on [F, -1] with c free
        A = np.hstack([F.values[train], np.ones((train.sum(), 1))])
        yv = D.loc[train, "E"].values.astype(float)
        sol, _ = nnls(np.hstack([-A[:, :-1], A[:, -1:]]), yv)
        w_fit, c_fit = sol[:-1], sol[-1]
        model = dict(symbols=list(SYMBOLS), w_fixed=w_fixed.tolist(), base_fixed=float(base),
                     w_fit=w_fit.tolist(), c_fit=float(c_fit), tol=TOL,
                     ref={s: {c: [float(m), float(sdv)] for c, (m, sdv) in g.items()} for s, g in ref.items()})
        (out / "rules_model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
        pred["rules_fitted"] = c_fit - F.values @ w_fit
        X = D[list(SYMBOLS)].copy()
        X = X.fillna(X[train].median())
        sc = StandardScaler().fit(X[train])
        rg = RidgeCV(alphas=np.logspace(-1, 4, 21)).fit(sc.transform(X[train]), yv)
        pred["ridge_symbols"] = rg.predict(sc.transform(X))

        print("2. EXECUTION SCORE ON TEST (95% CI by test videos)")
        print("   reference, frozen VideoMAE (Days 3-4, full test set): within vault 0.476-0.489, "
              "within vault+competition 0.393\n")
        dt = D[test].reset_index(drop=True)
        res = []
        for name, p in pred.items():
            (s, wv, wvv), lo, hi = evaluate(p[test], dt, args.boot)
            res.append(dict(model=name, SRCC=f"{s:.3f} [{lo[0]:.2f},{hi[0]:.2f}]",
                            within_vault=f"{wv:.3f} [{lo[1]:.2f},{hi[1]:.2f}]",
                            within_vault_video=f"{wvv:.3f} [{lo[2]:.2f},{hi[2]:.2f}]"))
        print(pd.DataFrame(res).to_string(index=False))
        n_wv = sum(1 for _, g in dt.groupby("vault_code") if len(g) >= 5)
        print(f"   test vault codes with >=5 clips: {n_wv}\n")

        print("3. DEDUCTION WEIGHTS (per SD beyond tolerance)")
        W = pd.DataFrame({"symbol": list(SYMBOLS), "fault": [v[3] for v in SYMBOLS.values()],
                          "fixed": w_fixed, "fitted": w_fit.round(3),
                          "clips with this fault (test)": (F[test] > 0).mean().values.round(2)})
        print(W.to_string(index=False))
        print(f"   fitted baseline (E with no faults): {c_fit:.2f} | fixed baseline: {base:.2f}")
        print("   fitted weight 0 = the train data gave no evidence this symbol lowers E\n")

        # ---- 4. explanations
        ex = []
        for i in np.where(test)[0]:
            d = [(w_fit[j] * F.iloc[i, j], s) for j, s in enumerate(SYMBOLS) if w_fit[j] * F.iloc[i, j] > 0.005]
            d.sort(reverse=True)
            text = "; ".join(f"{SYMBOLS[s][0]}: {SYMBOLS[s][3]} ({s}={D.at[i, s]:.2f}, "
                             f"typical {ref[s].get(D.at[i, 'vault_code'], ref[s]['__all__'])[0]:.2f}) -{v:.2f}"
                             for v, s in d[:4]) or "no faults beyond tolerance"
            missing = [s for s in SYMBOLS if pd.isna(D.at[i, s])]
            ex.append(dict(clip_id=D.at[i, "clip_id"], vault_code=D.at[i, "vault_code"], E_true=D.at[i, "E"],
                           E_pred=round(pred["rules_fitted"][i], 3), explanation=text,
                           not_measured=", ".join(missing)))
        EX = pd.DataFrame(ex)
        EX.to_csv(out / "explanations.csv", index=False)
        print("4. EXAMPLE EXPLANATIONS (rules_fitted, test clips: 3 lowest and 3 highest true E)")
        for _, r in pd.concat([EX.nsmallest(3, "E_true"), EX.nlargest(3, "E_true")]).iterrows():
            print(f"   {r.clip_id} vault {r.vault_code} | true E {r.E_true} | predicted {r.E_pred}\n"
                  f"      {r.explanation}")

    text = buf.getvalue()
    print(text)
    (out / "symbols_report.txt").write_text(text, encoding="utf-8")
    print("written", out / "symbols_report.txt", "|", out / "symbols.csv", "|", out / "explanations.csv")


if __name__ == "__main__":
    main()
