"""Week 2, Wed/Thu: do the system's fault explanations match what people see?

BLIND check: the checklist shows neither the judges' score nor the system's faults, so the team
marks what they actually see in the video.

Step 1 (make the checklist, 20 test clips: lowest, highest and middle E, mixed):
    python pose/fault_check.py make
  -> data/fault_check/fault_check.csv. For each clip, open clips/<clip_file>, jump to
     watch_from_s (about 1 s before touchdown) and mark:
       steps            0, 1 or 2 (2 = two or more, or a large step / hop)
       deep_squat       y/n   knees bent well below a normal landing
       trunk_forward    y/n   chest folded forward on landing
       arm_swing        y/n   arms swung to keep balance
       fall_or_hands    y/n   fell, sat down or touched the mat with hands
       bent_knees_air   y/n   knees visibly bent in the second flight (when the vault is straight)
       legs_apart_air   y/n   legs visibly apart in the second flight
     leave a cell empty if you cannot tell (camera cut, too small)

Step 2 (score it):
    python pose/fault_check.py score
  -> agreement per fault (system flag vs people), and whether people's fault counts follow the
     judges' E. Written to data/fault_check/fault_check_report.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import symbols as S  # noqa: E402

# human item -> system symbols that should flag it (any of them)
ITEMS = {
    "steps": ["ld_feet_range", "ld_step_events"],
    "deep_squat": ["ld_knee_min"],
    "trunk_forward": ["ld_hip_min"],
    "arm_swing": ["ld_arm_range"],
    "fall_or_hands": ["ld_knee_min", "ld_hip_min"],
    "bent_knees_air": ["sf_knee_min"],
    "legs_apart_air": ["sf_leg_sep_max"],
}


FALL_Z = 2.0   # a fall / hands down is only flagged for an extreme landing (>2 SD deep or folded)


def system_z(row, model):
    """How far each symbol is on the fault side of the same vault's typical value, in SDs."""
    out = {}
    for s in model["symbols"]:
        v = row.get(s, np.nan)
        if pd.isna(v):
            out[s] = np.nan; continue
        med, sdv = model["ref"][s].get(str(row["vault_code"]), model["ref"][s]["__all__"])
        out[s] = float(S.SYMBOLS[s][1] * (v - med) / sdv)
    return out


def make(args):
    sym, kf, out = Path(args.sym), Path(args.kf), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "fault_check.csv"
    if f.exists() and not args.force:
        sys.exit(f"{f} exists (your answers?). Use --force to overwrite.")
    D = pd.read_csv(sym / "symbols.csv", dtype={"vault_code": str})
    K = pd.read_csv(kf / "keyframes.csv")[["clip_id", "clip_file", "fps", "k_touchdown"]]
    d = D[D.split == "test"].merge(K, on="clip_id").dropna(subset=["ld_knee_min"]).sort_values("E")
    n = args.n
    pick = pd.concat([d.head(n // 3), d.tail(n // 3),
                      d.iloc[len(d) // 2 - (n - 2 * (n // 3)) // 2:][: n - 2 * (n // 3)]])
    pick = pick.drop_duplicates("clip_id").sample(frac=1, random_state=1)        # mix, hide order
    c = pd.DataFrame({"clip_id": pick.clip_id, "clip_file": pick.clip_file,
                      "watch_from_s": (pick.k_touchdown / pick.fps - 1.0).clip(lower=0).round(1)})
    for col in ITEMS:
        c[col] = ""
    c["notes"] = ""
    c.to_csv(f, index=False)
    print(f"written {f} ({len(c)} clips). Fill it in WITHOUT looking at explanations.csv, then run: "
          f"python pose/fault_check.py score")


def score(args):
    sym, out = Path(args.sym), Path(args.out)
    model = S.load_model(sym / "rules_model.json")
    D = pd.read_csv(sym / "symbols.csv", dtype={"vault_code": str}).set_index("clip_id")
    H = pd.read_csv(out / "fault_check.csv", dtype=str)
    yn = lambda x: {"y": 1.0, "yes": 1.0, "n": 0.0, "no": 0.0}.get(str(x).strip().lower(), np.nan)
    lines = []
    rows = []
    for _, h in H.iterrows():
        r = D.loc[h.clip_id].to_dict(); r["clip_id"] = h.clip_id
        z = system_z(r, model)
        for item, syms in ITEMS.items():
            raw = h[item]
            if item == "steps":
                hv = pd.to_numeric(raw, errors="coerce")
                hv = np.nan if pd.isna(hv) else float(hv > 0)
            else:
                hv = yn(raw)
            thr = FALL_Z if item == "fall_or_hands" else model["tol"]
            sv = [float(z[s] > thr) for s in syms if np.isfinite(z.get(s, np.nan))]
            rows.append(dict(clip_id=h.clip_id, item=item, human=hv,
                             system=(max(sv) if sv else np.nan), E=r["E"]))
    R = pd.DataFrame(rows)
    lines.append(f"FAULT CHECK: {H.clip_id.nunique()} clips, blind review vs system flags "
                 f"(fault beyond {model['tol']} SD of the same vault)\n")
    lines.append(f"{'fault':16s} {'clips':>5s} {'people saw':>10s} {'system flagged':>14s} {'agree':>6s} "
                 f"{'caught':>7s} {'false alarms':>12s}")
    tot_ok = tot_n = 0
    for item in ITEMS:
        x = R[(R.item == item)].dropna(subset=["human", "system"])
        if not len(x):
            lines.append(f"{item:16s} {'-':>5s}   (not filled in or not measured)"); continue
        agree = (x.human == x.system).mean()
        caught = x[x.human == 1].system.mean() if (x.human == 1).any() else np.nan
        false = x[x.human == 0].system.mean() if (x.human == 0).any() else np.nan
        tot_ok += (x.human == x.system).sum(); tot_n += len(x)
        lines.append(f"{item:16s} {len(x):5d} {x.human.mean():10.0%} {x.system.mean():14.0%} {agree:6.0%} "
                     f"{caught:7.0%} {false:12.0%}")
    lines.append(f"\noverall agreement: {tot_ok}/{tot_n} = {tot_ok / max(1, tot_n):.0%}")
    lines.append("caught = of the faults people saw, how many the system flagged (recall)")
    lines.append("false alarms = of the clips people saw NO such fault, how many the system flagged")

    # do people's faults follow the judges? (sanity check of the review itself)
    cnt = R.groupby("clip_id").agg(n_faults=("human", "sum"), E=("E", "first"))
    if cnt.n_faults.nunique() > 1:
        r = spearmanr(cnt.n_faults, cnt.E)[0]
        lines.append(f"\npeople's fault count vs judges' E: Spearman {r:.2f} (expected negative: more faults, lower E)")
    sysc = R.groupby("clip_id").agg(n=("system", "sum"), E=("E", "first"))
    if sysc.n.nunique() > 1:
        lines.append(f"system's fault count vs judges' E: Spearman {spearmanr(sysc.n, sysc.E)[0]:.2f}")
    dis = R.dropna(subset=["human", "system"])
    dis = dis[dis.human != dis.system]
    if len(dis):
        lines.append("\ndisagreements (look at these clips to see who is right):")
        for _, d in dis.head(15).iterrows():
            lines.append(f"  {d['clip_id']}: {d['item']} -> people {'yes' if d.human else 'no'}, "
                         f"system {'yes' if d.system else 'no'}")
    text = "\n".join(lines)
    print(text)
    (out / "fault_check_report.txt").write_text(text, encoding="utf-8")
    print("\nwritten", out / "fault_check_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--sym", default="data/symbols")
    ap.add_argument("--kf", default="data/keyframes")
    ap.add_argument("--out", default="data/fault_check")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    make(args) if args.mode == "make" else score(args)


if __name__ == "__main__":
    main()
