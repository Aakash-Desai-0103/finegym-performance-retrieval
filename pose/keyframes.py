"""Week 2, Mon: clean the gymnast's pose and find the key frames of each vault.

FineGym's three vault segments are: run-up | round-off + springboard + first flight onto the table
| table push-off + SECOND FLIGHT + landing. The second flight and the landing are where most
execution deductions happen, so the third segment is split here using the pose.

Key frames per clip (frame indices in the clip):
  K1 run_end        end of the run-up (start of the round-off)              from the annotation
  K2 table          hands on the vault table (start of the third segment)  from the annotation
  K3 flight_peak    peak of the second flight (hips highest)               pose
  K4 touchdown      first upright frame after the flight (feet down)       pose
  K5 deepest_squat  deepest knee bend just after touchdown (absorbing)     pose
  K6 settled        ~1 s after touchdown: steps / balance show here        pose + time
Phases saved for the symbols (Tue): second_flight = [K2, K4), landing = [K4, end of segment).
If the pose cannot locate a key frame, a fixed position inside the segment is used and the
method column says 'fallback'.

Cleaning per frame: drop camera close-ups (gymnast box much larger than usual), drop frames with
too few lower-body points, fill gaps of <=3 frames, light median smoothing.

Usage (from the repo root):
    python pose/keyframes.py --gym-dir data/pose/yolo26l_960/gymnast_v3 \
        --index features/videomae_k400_index.csv --clips clips --out data/keyframes
Outputs in --out:
    keyframes.csv           one row per clip: windows, K1-K6, methods, quality counts
    pose_clean/<clip>.npz   cleaned keypoints + valid / close-up masks (input for Tue and Wed)
    sheets/<clip>.png       6 key frames side by side, for --n-sheets clips (needs --clips)
    keyframe_review.csv     checklist for the sheets
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posekit as pk  # noqa: E402

KEYS = ["run_end", "table", "flight_peak", "touchdown", "deepest_squat", "settled"]
FALLBACK = {"flight_peak": 0.30, "touchdown": 0.55, "deepest_squat": 0.62, "settled": 0.90}


# ------------------------------------------------------------------------------ cleaning
def clean(k, b, closeup_ratio=1.8, min_lower=4, max_gap=3, smooth=3):
    T = len(k)
    h = b[:, 3] - b[:, 1]
    med_h = np.nanmedian(h) if np.isfinite(h).any() else np.nan
    closeup = np.isfinite(h) & (h > closeup_ratio * med_h)
    lower_ok = np.isfinite(k[:, pk.LOWER, 0]).sum(1) >= min_lower
    valid = np.isfinite(h) & ~closeup & lower_ok
    kc = k.copy()
    kc[~valid] = np.nan
    xy = kc[..., :2].reshape(T, -1)
    df = pd.DataFrame(xy).interpolate(limit=max_gap, limit_area="inside")
    df = df.rolling(smooth, center=True, min_periods=1).median()
    filled = df.values.reshape(T, 17, 2)
    out = np.concatenate([filled, kc[..., 2:3]], -1).astype(np.float32)
    valid_f = np.isfinite(out[:, pk.LOWER, 0]).sum(1) >= min_lower
    return out, valid_f, closeup


def body_y(k):
    """Mean image-y (down = larger) of shoulders, hips, ankles per frame."""
    m = lambda a, c: np.nanmean(k[:, [pk.KP[a], pk.KP[c]], 1], axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return m("l_sho", "r_sho"), m("l_hip", "r_hip"), m("l_ank", "r_ank")


# ------------------------------------------------------------------------------ key frames
def find_keys(k, valid, fr, fps):
    keys, meth = {}, {}
    if not all(s in fr for s in pk.STAGES):
        return None, None
    keys["run_end"] = fr["approach"][1] - 1; meth["run_end"] = "annotation"
    a, z = fr["landing"]
    keys["table"] = a; meth["table"] = "annotation"
    n = max(1, z - a)
    fb = lambda name: a + int(round(FALLBACK[name] * (n - 1)))
    sho, hip, ank = body_y(k)
    upright = valid & (sho < hip) & (hip < ank)
    ang = pk.joint_angles(k)["knee"]

    # touchdown: first upright frame after the first ~20% of the segment that STAYS upright
    # (>=70% of the valid frames in the next 0.3 s), so a body passing upright mid-air
    # during a salto is not taken for the landing
    start = a + max(1, int(0.2 * n))
    w = max(2, int(round(0.3 * fps)))

    def stays_up(t):
        seg = np.arange(t, min(z, t + w))
        v = valid[seg]
        return v.sum() >= 2 and upright[seg][v].mean() >= 0.7

    t_up = next((t for t in range(start, z - 1) if upright[t] and stays_up(t)), None)

    # v2 touchdown. The body opens to upright BEFORE the feet reach the mat (review: touchdown was
    # early in ~45% of clips), so upright only starts the search. Landing = knees absorbing:
    #   deepest squat = smallest knee angle in the 0.8 s after the body is upright
    #   touchdown     = last frame before that squat where the knees were still (nearly) extended,
    #                   i.e. the moment the absorption starts
    # Stiff landings (knees barely bend): touchdown = when the ankles stop moving down.
    kn = pd.Series(np.where(valid, ang, np.nan)).interpolate(limit=3, limit_area="inside") \
        .rolling(3, center=True, min_periods=1).median().values
    if t_up is None:
        td, m_td = fb("touchdown"), "fallback"
        ds, m_ds = fb("deepest_squat"), "fallback"
    else:
        win = np.arange(t_up, min(z, t_up + int(round(0.8 * fps)) + 1))
        okw = win[np.isfinite(kn[win])]
        if len(okw) >= 3:
            ds = int(okw[np.argmin(kn[okw])]); m_ds = "pose"
            pre = okw[okw <= ds]
            kmax = np.nanmax(kn[pre])
            if kmax - kn[ds] >= 10:                                   # clear absorption
                ext = pre[kn[pre] >= kmax - 8]
                td, m_td = int(ext[-1]), "pose"
            else:
                # small knee bend (stiff landing): just before the deepest bend. (v2 used the
                # ankles stopping in the image, but the camera follows the gymnast down, so the
                # ankles look still while she is still in the air.)
                td, m_td = max(int(t_up), ds - int(round(0.12 * fps))), "pose-stiff"
        else:
            td, m_td = int(t_up), "upright"
            ds, m_ds = fb("deepest_squat"), "fallback"
    keys["touchdown"], meth["touchdown"] = td, m_td
    keys["deepest_squat"], meth["deepest_squat"] = ds, m_ds

    # flight peak: highest hips (smallest y) after the hands have left the table (skip the first
    # 0.15 s of the segment: the handstand on the table also has the hips high) and before touchdown
    seg = np.arange(a + int(round(0.15 * fps)), td)
    ok = seg[valid[seg] & np.isfinite(hip[seg])] if len(seg) else seg
    if len(ok) >= 3:
        keys["flight_peak"], meth["flight_peak"] = int(ok[np.argmin(hip[ok])]), "pose"
    else:
        keys["flight_peak"], meth["flight_peak"] = (a + td) // 2 if td > a else fb("flight_peak"), "fallback"

    # settled: about 1 s after touchdown, inside the segment
    s = min(z - 1, td + int(round(1.0 * fps)))
    keys["settled"], meth["settled"] = s, "time" if meth["touchdown"] != "fallback" else "fallback"
    return keys, meth


# ------------------------------------------------------------------------------ sheets
def key_sheet(video, k, b, keys, meth, title, size=200):
    import cv2
    want = sorted(set(keys.values()))
    cap, frames, i = cv2.VideoCapture(str(video)), {}, 0
    while i <= max(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
        i += 1
    cap.release()
    tiles = []
    for name in KEYS:
        f = keys[name]
        img = frames.get(f, np.zeros((size, size, 3), np.uint8)).copy()
        box = b[f] if f < len(b) else None
        if box is not None and np.isfinite(box).all():
            pk.draw_skeleton(img, k[f])
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            s = min(max(box[2] - box[0], box[3] - box[1]) * 0.8 + 20, max(img.shape[:2]) / 2)
            pad = int(s) + 1
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            x0, y0 = int(cx - s) + pad, int(cy - s) + pad
            img = img[y0:y0 + int(2 * s), x0:x0 + int(2 * s)]
        t = cv2.resize(img, (size, size))
        lab = np.full((44, size, 3), 255, np.uint8)
        cv2.putText(lab, f"{name}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(lab, f"f{f} {meth[name]}", (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200) if meth[name] == "fallback" else (0, 120, 0), 1)
        tiles.append(np.vstack([lab, t]))
    row = np.hstack(tiles)
    head = np.full((32, row.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return np.vstack([head, row])


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym-dir", default="data/pose/yolo26l_960/gymnast_v3")
    ap.add_argument("--index", default="features/videomae_k400_index.csv")
    ap.add_argument("--clips", default="clips", help="folder with the clip .mp4 files (for sheets)")
    ap.add_argument("--out", default="data/keyframes")
    ap.add_argument("--n-sheets", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out); (out / "pose_clean").mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.index, dtype={"vault_code": str}).set_index("clip_id")
    files = sorted(Path(args.gym_dir).glob("*.npz"))
    assert files, f"no .npz files in {args.gym_dir}"
    rows = []
    for f in files:
        cid = f.stem
        g = np.load(f)
        k, b, fps = g["kpts"], g["boxes"], float(g["fps"])
        fr = {s: tuple(v) for s, v in json.loads(str(g["stage_frames"])).items()}
        kc, valid, closeup = clean(k, b)
        keys, meth = find_keys(kc, valid, fr, fps)
        row = dict(clip_id=cid, fps=fps, n_frames=len(k), has_stages=keys is not None,
                   closeup_frames=int(closeup.sum()))
        if cid in m.index:
            row.update(video_id=m.at[cid, "video_id"], vault_code=m.at[cid, "vault_code"],
                       E=m.at[cid, "E"], split=m.at[cid, "split"], clip_file=m.at[cid, "clip_file"])
        if keys:
            for name in KEYS:
                row[f"k_{name}"] = keys[name]; row[f"m_{name}"] = meth[name]
            a, z = fr["landing"]
            row["second_flight_s"] = (keys["touchdown"] - keys["table"]) / fps
            sf = np.arange(keys["table"], keys["touchdown"])
            ld = np.arange(keys["touchdown"], z)
            row["second_flight_valid"] = float(valid[sf].mean()) if len(sf) else np.nan
            row["landing_valid"] = float(valid[ld].mean()) if len(ld) else np.nan
        rows.append(row)
        np.savez_compressed(out / "pose_clean" / f"{cid}.npz", kpts=kc, valid=valid, closeup=closeup,
                            boxes=b, fps=fps, stage_frames=json.dumps(fr),
                            keys=json.dumps(keys or {}), methods=json.dumps(meth or {}))
    K = pd.DataFrame(rows)
    K.to_csv(out / "keyframes.csv", index=False)

    # ---- summary
    S = K[K.has_stages]
    print(f"clips: {len(K)} | with stage annotations: {len(S)} | close-up frames dropped: "
          f"{K.closeup_frames.sum()} (in {(K.closeup_frames > 0).sum()} clips)")
    print("key frames located from the pose (rest: fallback position):")
    for name in ["flight_peak", "touchdown", "deepest_squat"]:
        print(f"  {name:14s} {S[f'm_{name}'].astype(str).str.startswith('pose').mean():.0%}")
    print("  touchdown method:", S.m_touchdown.value_counts().to_dict(),
          "(pose = knee absorption, pose-stiff = small knee bend, upright = body upright only)")
    posed = S.m_touchdown.astype(str).str.startswith("pose")
    sf = S.loc[posed, "second_flight_s"]
    print(f"second flight (table -> touchdown), pose-located clips: median {sf.median():.2f} s, "
          f"10-90% range {sf.quantile(.1):.2f}-{sf.quantile(.9):.2f} s")
    print(f"valid pose frames: second flight {S.second_flight_valid.median():.0%} (median), "
          f"landing {S.landing_valid.median():.0%}")
    odd = S[posed & ((S.second_flight_s < 0.3) | (S.second_flight_s > 1.6))]
    print(f"suspicious second-flight durations (<0.3 s or >1.6 s): {len(odd)} clips")

    # ---- sheets + review checklist
    clips_dir = Path(args.clips)
    if not clips_dir.exists():
        print(f"\n{clips_dir} not found: skipping sheets (pass --clips <folder with the .mp4 files>)")
        return
    (out / "sheets").mkdir(exist_ok=True)
    pick = S.sample(min(args.n_sheets, len(S)), random_state=0) if len(S) else S
    import cv2
    made = []
    for r in pick.itertuples():
        video = clips_dir / str(getattr(r, "clip_file", f"{r.clip_id}.mp4"))
        if not video.exists():
            continue
        z = np.load(out / "pose_clean" / f"{r.clip_id}.npz")
        img = key_sheet(video, z["kpts"], z["boxes"], json.loads(str(z["keys"])),
                        json.loads(str(z["methods"])),
                        f"{r.clip_id} | vault {r.vault_code} | E {r.E} | 2nd flight {r.second_flight_s:.2f}s")
        cv2.imwrite(str(out / "sheets" / f"{r.clip_id}.png"), img)
        made.append(r.clip_id)
    rev = out / "keyframe_review.csv"
    if made and not rev.exists():
        c = pd.DataFrame({"clip_id": made, "sheet": [f"sheets/{x}.png" for x in made]})
        for col in ["flight_peak_ok", "touchdown_ok", "deepest_squat_ok", "settled_ok", "notes"]:
            c[col] = ""
        c.to_csv(rev, index=False)
    print(f"\n{len(made)} key-frame sheets in {out / 'sheets'}; checklist: {rev}")


if __name__ == "__main__":
    main()
