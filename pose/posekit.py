"""Pose utilities shared by the Colab notebook and the laptop scripts.

Pipeline per clip:
  1. raw detections   every person the pose model finds in every frame (saved once, on Colab)
  2. gymnast track    offline: link detections into tracklets, score them, chain the best one
                      across ID breaks (flips often break tracking), -> one skeleton per frame
  3. quality          coverage, keypoint confidence, jitter, limb-length stability, per stage
  4. symbols          judge-style measurements (knee angle, leg separation, body line, landing)

Selection runs offline from the raw file, so it can be re-tuned on a laptop without a GPU.
Keypoints: 17 COCO points, (x, y, confidence) in pixels of the original frame.
"""
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_sho=5, r_sho=6, l_elb=7, r_elb=8,
          l_wri=9, r_wri=10, l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ank=15, r_ank=16)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
LOWER = [11, 12, 13, 14, 15, 16]
STAGES = ["approach", "flight", "landing"]
CONF_MIN = 0.3  # keypoints below this confidence are treated as missing


# ---------------------------------------------------------------------------- raw detections
def pack_raw(per_frame, fps, width, height):
    """per_frame: list over frames of (boxes [n,4], conf [n], kpts [n,17,3]) -> flat dict for npz."""
    f, b, c, k = [], [], [], []
    for i, (bb, cc, kk) in enumerate(per_frame):
        n = len(cc)
        f.append(np.full(n, i, np.int32)); b.append(bb); c.append(cc); k.append(kk)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
    return dict(det_frame=cat(f, (0,)).astype(np.int32), det_box=cat(b, (0, 4)).astype(np.float32),
                det_conf=cat(c, (0,)).astype(np.float32),
                det_kpt=cat(k, (0, 17, 3)).astype(np.float16),
                n_frames=np.int32(len(per_frame)), fps=np.float32(fps),
                width=np.int32(width), height=np.int32(height))


def load_raw(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def stage_frames(row, fps, n_frames):
    """Frame ranges [a, b) for whole / approach / flight / landing from the manifest row."""
    off = float(row["stage_offset"])
    to_f = lambda t: int(min(n_frames, max(0, round(t * fps))))
    out = {"whole": (to_f(off), to_f(off + float(row["duration"])))}
    raw = row.get("stages")
    if isinstance(raw, str) and raw.strip().startswith("["):
        st = json.loads(raw)
        if len(st) == 3:
            for name, (a, b) in zip(STAGES, st):
                out[name] = (to_f(off + a), max(to_f(off + a) + 1, to_f(off + b)))
    return out


# ---------------------------------------------------------------------------- rotation pass
# Pose models rarely find upside-down people (handstand on the table, inverted second flight).
# Second pass: rotate frames where the gymnast is missing, detect again, rotate results back.
ROTATIONS = {"r180": 180, "r90cw": 90, "r90ccw": 270}


def unrotate_points(x, y, rot, W, H):
    """Map points from a rotated image back to the original W x H frame.
    rot: 180 (cv2.ROTATE_180), 90 (cv2.ROTATE_90_CLOCKWISE), 270 (cv2.ROTATE_90_COUNTERCLOCKWISE)."""
    if rot == 180:
        return W - 1 - x, H - 1 - y
    if rot == 90:        # clockwise: original (x, y) -> (H-1-y, x)
        return y, H - 1 - x
    if rot == 270:       # counter-clockwise: original (x, y) -> (y, W-1-x)
        return W - 1 - y, x
    return x, y


def unrotate_dets(boxes, kpts, rot, W, H):
    """boxes [n,4], kpts [n,17,3] predicted on a rotated frame -> original frame coordinates."""
    if len(boxes) == 0:
        return boxes, kpts
    xs, ys = unrotate_points(boxes[:, [0, 2, 0, 2]], boxes[:, [1, 1, 3, 3]], rot, W, H)
    b = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], 1)
    k = kpts.copy()
    k[..., 0], k[..., 1] = unrotate_points(kpts[..., 0], kpts[..., 1], rot, W, H)
    return b, k


def nms(boxes, conf, thr=0.5):
    keep = []
    for i in np.argsort(-conf):
        if all(_iou(boxes[i:i + 1], boxes[j:j + 1])[0, 0] < thr for j in keep):
            keep.append(i)
    return np.array(keep, int)


def merge_raw(raw, extra):
    """Add rotated-pass detections (same flat format) to the raw detections."""
    if extra is None or len(extra["det_frame"]) == 0:
        out = dict(raw); out["det_rot"] = np.zeros(len(raw["det_frame"]), np.int8)
        return out
    out = dict(raw)
    keys = ["det_frame", "det_box", "det_conf", "det_kpt", "det_rot"]
    base_rot = raw.get("det_rot", np.zeros(len(raw["det_frame"]), np.int8))
    out["det_rot"] = base_rot
    ex = dict(extra); ex["det_rot"] = np.ones(len(extra["det_frame"]), np.int8)
    for k in keys:
        out[k] = np.concatenate([out[k], ex[k].astype(out[k].dtype)])
    o = np.argsort(out["det_frame"], kind="stable")
    for k in keys:
        out[k] = out[k][o]
    return out


# ---------------------------------------------------------------------------- gymnast tracking
def _iou(a, b):
    """a [n,4], b [m,4] -> [n,m]"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _centre(b):
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


def build_tracklets(raw, max_gap=5, iou_min=0.2, dist_max=0.6):
    """Greedy frame-to-frame linking (Hungarian on IoU + centre distance). Returns a list of
    tracklets, each a list of detection indices in time order."""
    fr, box = raw["det_frame"], raw["det_box"]
    by_frame = [[] for _ in range(int(raw["n_frames"]))]
    for i, f in enumerate(fr):
        by_frame[f].append(i)
    tracks, last = [], []  # last: (frame, det index) of each track's end
    for f, idx in enumerate(by_frame):
        if not idx:
            continue
        idx = np.array(idx)
        act = [t for t in range(len(tracks)) if f - last[t][0] <= max_gap]
        used = set()
        if act:
            lb = box[[last[t][1] for t in act]]
            cb = box[idx]
            iou = _iou(lb, cb)
            h = np.maximum(lb[:, 3] - lb[:, 1], 1)[:, None]
            dist = np.linalg.norm(_centre(lb)[:, None] - _centre(cb)[None], axis=-1) / h
            ok = (iou > iou_min) | (dist < dist_max)
            cost = np.where(ok, (1 - iou) + 0.5 * dist, 1e6)
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e6:
                    tracks[act[r]].append(int(idx[c])); last[act[r]] = (f, int(idx[c])); used.add(c)
        for c in range(len(idx)):
            if c not in used:
                tracks.append([int(idx[c])]); last.append((f, int(idx[c])))
    return tracks


def score_tracklet(raw, tr, window):
    """Higher = more likely the gymnast: present through the vault, large, near the centre of a
    camera that follows the athlete, and with a body shape that changes a lot (runs, flips)."""
    fr = raw["det_frame"][tr]; box = raw["det_box"][tr]; kp = raw["det_kpt"][tr].astype(np.float32)
    W, H = float(raw["width"]), float(raw["height"])
    a, b = window
    inw = (fr >= a) & (fr < b)
    if not inw.any():
        return 0.0, {}
    cov = inw.sum() / max(1, b - a)
    h = np.median(box[inw, 3] - box[inw, 1]) / H
    c = _centre(box[inw])
    cen = np.mean(np.linalg.norm((c - [W / 2, H / 2]) / [W / 2, H / 2], axis=1)) / math.sqrt(2)
    bh = np.maximum(box[:, 3:4] - box[:, 1:2], 1)
    shape = (kp[:, :, :2] - _centre(box)[:, None]) / bh[:, None]   # pose in box-normalised coords
    act = float(np.mean(np.std(shape[inw], axis=0))) if inw.sum() > 2 else 0.0
    s = cov * h * (1.2 - cen) * (0.5 + 4 * act)
    return float(s), dict(cov=round(float(cov), 3), h=round(float(h), 3), cen=round(float(cen), 3),
                          act=round(act, 3))


def select_gymnast(raw, fr, min_rel=0.3, max_overlap=2):
    """v2: pick the gymnast separately inside each stage window.

    Broadcast vaults contain camera cuts (close-ups, a second angle) and flips break tracking, so
    one track rarely survives the whole vault (v1 chained from a single best track and lost the
    gymnast after the first break: landing coverage ~0). Here, inside each window
    (approach / flight / landing, or the whole vault if there are no stages):
      1. score every tracklet on that window only
      2. take the best, then greedily add tracklets that do not overlap it in time and score at
         least `min_rel` x the best (the gymnast's other segments after a cut or a flip)
    Returns per-frame detection index (-1 = none) and debug info."""
    if isinstance(fr, tuple):
        fr = {"whole": fr}
    tracks = build_tracklets(raw)
    T = int(raw["n_frames"])
    idx = np.full(T, -1, np.int64)
    if not tracks:
        return idx, dict(n_tracks=0)
    det_fr = raw["det_frame"]
    windows = {s: fr[s] for s in STAGES if s in fr} or {"whole": fr["whole"]}
    info = dict(n_tracks=len(tracks), per_stage={})
    for st, (a, z) in windows.items():
        sc = [score_tracklet(raw, t, (a, z)) for t in tracks]
        order = [int(i) for i in np.argsort([-s for s, _ in sc]) if sc[i][0] > 0]
        if not order:
            info["per_stage"][st] = dict(n_used=0)
            continue
        best = sc[order[0]][0]
        taken = np.zeros(T, bool)
        used = []
        for t in order:
            if sc[t][0] < min_rel * best:
                break
            f = det_fr[tracks[t]]
            f = f[(f >= a) & (f < z)]
            if taken[f].sum() > max_overlap:
                continue
            for d in tracks[t]:
                fd = det_fr[d]
                if a <= fd < z and not taken[fd]:
                    idx[fd] = d; taken[fd] = True
            used.append(t)
        info["per_stage"][st] = dict(n_used=len(used), best=round(best, 4), **sc[order[0]][1])
    return idx, info


def select_gymnast_v1(raw, window, max_chain_gap=10, chain_dist=1.5):
    """v1 (kept for reference): best tracklet over the whole vault, chained across short ID breaks.
    Loses the gymnast after camera cuts; superseded by select_gymnast (per stage)."""
    tracks = build_tracklets(raw)
    if not tracks:
        return np.full(int(raw["n_frames"]), -1, np.int64), dict(n_tracks=0)
    scores = [score_tracklet(raw, t, window) for t in tracks]
    order = np.argsort([-s for s, _ in scores])
    best = int(order[0])
    fr, box = raw["det_frame"], raw["det_box"]
    span = lambda t: (fr[tracks[t][0]], fr[tracks[t][-1]])
    chosen = [best]

    def extend(direction):
        while True:
            ends = [span(t) for t in chosen]
            lo, hi = min(e[0] for e in ends), max(e[1] for e in ends)
            edge_t = min(chosen, key=lambda t: span(t)[0]) if direction < 0 else \
                max(chosen, key=lambda t: span(t)[1])
            edge_det = tracks[edge_t][0] if direction < 0 else tracks[edge_t][-1]
            eb = box[edge_det]; eh = max(eb[3] - eb[1], 1)
            cands = []
            for t in range(len(tracks)):
                if t in chosen:
                    continue
                s, e = span(t)
                gap = (lo - e) if direction < 0 else (s - hi)
                if 0 < gap <= max_chain_gap:
                    d = tracks[t][-1] if direction < 0 else tracks[t][0]
                    dist = np.linalg.norm(_centre(box[d]) - _centre(eb)) / eh
                    if dist < chain_dist:
                        cands.append((dist, t))
            if not cands:
                return
            chosen.append(min(cands)[1])

    extend(+1); extend(-1)
    idx = np.full(int(raw["n_frames"]), -1, np.int64)
    for t in chosen:
        for d in tracks[t]:
            idx[fr[d]] = d
    top = [dict(score=round(scores[t][0], 4), frames=len(tracks[t]), **scores[t][1]) for t in order[:3]]
    return idx, dict(n_tracks=len(tracks), n_chained=len(chosen), top=top)


def gymnast_arrays(raw, idx):
    """-> kpts [T,17,3] (NaN where missing or conf < CONF_MIN), boxes [T,4] (NaN where missing)"""
    T = len(idx)
    k = np.full((T, 17, 3), np.nan, np.float32); b = np.full((T, 4), np.nan, np.float32)
    ok = idx >= 0
    k[ok] = raw["det_kpt"][idx[ok]].astype(np.float32); b[ok] = raw["det_box"][idx[ok]]
    low = k[..., 2] < CONF_MIN
    k[..., :2][low] = np.nan
    return k, b


# ---------------------------------------------------------------------------- geometry
def angle(a, b, c):
    """Angle ABC in degrees for arrays [..., 2]; 180 = straight."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(k):
    """k [T,17,3] -> dict of per-frame angles (both sides averaged where both visible)."""
    p = k[..., :2]
    side = lambda f: np.nanmean(np.stack([f("l"), f("r")]), axis=0)
    J = lambda s, n: p[:, KP[f"{s}_{n}"]]
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return dict(
            knee=side(lambda s: angle(J(s, "hip"), J(s, "knee"), J(s, "ank"))),
            hip=side(lambda s: angle(J(s, "sho"), J(s, "hip"), J(s, "knee"))),
            elbow=side(lambda s: angle(J(s, "sho"), J(s, "elb"), J(s, "wri"))),
        )


def body_scale(k):
    """Robust body size: median hip-to-ankle length over the clip (pixels)."""
    p = k[..., :2]
    leg = [np.linalg.norm(p[:, KP[f"{s}_hip"]] - p[:, KP[f"{s}_knee"]], axis=-1) +
           np.linalg.norm(p[:, KP[f"{s}_knee"]] - p[:, KP[f"{s}_ank"]], axis=-1) for s in "lr"]
    v = np.concatenate(leg)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


# ---------------------------------------------------------------------------- quality
def quality(k, b, a, z, H):
    """Quality of the gymnast skeleton inside frames [a, z)."""
    seg, bs = k[a:z], b[a:z]
    n = max(1, z - a)
    found = np.isfinite(bs[:, 0])
    out = dict(coverage=found.mean() if n else np.nan)
    conf = seg[:, LOWER, 2]
    out["lower_conf"] = float(np.nanmean(conf)) if found.any() else np.nan
    pts = seg[:, LOWER, :2]
    good = np.isfinite(pts).all(-1)
    out["lower_visible"] = float(good[found].mean()) if found.any() else np.nan
    h = np.nanmedian(bs[:, 3] - bs[:, 1]) if found.any() else np.nan
    step = np.linalg.norm(np.diff(pts, axis=0), axis=-1) / (h if h and h > 0 else np.nan)
    out["jitter"] = float(np.nanmedian(step)) if np.isfinite(step).any() else np.nan
    p = seg[..., :2]
    limbs = [np.linalg.norm(p[:, KP[f"{s}_{x}"]] - p[:, KP[f"{s}_{y}"]], axis=-1)
             for s in "lr" for x, y in [("hip", "knee"), ("knee", "ank")]]
    cvs = [np.nanstd(l) / np.nanmean(l) for l in limbs if np.isfinite(l).sum() > 3]
    out["limb_cv"] = float(np.median(cvs)) if cvs else np.nan
    out["box_h"] = float(h / H) if np.isfinite(h) else np.nan
    return out


# ---------------------------------------------------------------------------- first symbols
def symbols(k, b, fr):
    """Judge-style measurements per stage (first version; thresholds come later).
    fr: dict of stage frame ranges. Angles in degrees (180 = straight), distances in leg lengths."""
    ang = joint_angles(k)
    L = body_scale(k)
    p = k[..., :2]
    out = {"leg_len_px": L}
    q = lambda x, lo, hi, f: (f(x[lo:hi]) if np.isfinite(x[lo:hi]).any() else np.nan)
    if "flight" in fr:
        a, z = fr["flight"]
        out["flight_knee_min"] = q(ang["knee"], a, z, np.nanmin)     # bent knees
        out["flight_knee_mean"] = q(ang["knee"], a, z, np.nanmean)
        out["flight_hip_mean"] = q(ang["hip"], a, z, np.nanmean)     # body shape (pike/tuck/layout)
        sep = np.linalg.norm(p[:, KP["l_ank"]] - p[:, KP["r_ank"]], axis=-1) / L
        out["flight_leg_sep_max"] = q(sep, a, z, np.nanmax)          # legs apart
    if "landing" in fr:
        a, z = fr["landing"]
        out["landing_knee_min"] = q(ang["knee"], a, z, np.nanmin)    # squat depth
        out["landing_hip_min"] = q(ang["hip"], a, z, np.nanmin)      # trunk folding forward
        feet = (p[:, KP["l_ank"], 0] - p[:, KP["r_ank"], 0]) / L     # camera-motion invariant
        seg = feet[a:z]
        out["landing_feet_range"] = float(np.nanmax(seg) - np.nanmin(seg)) if np.isfinite(seg).sum() > 2 else np.nan
        hipx = np.nanmean(p[:, [KP["l_hip"], KP["r_hip"]], 0], axis=1)
        ankx = np.nanmean(p[:, [KP["l_ank"], KP["r_ank"]], 0], axis=1)
        rel = (hipx - ankx) / L                                       # balance: hips over feet?
        out["landing_balance_range"] = float(np.nanmax(rel[a:z]) - np.nanmin(rel[a:z])) \
            if np.isfinite(rel[a:z]).sum() > 2 else np.nan
    return out


# ---------------------------------------------------------------------------- drawing
def draw_skeleton(img, kp, color=(0, 255, 255), thick=2):
    import cv2
    for i, j in SKELETON:
        if np.isfinite(kp[[i, j], :2]).all():
            cv2.line(img, tuple(int(v) for v in kp[i, :2]), tuple(int(v) for v in kp[j, :2]), color, thick)
    for x, y, c in kp:
        if np.isfinite(x):
            cv2.circle(img, (int(x), int(y)), thick + 1, (0, 0, 255), -1)
    return img
