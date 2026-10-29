"""Week 2, Wed: key frames + LSTM execution scorer (mentor's suggestion), with attention as the
explanation of WHEN the model looked.

Sequence per vault: a short window (+-2 frames) around each of the 6 key frames from keyframes.py
(run_end, table, flight_peak, touchdown, deepest_squat, settled) = 30 time steps.
Features per time step (body-normalised, so camera position and zoom do not matter):
  12 joints (shoulders, elbows, wrists, hips, knees, ankles) relative to the hip centre, divided by
  leg length (24) + knee / hip / elbow angles (3) + "pose present" flag (1) + which key frame (6).
Model: 1-layer LSTM (hidden 32) -> attention over the 30 steps -> E. Small and regularised on
purpose (~800 training clips). Mirror augmentation (left-right flip). 5-seed ensemble. Fixed
number of epochs (Day 4 lesson: never early-stop on a random clip split).

Compared on the same test clips with:
  ridge_flat     ridge on the same 30 x 34 numbers, flattened (no sequence model)
  lstm           the LSTM
  lstm+symbols   LSTM with the Tuesday symbols added before the output layer (needs --symbols)

Usage (CPU is fine, ~5-15 min):
    pip install torch
    python pose/lstm_score.py --kf data/keyframes --symbols data/symbols/symbols.csv --out data/lstm
Writes lstm_report.txt, predictions.csv, attention.csv into --out.
"""
import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posekit as pk  # noqa: E402
from symbols import SYMBOLS, evaluate  # noqa: E402

KEYS = ["run_end", "table", "flight_peak", "touchdown", "deepest_squat", "settled"]
HALF = 2                                            # frames either side of each key frame
JOINTS = ["sho", "elb", "wri", "hip", "knee", "ank"]
MIRROR = {pk.KP[f"l_{j}"]: pk.KP[f"r_{j}"] for j in JOINTS}
MIRROR.update({v: k for k, v in MIRROR.items()})


def frame_features(k, t, L, mirror=False):
    if t < 0 or t >= len(k) or not np.isfinite(k[t, pk.LOWER, 0]).sum() >= 4:
        return np.zeros(28, np.float32)
    p = k[t, :, :2].copy()
    if mirror:
        p[:, 0] = -p[:, 0]
        p[list(MIRROR)] = p[list(MIRROR.values())]
    hip = np.nanmean(p[[pk.KP["l_hip"], pk.KP["r_hip"]]], axis=0)
    idx = [pk.KP[f"{s}_{j}"] for j in JOINTS for s in "lr"]
    rel = ((p[idx] - hip) / L).reshape(-1)
    kk = np.concatenate([p, np.ones((17, 1))], 1)[None]
    a = pk.joint_angles(kk)
    ang = np.array([a["knee"][0], a["hip"][0], a["elbow"][0]]) / 180.0
    f = np.concatenate([rel, ang, [1.0]]).astype(np.float32)
    return np.nan_to_num(f, nan=0.0)


def sequence(k, keys, mirror=False):
    L = pk.body_scale(k)
    if not np.isfinite(L) or L <= 0:
        L = 1.0
    seq = []
    for i, name in enumerate(KEYS):
        onehot = np.eye(len(KEYS), dtype=np.float32)[i]
        for dt in range(-HALF, HALF + 1):
            seq.append(np.concatenate([frame_features(k, keys[name] + dt, L, mirror), onehot]))
    return np.stack(seq)                                   # [30, 34]


class Scorer(nn.Module):
    def __init__(self, d_in, d_sym=0, hidden=32, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hidden + d_sym, 1))

    def forward(self, x, s=None):
        h, _ = self.lstm(self.inp(x))
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)          # attention over time steps
        z = (h * w.unsqueeze(-1)).sum(1)
        if s is not None:
            z = torch.cat([z, s], 1)
        return self.head(z).squeeze(-1), w


def train_lstm(Xtr, ytr, Xte, Str=None, Ste=None, seeds=5, epochs=80, lr=1e-3, wd=1e-3):
    preds, atts = [], []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Scorer(Xtr.shape[-1], 0 if Str is None else Str.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        X, y = torch.tensor(Xtr), torch.tensor(ytr)
        S = None if Str is None else torch.tensor(Str)
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 32):
                b = perm[i:i + 32]
                out, _ = m(X[b], None if S is None else S[b])
                loss = nn.functional.mse_loss(out, y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            o, w = m(torch.tensor(Xte), None if Ste is None else torch.tensor(Ste))
        preds.append(o.numpy()); atts.append(w.numpy())
    return np.mean(preds, 0), np.mean(atts, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", default="data/keyframes")
    ap.add_argument("--symbols", default=None, help="symbols.csv from symbols.py (optional)")
    ap.add_argument("--out", default="data/lstm")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    kf, out = Path(args.kf), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    K = pd.read_csv(kf / "keyframes.csv", dtype={"vault_code": str})
    K = K[K.has_stages & K.E.notna() & K.split.isin(["train", "test"])].reset_index(drop=True)
    K["vault_code"] = K.vault_code.fillna("NA")
    seqs, seqs_m = [], []
    for r in K.itertuples():
        z = np.load(kf / "pose_clean" / f"{r.clip_id}.npz")
        keys = json.loads(str(z["keys"]))
        seqs.append(sequence(z["kpts"], keys)); seqs_m.append(sequence(z["kpts"], keys, mirror=True))
    X, Xm = np.stack(seqs), np.stack(seqs_m)
    tr, te = (K.split == "train").values, (K.split == "test").values
    y = K.E.values.astype(np.float32)
    mu, sd = y[tr].mean(), y[tr].std()
    yz = (y - mu) / sd
    # standardise features with train statistics (time-step-wise)
    fm = X[tr].reshape(-1, X.shape[-1]).mean(0); fs = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    Z, Zm = (X - fm) / fs, (Xm - fm) / fs
    Ztr = np.concatenate([Z[tr], Zm[tr]]).astype(np.float32)          # mirror augmentation
    ytr = np.concatenate([yz[tr], yz[tr]]).astype(np.float32)

    pred, att = {}, {}
    rg = RidgeCV(alphas=np.logspace(0, 5, 21)).fit(Ztr.reshape(len(Ztr), -1), ytr)
    pred["ridge_flat"] = rg.predict(Z[te].reshape(te.sum(), -1)) * sd + mu
    p, a = train_lstm(Ztr, ytr, Z[te].astype(np.float32), seeds=args.seeds, epochs=args.epochs)
    pred["lstm"], att["lstm"] = p * sd + mu, a
    if args.symbols:
        S = pd.read_csv(args.symbols, dtype={"vault_code": str}).set_index("clip_id")
        cols = list(SYMBOLS)
        Sx = S.loc[K.clip_id, cols]
        Sx = Sx.fillna(Sx[tr].median())
        sc = StandardScaler().fit(Sx[tr])
        Ss = sc.transform(Sx).astype(np.float32)
        p, a = train_lstm(Ztr, ytr, Z[te].astype(np.float32), np.concatenate([Ss[tr], Ss[tr]]), Ss[te],
                          seeds=args.seeds, epochs=args.epochs)
        pred["lstm+symbols"], att["lstm+symbols"] = p * sd + mu, a

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"clips: {len(K)} (train {tr.sum()}, test {te.sum()}) | sequence {X.shape[1]} steps x "
              f"{X.shape[2]} features | train x2 with mirror augmentation\n")
        print("1. EXECUTION SCORE ON TEST (95% CI by test videos)")
        print("   reference, frozen VideoMAE (Days 3-4, full test set): within vault 0.476-0.489, "
              "within vault+competition 0.393\n")
        dt = K[te].reset_index(drop=True)
        rows = []
        for name, p in pred.items():
            (s, wv, wvv), lo, hi = evaluate(p, dt, args.boot)
            rows.append(dict(model=name, SRCC=f"{s:.3f} [{lo[0]:.2f},{hi[0]:.2f}]",
                             within_vault=f"{wv:.3f} [{lo[1]:.2f},{hi[1]:.2f}]",
                             within_vault_video=f"{wvv:.3f} [{lo[2]:.2f},{hi[2]:.2f}]",
                             pred_std=round(float(np.std(p)), 3)))
        print(pd.DataFrame(rows).to_string(index=False))
        print("   pred_std near 0 = the model predicts almost the same E for everyone (collapsed)\n")

        print("2. WHERE THE LSTM LOOKS (mean attention per key-frame window, test clips; uniform = "
              f"{1 / len(KEYS):.2f})")
        A = att["lstm"].reshape(len(dt), len(KEYS), 2 * HALF + 1).sum(2)
        att_df = pd.DataFrame(A, columns=KEYS)
        print(att_df.mean().round(3).to_string())
        lo_e = dt.E <= dt.E.quantile(0.25)
        print("\n   low-E clips (bottom quarter) vs the rest:")
        print(pd.DataFrame({"low E": att_df[lo_e.values].mean(), "others": att_df[~lo_e.values].mean()})
              .round(3).to_string())

    text = buf.getvalue()
    print(text)
    (out / "lstm_report.txt").write_text(text, encoding="utf-8")
    P = dt[["clip_id", "video_id", "vault_code", "E"]].copy()
    for name, p in pred.items():
        P[name] = np.round(p, 3)
    P.to_csv(out / "predictions.csv", index=False)
    att_df.insert(0, "clip_id", dt.clip_id.values)
    att_df.round(4).to_csv(out / "attention.csv", index=False)
    print("written", out / "lstm_report.txt", "|", out / "predictions.csv", "|", out / "attention.csv")


if __name__ == "__main__":
    main()
