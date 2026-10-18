"""Download only the needed vault clips (not full broadcasts) from YouTube.

Requirements (install once):
    pip install yt-dlp pandas
    ffmpeg must be on PATH  (Windows: `winget install Gyan.FFmpeg`; Mac: `brew install ffmpeg`)

Usage:
    # today: test on ONE clip
    python download_clips.py vault_manifest.csv clips --limit 1
    # Monday: the test split, then part of train
    python download_clips.py vault_manifest.csv clips --split test
    python download_clips.py vault_manifest.csv clips --split train --limit 300

Writes clips/<video_id>__<event_key>.mp4 and appends every attempt to clips/download_log.csv
(ok / exists / failed + error), so unavailable YouTube videos are recorded for the write-up.
Re-running skips clips that already exist.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds of padding each side")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    if a.split:
        df = df[df.split == a.split]
    if a.limit:
        df = df.head(a.limit)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "download_log.csv"
    new_log = not log_path.exists()
    dead_videos = set()

    with open(log_path, "a", newline="") as lf:
        log = csv.writer(lf)
        if new_log:
            log.writerow(["video_id", "event_key", "status", "detail"])
        for i, r in enumerate(df.itertuples(), 1):
            name = f"{r.video_id}__{r.event_key}.mp4"
            dst = out / name
            if dst.exists():
                log.writerow([r.video_id, r.event_key, "exists", ""])
                continue
            if r.video_id in dead_videos:
                log.writerow([r.video_id, r.event_key, "failed", "video unavailable (earlier clip)"])
                continue
            s, e = max(0, r.start - a.pad), r.end + a.pad
            cmd = [
                sys.executable, "-m", "yt_dlp",
                f"https://www.youtube.com/watch?v={r.video_id}",
                "--download-sections", f"*{s:.2f}-{e:.2f}",
                "--force-keyframes-at-cuts",
                "-f", "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480]/b",
                "--merge-output-format", "mp4",
                "-o", str(dst),
                "--quiet", "--no-warnings",
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0 and dst.exists():
                log.writerow([r.video_id, r.event_key, "ok", f"{s:.2f}-{e:.2f}"])
                print(f"[{i}/{len(df)}] ok      {name}")
            else:
                err = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["unknown error"]
                if "unavailable" in err[0].lower() or "private" in err[0].lower():
                    dead_videos.add(r.video_id)
                log.writerow([r.video_id, r.event_key, "failed", err[0][:200]])
                print(f"[{i}/{len(df)}] FAILED  {name}: {err[0][:120]}")
            lf.flush()


if __name__ == "__main__":
    main()
