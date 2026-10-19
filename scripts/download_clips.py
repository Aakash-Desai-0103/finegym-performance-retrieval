"""Download only the needed vault clips (not full broadcasts) from YouTube.

Requirements (install once):
    pip install yt-dlp pandas
    ffmpeg must be on PATH  (Windows: `winget install Gyan.FFmpeg`; Mac: `brew install ffmpeg`)

Usage:
    # one clip, to test
    python scripts/download_clips.py out/vault_manifest.csv clips --limit 1
    # whole test split
    python scripts/download_clips.py out/vault_manifest.csv clips --split test
    # 300 train clips sampled evenly across train videos (reproducible)
    python scripts/download_clips.py out/vault_manifest.csv clips --split train --sample 300

How it avoids getting stuck:
  1. Before downloading, each source video is checked ONCE (metadata only, 60 s limit).
     Unavailable / private / hanging videos are logged and all their clips are skipped.
  2. Every clip download has a hard time limit (--clip-timeout, default 180 s). On timeout the
     whole yt-dlp + ffmpeg process tree is killed, partial files are deleted, and it moves on.
  3. After 2 timeouts on the same video, the rest of that video's clips are skipped.

Options:
    --workers N            parallel clip downloads (default 3; use 1 if YouTube starts refusing)
    --cookies-browser X    use browser cookies (chrome / edge / firefox) if YouTube says
                           "sign in to confirm you're not a bot"
    --clip-timeout S       seconds allowed per clip (default 180)
    --recheck              re-check videos that an earlier run marked unavailable

Writes clips/<video_id>__<event_key>.mp4, appends every attempt to clips/download_log.csv and
keeps clips/video_status.csv (one line per source video: ok / unavailable / timeout).
Re-running skips clips that already exist, so it is safe to stop (Ctrl+C) and restart.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SEED = 42
PROBE_TIMEOUT = 60
lock = threading.Lock()
bad_videos = {}          # video_id -> reason
timeouts = {}            # video_id -> count
YTDLP_NET = ["--socket-timeout", "20", "--retries", "3", "--fragment-retries", "3",
             "--extractor-retries", "2", "--no-playlist"]


def run_with_timeout(cmd, timeout):
    """Run cmd; on timeout kill it AND its children (ffmpeg). Returns (returncode, output, timed_out)."""
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", **kw)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out or "", False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGKILL)
        try:
            p.communicate(timeout=10)
        except Exception:
            pass
        return -1, "", True


def last_error(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    errs = [l for l in lines if "ERROR" in l]
    return (errs or lines or ["unknown error"])[-1]


def is_unavailable(msg):
    low = msg.lower()
    return any(w in low for w in ("unavailable", "private", "removed", "terminated", "copyright",
                                  "does not exist", "account associated", "not available"))


def sample_across_videos(df, n):
    """Round-robin over videos so every source video contributes before any gives a second clip."""
    groups = [g.sample(frac=1, random_state=SEED) for _, g in df.groupby("video_id")]
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g.iloc[i])
        i += 1
    return pd.DataFrame(picked)


def cleanup_partials(out, stem):
    for f in out.glob(stem + "*"):
        if f.suffix != ".mp4" or f.stat().st_size == 0:
            try:
                f.unlink()
            except OSError:
                pass
    f = out / (stem + ".mp4")
    if f.exists() and f.stat().st_size == 0:
        f.unlink()


def probe_video(vid, cookies):
    cmd = [sys.executable, "-m", "yt_dlp", f"https://www.youtube.com/watch?v={vid}",
           "--simulate", "--quiet", "--no-warnings", "--print", "duration"] + YTDLP_NET
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    rc, out, to = run_with_timeout(cmd, PROBE_TIMEOUT)
    if to:
        return "timeout", f"metadata check took > {PROBE_TIMEOUT}s"
    if rc == 0:
        return "ok", ""
    err = last_error(out)
    return ("unavailable" if is_unavailable(err) else "error"), err[:200]


def download_one(r, out, a, write, total, idx):
    stem = f"{r.video_id}__{r.event_key}"
    dst = out / f"{stem}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        write(r, "exists")
        return
    with lock:
        reason = bad_videos.get(r.video_id)
    if reason:
        write(r, "skipped", reason)
        return
    s, e = max(0.0, r.start - a.pad), r.end + a.pad
    cmd = [sys.executable, "-m", "yt_dlp", f"https://www.youtube.com/watch?v={r.video_id}",
           "--download-sections", f"*{s:.2f}-{e:.2f}", "--force-keyframes-at-cuts",
           "-f", "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480]/b",
           "--merge-output-format", "mp4", "-o", str(dst),
           "--quiet", "--no-warnings", "--no-progress"] + YTDLP_NET
    if a.cookies_browser:
        cmd += ["--cookies-from-browser", a.cookies_browser]
    rc, text, timed_out = run_with_timeout(cmd, a.clip_timeout)
    if not timed_out and rc == 0 and dst.exists() and dst.stat().st_size > 0:
        write(r, "ok", f"{s:.2f}-{e:.2f}")
        print(f"[{idx}/{total}] ok       {stem}", flush=True)
        return
    cleanup_partials(out, stem)
    if timed_out:
        with lock:
            timeouts[r.video_id] = timeouts.get(r.video_id, 0) + 1
            if timeouts[r.video_id] >= 2:
                bad_videos[r.video_id] = "2 clip timeouts on this video"
        write(r, "timeout", f"> {a.clip_timeout}s")
        print(f"[{idx}/{total}] TIMEOUT  {stem}", flush=True)
        return
    err = last_error(text)
    if is_unavailable(err):
        with lock:
            bad_videos[r.video_id] = err[:200]
    write(r, "failed", err[:200])
    print(f"[{idx}/{total}] FAILED   {stem}: {err[:110]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, help="first N rows")
    ap.add_argument("--sample", type=int, help="N rows spread evenly across videos")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds of padding each side")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cookies-browser", default="")
    ap.add_argument("--clip-timeout", type=int, default=180)
    ap.add_argument("--recheck", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    if a.split:
        df = df[df.split == a.split]
    if a.sample:
        df = sample_across_videos(df, a.sample)
    elif a.limit:
        df = df.head(a.limit)
    df = df.sort_values(["video_id", "start"])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "download_log.csv"
    if log_path.exists():  # logs from the first version had no `split` column
        with open(log_path) as f:
            if "split" not in f.readline():
                log_path.rename(out / "download_log_old.csv")

    # ---- step 1: check each source video once -------------------------------------------
    status_path = out / "video_status.csv"
    status = {}
    if status_path.exists():
        for row in csv.DictReader(open(status_path, newline="")):
            status[row["video_id"]] = (row["status"], row["detail"])
    vids = list(dict.fromkeys(df.video_id))
    print(f"Checking {len(vids)} source videos (metadata only)...", flush=True)
    for i, v in enumerate(vids, 1):
        known = status.get(v)
        if known and (known[0] == "ok" or not a.recheck):
            st, detail = known
        else:
            st, detail = probe_video(v, a.cookies_browser)
            status[v] = (st, detail)
            with open(status_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video_id", "status", "detail"])
                w.writerows([k, s, d] for k, (s, d) in status.items())
        print(f"  [{i}/{len(vids)}] {v}: {st}{'  ' + detail[:90] if detail else ''}", flush=True)
        if st != "ok":
            bad_videos[v] = f"video {st}: {detail}"[:200]
    usable = df[~df.video_id.isin(bad_videos)]
    print(f"\n{len(usable)} of {len(df)} clips are on available videos "
          f"({len(vids) - len(bad_videos)}/{len(vids)} videos ok). Downloading...\n", flush=True)

    # ---- step 2: download clips ---------------------------------------------------------
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as lf:
        log = csv.writer(lf)
        if new_log:
            log.writerow(["video_id", "event_key", "split", "status", "detail"])

        def write(r, st, detail=""):
            with lock:
                log.writerow([r.video_id, r.event_key, r.split, st, detail])
                lf.flush()

        for r in df[df.video_id.isin(bad_videos)].itertuples():
            write(r, "skipped", bad_videos[r.video_id])
        rows = list(usable.itertuples())
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(rows, 1):
                ex.submit(download_one, r, out, a, write, len(rows), i)

    got = sum((out / f"{r.video_id}__{r.event_key}.mp4").exists() for r in df.itertuples())
    print(f"\nDone. {got}/{len(df)} clips present in {out}. "
          f"Videos unavailable/skipped: {len(bad_videos)}. See {log_path.name}, {status_path.name}.")


if __name__ == "__main__":
    main()
