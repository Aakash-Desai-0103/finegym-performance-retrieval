"""Check downloaded clips and write the list of usable clips for Colab.

Usage:
    python scripts/check_clips.py out/vault_manifest.csv clips out

For every manifest row whose clip file exists, it reads the real duration with ffprobe and flags
clips that are empty, unreadable, or much shorter than expected. Outputs:

    out/clips_available.csv   manifest rows that have a good clip, plus:
                                clip_file      file name inside the clips folder
                                clip_duration  real duration (s)
                                stage_offset   seconds to ADD to the `stages` times to find them
                                               inside the downloaded clip (= padding actually used)
    out/download_report.txt   per-split counts, unavailable videos, problem clips (for notes.md)
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

PAD = 0.5


def probe_duration(path: Path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("clips_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest, dtype={"bib": str, "vault_code": str})
    clips = Path(a.clips_dir)
    out = Path(a.out_dir)
    log_path = clips / "download_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(
        columns=["video_id", "event_key", "split", "status", "detail"])

    attempted = set(zip(log.video_id, log.event_key))
    good, problems = [], []
    for r in df.itertuples():
        f = clips / f"{r.video_id}__{r.event_key}.mp4"
        if not f.exists():
            continue
        dur = probe_duration(f) if f.stat().st_size > 0 else None
        expected = (r.end - r.start) + 2 * PAD
        if dur is None:
            problems.append((f.name, "unreadable or empty"))
            continue
        if dur < 0.7 * expected:
            problems.append((f.name, f"too short: {dur:.1f}s vs ~{expected:.1f}s expected"))
            continue
        row = r._asdict()
        row.pop("Index")
        row.update(clip_file=f.name, clip_duration=round(dur, 2),
                   stage_offset=round(r.start - max(0.0, r.start - PAD), 2))
        good.append(row)

    avail = pd.DataFrame(good)
    avail.to_csv(out / "clips_available.csv", index=False)

    # report
    last = log.drop_duplicates(["video_id", "event_key"], keep="last")
    have = set(zip(avail.video_id, avail.event_key)) if len(avail) else set()
    failed = last[last.status.isin(["failed", "timeout"])
                  & ~pd.Series(list(zip(last.video_id, last.event_key)), index=last.index).isin(have)]
    status_path = clips / "video_status.csv"
    vstat = pd.read_csv(status_path).fillna("") if status_path.exists() else pd.DataFrame(
        columns=["video_id", "status", "detail"])
    dead = sorted(vstat[vstat.status == "unavailable"].video_id)
    hung = sorted(vstat[vstat.status.isin(["timeout", "error"])].video_id)
    lines = ["DOWNLOAD REPORT", ""]
    for split in ["train", "val", "test"]:
        want = df[df.split == split]
        tried = sum((v, k) in attempted for v, k in zip(want.video_id, want.event_key))
        got = (avail.split == split).sum() if len(avail) else 0
        if tried:
            lines.append(f"{split:>5}: attempted {tried:4d}  usable {got:4d}  "
                         f"({100 * got / tried:.0f}%)  of {len(want)} in manifest")
    lines += ["", f"usable clips total: {len(avail)}"]
    if len(avail):
        lines.append(f"usable source videos: {avail.video_id.nunique()}")
        lines.append(f"distinct vaults (routine groups): {avail.routine_group.nunique()}")
        for split in ["train", "test"]:
            s = avail[avail.split == split]
            if len(s):
                lines.append(f"{split} E: min {s.E.min():.2f} median {s.E.median():.2f} "
                             f"max {s.E.max():.2f} | distinct vault codes {s.vault_code.nunique()}")
    lines += ["", f"source videos checked: {len(vstat)}",
              f"unavailable YouTube videos ({len(dead)}): " + (", ".join(dead) or "none"),
              f"videos that hung or errored on check ({len(hung)}): " + (", ".join(hung) or "none")]
    other = failed[~failed.video_id.isin(dead + hung)]
    lines.append(f"clip-level failures/timeouts on available videos: {len(other)}")
    for d in other.detail.value_counts().head(5).items():
        lines.append(f"   {d[1]:3d} x {d[0][:110]}")
    lines.append(f"problem files (excluded): {len(problems)}")
    lines += [f"   {n}: {why}" for n, why in problems[:20]]
    (out / "download_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
