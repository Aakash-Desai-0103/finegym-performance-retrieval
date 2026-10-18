"""Day 1: build a clean, scored vault clip list from FineGym-AQA.

Usage:
    python build_vault_manifest.py path/to/finegym_annotation_info_v1.1_scores.json out_dir/

Outputs (in out_dir):
    vault_manifest.csv   one row per usable vault clip, with scores and split
    vault_summary.txt    counts and score ranges for the write-up

Cleaning rules:
  * keep event == 1 (VT) with a dict score that has diff, execution, total
  * nd is stored with either sign in the source; total = diff + execution - |nd|
  * clips in the same video that overlap in time are duplicate annotations: keep the longest
  * clips in the same video with an identical score set are treated as the same vault
    (e.g. a replay) and share a `routine_group`, so they never straddle train/test
  * split by source video (competition), not by clip, to avoid athlete/routine leakage
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def main(scores_path: str, out_dir: str) -> None:
    data = json.load(open(scores_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clips = []
    for vid, events in data.items():
        for key, e in events.items():
            s = e.get("score")
            if e.get("event") != 1 or not isinstance(s, dict):
                continue
            if s.get("diff") is None or s.get("execution") is None or s.get("total") is None:
                continue
            start, end = e["timestamps"][0]
            segs = e.get("segments") or {}
            seg_key, seg = next(iter(segs.items()), (None, {}))
            clips.append({
                "video_id": vid,
                "event_key": key,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "athlete": e.get("name") or "",
                "bib": e.get("number") or "",
                "D": s["diff"],
                "E": s["execution"],
                "ND": abs(s.get("nd") or 0),
                "Total": s["total"],
                "vault_code": seg.get("code") or "",
                "segment_key": seg_key or "",
                # stage boundaries are relative to the clip start
                "stages": json.dumps(seg.get("timestamps") or []),
            })
    n_raw = len(clips)

    # 1. drop overlapping duplicates within a video (keep the longest)
    by_vid = defaultdict(list)
    for c in clips:
        by_vid[c["video_id"]].append(c)
    kept = []
    for vid, cs in by_vid.items():
        cs.sort(key=lambda c: -c["duration"])
        chosen = []
        for c in cs:
            if all(min(c["end"], k["end"]) - max(c["start"], k["start"]) <= 0 for k in chosen):
                chosen.append(c)
        kept.extend(chosen)
    n_overlap = n_raw - len(kept)

    # 2. group same-video, same-score clips (replays / repeated views)
    gid = {}
    for c in kept:
        sig = (c["video_id"], c["D"], c["E"], c["ND"], c["Total"], c["athlete"])
        gid.setdefault(sig, f"g{len(gid):04d}")
        c["routine_group"] = gid[sig]

    # 3. split by source video
    vids = sorted({c["video_id"] for c in kept})
    random.Random(SEED).shuffle(vids)
    n = len(vids)
    n_tr, n_va = round(n * SPLIT["train"]), round(n * SPLIT["val"])
    split_of = {v: "train" for v in vids[:n_tr]}
    split_of.update({v: "val" for v in vids[n_tr:n_tr + n_va]})
    split_of.update({v: "test" for v in vids[n_tr + n_va:]})
    for c in kept:
        c["split"] = split_of[c["video_id"]]

    kept.sort(key=lambda c: (c["video_id"], c["start"]))
    fields = ["video_id", "event_key", "start", "end", "duration", "athlete", "bib",
              "D", "E", "ND", "Total", "vault_code", "segment_key", "stages",
              "routine_group", "split"]
    with open(out / "vault_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)

    # summary
    sc = Counter(c["split"] for c in kept)
    sv = Counter(split_of.values())
    lines = [
        f"scored vault clips (D,E,Total present): {n_raw}",
        f"removed overlapping duplicates:          {n_overlap}",
        f"final clips:                             {len(kept)}",
        f"distinct vaults (routine groups):        {len(gid)}",
        f"source videos:                           {n}",
        "split (clips / videos): " + ", ".join(f"{k} {sc[k]}/{sv[k]}" for k in SPLIT),
        f"clips without element segment:           {sum(1 for c in kept if not c['segment_key'])}",
    ]
    for col in ["D", "E", "ND", "Total", "duration"]:
        vals = sorted(c[col] for c in kept)
        lines.append(f"{col:>8}: min {vals[0]:.2f}  median {vals[len(vals)//2]:.2f}  max {vals[-1]:.2f}")
    (out / "vault_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
