"""Day 6: did the TF-CoVR backbone see our test competitions during its training?

The AIM encoder was fine-tuned by the TF-CoVR authors on FineGym clips. If those clips came from the
same YouTube videos as our TEST clips, the backbone has seen our test competitions (not their E
scores, but their broadcast and athletes), which would flatter its results.

This script reads the TF-CoVR annotation CSVs (downloaded by the Day 6 notebook to
capstone/tfcovr_annotations/), pulls out the YouTube IDs of every FineGym clip they mention, and
compares them with our train/test videos.

Caveat: the CSVs are TF-CoVR's retrieval triplets. The list of clips used for stage-1
(classification) training is not released; the triplet CSVs are the best available proxy.

If any of our test videos overlap with TF-CoVR's training CSVs, pass --feats/--index pairs and the
script also writes a "clean" copy of each feature file with those test clips removed, so both
backbones can be compared on exactly the same unseen clips.

Usage:
    python analysis/day6_tfcovr_overlap.py --tfcovr-dir tfcovr_annotations \
        --feats features/videomae_k400_all.npz features/aim_tfcovr_all.npz \
        --index features/videomae_k400_index.csv features/aim_tfcovr_index.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tfcovr-dir", default="tfcovr_annotations")
ap.add_argument("--feats", nargs="*", default=[])
ap.add_argument("--index", nargs="*", default=[])
args = ap.parse_args()
assert len(args.feats) == len(args.index), "give one --index per --feats"

# FineGym clip names look like  yvZPHJu3kb0_E_001931_001939_A_0000_0008  (11-char YouTube ID first)
PAT = re.compile(r"([A-Za-z0-9_-]{11})_E_\d+_\d+")


def ids_in(series):
    out = set()
    for v in series.dropna().astype(str):
        out.update(PAT.findall(v))
    return out


csvs = sorted(Path(args.tfcovr_dir).rglob("*.csv"))
assert csvs, f"no CSV files in {args.tfcovr_dir}"
tf = {}
for f in csvs:
    d = pd.read_csv(f, low_memory=False)
    vids = set()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):  # text columns (object or string dtype)
            vids |= ids_in(d[c])
    tf[f.stem] = vids
    print(f"{f.name:40s} rows {len(d):7d} | FineGym YouTube videos {len(vids)}")

train_like = set().union(*[v for k, v in tf.items() if "train" in k.lower()]) if any(
    "train" in k.lower() for k in tf) else set()
if not train_like:
    print("warning: no CSV with 'train' in its name; treating ALL CSVs as possibly seen in training")
    train_like = set().union(*tf.values())
all_tf = set().union(*tf.values())

ours = pd.read_csv(args.index[0] if args.index else "features/videomae_k400_index.csv")
print()
for split in ["train", "test"]:
    v = set(ours.loc[ours.split == split, "video_id"])
    clips = ours[ours.split == split]
    in_tr = v & train_like
    print(f"our {split:5s}: {len(v):3d} videos / {len(clips)} clips | in TF-CoVR train CSVs: "
          f"{len(in_tr)} videos ({clips.video_id.isin(in_tr).sum()} clips) | in any TF-CoVR CSV: "
          f"{len(v & all_tf)} videos")

test_seen = sorted(set(ours.loc[ours.split == "test", "video_id"]) & train_like)
print("\nour TEST videos found in TF-CoVR training CSVs:", test_seen if test_seen else "none")
pd.DataFrame({"video_id": sorted(set(ours.video_id))}).assign(
    split=lambda d: d.video_id.map(ours.groupby("video_id").split.first()),
    in_tfcovr_train=lambda d: d.video_id.isin(train_like),
    in_tfcovr_any=lambda d: d.video_id.isin(all_tf),
).to_csv("tfcovr_overlap.csv", index=False)
print("written tfcovr_overlap.csv")

if test_seen and args.feats:
    print("\nwriting clean copies (overlapping TEST clips removed; train unchanged):")
    for fp, ip in zip(args.feats, args.index):
        F = dict(np.load(fp, allow_pickle=True))
        m = pd.read_csv(ip, dtype={"vault_code": str, "bib": str})
        assert (F["ids"] == m.clip_id.values).all(), f"{fp} and {ip} are not aligned"
        keep = ~((m.split == "test") & m.video_id.isin(test_seen)).values
        fo = Path(fp).with_name(Path(fp).stem + "_clean.npz")
        io = Path(ip).with_name(Path(ip).stem.replace("_index", "_clean_index") + ".csv")
        np.savez_compressed(fo, **{k: v[keep] for k, v in F.items()})
        m[keep].to_csv(io, index=False)
        print(f"  {fo}  +  {io}   ({keep.sum()} clips, test {(m[keep].split == 'test').sum()})")
