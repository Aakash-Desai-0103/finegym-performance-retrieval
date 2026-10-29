# Action Performance-Aware Fine-Grained Composite Video Retrieval

Measuring **how well** a gymnast performed, not just what they performed, on FineGym vault videos, and explaining the score in judging terms. This is the performance-measurement foundation for performance-aware retrieval and comparison.

**Target:** the judges' execution score **E** from FineGym-AQA, compared among gymnasts performing the **same vault** (Total and difficulty mostly encode which vault was done).
**Evaluation:** Spearman rank correlation (SRCC) overall, within the same vault, and within the same vault **and** competition (removes the competition-level shortcut), with 95% bootstrap intervals over test competitions. Train/test split by competition: 784 / 177 clips.

## Results so far (vault, test set)

| Method | Overall | Within vault | Within vault + competition | Explains the score? |
|---|---|---|---|---|
| Frozen VideoMAE, 3 stage features + ridge (Week 1) | 0.57 | **0.49** | **0.39** | No |
| Pose symbols + fixed deductions (no training) | 0.11 | 0.25 | 0.23 | Yes: deduction list |
| Pose symbols + fitted deductions | 0.18 | 0.31 | 0.25 | Yes: deduction list |
| Ridge on 10 pose symbols | 0.16 | 0.24 | 0.27 | Partly |
| LSTM over 6 key frames | 0.41 | 0.22 | 0.13 | When (attention) |
| LSTM + symbols | 0.46 | 0.24 | 0.12 | When and which |

Pose models use the 170 test clips with stage annotations; VideoMAE uses 177. Intervals are about ±0.15.

**Explainability check (blind, 20 test clips, one non-expert reviewer):** the system caught 5/5 deep-squat and 5/5 trunk-forward landings the reviewer saw and flagged no false falls; its fault count follows the judges' E (Spearman −0.69) about as well as the reviewer's (−0.74). In-air and arm-swing flags are not yet reliable.

## Key findings

1. **Total is mostly "which vault"**: difficulty alone predicts Total at 0.90, so E within the same vault is the real performance target.
2. **Generic video features encode the broadcast**: 76% of nearest neighbours come from the same competition video.
3. **Competition shortcut**: models can rank by recognising elite competitions; evaluating within the same vault and competition removes it.
4. **The landing carries the execution signal**: shown independently by VideoMAE stage features, the fitted deduction weights and the LSTM's attention (~63% on landing frames).
5. **FineGym's third vault segment is table push-off + second flight + landing**, not just the landing.
6. **A zero-shot VLM (Qwen2.5-VL-3B) cannot judge execution**: constant score, position bias, ungrounded explanations.
7. **TF-CoVR's AIM backbone was trained on 12 of our 16 test competitions**; FineGym evaluations built on it need an overlap check.

## Pipeline

```text
FineGym-AQA scores ──> vault manifest ──> clip download ──> 961 clips (split by competition)
                                                     │
          ┌──────────────────────────────────────────┴───────────────────────────────┐
  Week 1: frozen VideoMAE (whole + 3 stages) ──> ridge / MLP / pairwise ──> E          │
  Week 2: pose (YOLO26-L, gymnast picked per stage, rotated pass for upside-down frames)
          ──> 6 key frames ──> 10 judge-style symbols ──> rules (deduction list) / LSTM ──> E + explanation
                                                     │
  Checks: within-vault / within-competition SRCC, leakage checks, key-frame review, blind fault check
```

## Repository structure

```text
├── scripts/      data pipeline: manifest, clip download, clip checks
├── notebooks/    Colab notebooks: VideoMAE features, VLM zero-shot, AIM features, pose pilot + full run
├── analysis/     Week 1 analysis: feature probes, score models, competition shortcut, VLM, TF-CoVR overlap
├── pose/         Week 2: pose utilities, pose quality, key frames, symbols + rules, LSTM, fault check
├── results/      Week 2 result files and reports (Week 1 results are in analysis/)
└── out/          vault manifest and clip availability
```

Large data (clips, features, raw pose detections) lives on Google Drive and in the git-ignored `data/`, `clips/` and `features/` folders.

## Running it

```bash
pip install -r requirements.txt
# 1) data
python scripts/build_vault_manifest.py ...        # see the script header
python scripts/download_clips.py out/vault_manifest.csv clips
python scripts/check_clips.py out/vault_manifest.csv clips out
# 2) pose (Colab GPU): notebooks/week2_pose_pilot.ipynb with MODE='full', MODELS=['yolo26l-pose.pt']
#    download pose/yolo26l_960/gymnast_v3/ into data/pose/yolo26l_960/gymnast_v3/
# 3) key frames, symbols, scoring (laptop CPU)
python pose/keyframes.py --gym-dir data/pose/yolo26l_960/gymnast_v3 --index features/videomae_k400_index.csv --clips clips --out data/keyframes
python pose/symbols.py --kf data/keyframes --out data/symbols
python pose/lstm_score.py --kf data/keyframes --symbols data/symbols/symbols.csv --out data/lstm
# 4) explanation check
python pose/fault_check.py make     # fill in data/fault_check/fault_check.csv by watching the clips
python pose/fault_check.py score
```

## Limitations and next steps

- Touchdown detection is still early in some clips; in-air pose is noisy, so second-flight symbols are weak.
- One reviewer for the fault check; vault only.
- Next: learned touchdown detector, better in-air pose, a hybrid (learned score + symbolic explanation), a second apparatus, then performance-aware retrieval and comparison.

---

**Capstone Project**

**Project Title:** Action Performance-Aware Fine-Grained Composite Video Retrieval
**Project ID:** PW_25_GT_02
**Project Guide:** Dr. Gayathri T
**Project Team:**
- Aakash Desai (PES1UG24CS006)
- Aariz Qureshi (PES1UG24CS008)
- Amogh Sharma (PES1UG24CS053)
- Anagha B Poojari (PES1UG24CS056)
