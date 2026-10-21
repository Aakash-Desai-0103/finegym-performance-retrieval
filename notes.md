# Working notes: FineGym performance-aware retrieval

Week of 18–21 Oct 2026 (Sun–Wed). Goal from mentor: work with FineGym, then test performance-aware (AQA) models on FineGym clips and report results.

Scope this week: **vault only** (short, single-element clips). Main target: **E (execution)**.

---

## Day 1 (Sun): FineGym-AQA annotations and the vault manifest

### Source

- FineGym itself has no quality scores. **FineGym-AQA** (released with FineGrade, CVPR Findings 2026, `github.com/yicong1/FineGym-AQA`) adds official result-book scores to FineGym clips: D (difficulty), E (execution), ND (neutral deductions), Total.
- File used: `json/finegym_annotation_info_v1.1_scores.json` (305 videos, 12,812 event clips, 5,564 with scores). FineGrade's model code is not released; only the annotations and result books.
- Licence: FineGym annotations are CC BY-NC 4.0 (research use only).

### Manifest (`scripts/build_vault_manifest.py` → `out/vault_manifest.csv`)

- 1,294 scored vault clips with D, E and Total → **1,247 after cleaning**
- 1,173 distinct vaults (routine groups), 125 source videos
- Split **by source video**: train 886 clips / 88 videos, val 155 / 19, test 206 / 18
- Median clip duration 7.36 s; 1,160 clips have 3 stage windows (≈ approach / flight / landing), 87 have none
- Scores: D 0–7.10 (median 5.40), E 0–9.90 (median 8.90), ND 0–2.0, Total 0–16.30 (median 14.33)

### Findings while building the manifest

- **ND is stored with mixed signs** in the source file. The script uses Total = D + E − |ND|, which holds for every clip.
- **47 duplicate clips** overlapped in time with another clip in the same video; removed, keeping the longest.
- **Replays** of the same vault share an identical score set; they get one `routine_group`, and the split is by video so a vault never appears in both train and test.
- **Vault D equals the difficulty value of the vault performed**, so D (and largely Total) mostly encodes *which* vault was done. **E is the real quality target.**
- Men's apparatus (event IDs 5–10) have scores but no element segments; women's VT/FX/BB/UB have both.

### Manual spot-checks (annotation vs. YouTube vs. official result PDF)

| Athlete | Video ID | Time | D / E / Total | Result |
|---|---|---|---|---|
| Jordyn Wieber | `0jqn1vxdhls` | 2:13–2:20 | 6.5 / 9.6 / 16.1 | Vault in range; scores match result sheet |
| Sofia Busato | `1rkcLEAMTpw` | 2:56–3:03 | 5.8 / 9.066 / 14.866 | Vault in range; scores match |
| Gabrielle Douglas | `26Y8BsNiiL8` | 0:43–0:52 | 6.5 / 9.5 / 16.0 | Vault in range; matches Finals score 16.000 |
| Lauren Hernandez | `2giTb7IpJDU` | 15:02–15:09 | 5.8 / 9.0 / 14.8 | Vault in range; matches Finals score 14.800 |

Conclusion: temporal annotations and scores are consistent with the videos and official result sheets; no issue found in 4 of 4 checks.

---

## Day 2 (Mon): clip download and availability

Script: `scripts/download_clips.py` (downloads only the annotated seconds, 480p, 0.5 s padding each side; checks each source video once; per-clip time limit) and `scripts/check_clips.py` (verifies files with ffprobe, writes `out/clips_available.csv` and `out/download_report.txt`).

- **Test split:** 16 of 18 source videos available; 2 unavailable (`GCyW1jPfWdg`, `iSQqsVFkQAA`). 177 of 206 test clips usable (86%); 6 test clips failed on an FFmpeg error.
- **Train split:** first a reproducible 300-clip sample spread across all train videos (260 usable); later **the full train split: 886 attempted, 784 usable (88%)**.
- Losses come from YouTube videos that are no longer available and from a Windows FFmpeg error on a few cuts (≈20 clips in the first round). Full lists in `out/download_report.txt`, `clips/video_status.csv`, `clips/download_log.csv`.
- The first downloader version hung indefinitely on some unavailable videos; fixed by checking each video's metadata first and putting a hard time limit on every clip.
- No corrupted or truncated files detected; manual spot-check of several clips from different videos showed the expected vaults.
- Usable data: **961 clips (784 train from 81 videos, 177 test from 16 videos)**. One val clip from the initial single-clip test is ignored.
- Clips and `clips_available.csv` are on Google Drive (`capstone/`), not in Git.

---

## Day 3 (Tue): VideoMAE features and first analysis

Notebook: `notebooks/day3_feature_extraction.ipynb` (Colab T4, ≈12 min for the 524 new clips; resumable). Analysis: `analysis/probe.py`, `analysis/probe2.py`.

- Frozen **VideoMAE-Base (Kinetics-400)**, 16 frames per window. Per clip: `whole` (768), `whole_seq` (8 × 768, temporal sequence), `stages` (3 × 768: approach / flight / landing).
- 961 clips, 0 errors; 886 with stage features.
- Model for score prediction: ridge regression on standardised features. 95% CIs from bootstrap resampling of **test videos**.

### 1. Whole-clip features capture the broadcast, not the vault

- A train clip's nearest neighbour comes from the **same source video 76.0%** of the time (chance 1.5%).
- Vault identity from the features is below a trivial baseline: nearest-neighbour accuracy 0.17–0.19 and a linear classifier 0.20–0.21, vs. 0.26 for always guessing the most common vault.
- Interpretation: generic Kinetics features encode the competition/broadcast (arena, camera, lighting), not fine-grained vault structure (twists, saltos). Consistent with FineGym and TF-CoVR's motivation.

### 2. Stage features are distinct

- A classifier separates approach / flight / landing with **98.6%** accuracy (chance 33.3%).
- The notebook's raw-cosine check (0.766 vs 0.769) is misleading: all features share a large common component. After centring, stages are clearly separable.

### 3. Execution (E) prediction

| Features (ridge) | E SRCC [95% CI] | E within-same-vault SRCC [95% CI] |
|---|---|---|
| Whole clip | 0.302 [0.07, 0.47] | 0.226 [0.01, 0.36] |
| Approach | 0.426 [0.23, 0.55] | 0.119 [−0.07, 0.31] |
| Flight | 0.480 [0.32, 0.60] | 0.332 [0.11, 0.51] |
| Landing | 0.577 [0.39, 0.69] | 0.476 [0.29, 0.58] |
| All 3 stages | 0.566 [0.42, 0.65] | 0.489 [0.30, 0.61] |
| Whole + stages | 0.505 [0.35, 0.61] | 0.450 [0.25, 0.58] |

- *Within-same-vault SRCC*: Spearman correlation computed separately inside each vault code with ≥5 test clips (9 codes, 154 clips), weighted by size.
- Permutation test (whole + stages, shuffled training labels, 200 runs): observed 0.450 vs. null 95th percentile 0.226, **p < 0.005** (0 of 200 shuffled runs reached it).
- Tripling training data (260 → 784 clips) raised all-stages within-vault SRCC from 0.385 to 0.489.
- **Correction (see Day 4):** part of this within-vault signal turned out to be competition level, not execution; the landing-only result is affected most.

### 4. Total and D are mostly "which vault"

- Using the **ground-truth** D as the only input gives Total SRCC 0.895 and E SRCC 0.436. These are *oracle references*, not video baselines: they show how much of each score is explained by vault identity.
- Best video-only Total SRCC: 0.643 (whole + stages, ridge).
- So Total-level SRCC mostly measures vault recognition; **E within the same vault is the meaningful performance measure.**

---

## Day 4 (Tue night): stronger models, FineGrade comparison, and the competition shortcut

Scripts: `analysis/day4_models.py`, `analysis/day4_same_video.py`. Results in `analysis/day4_results.csv`, `analysis/day4_within_video_all.csv`.

### 1. Model comparison (E)

| Features | Ridge: within vault | Neural network: within vault | Pairwise (same-vault pairs): within vault |
|---|---|---|---|
| Landing | 0.476 | 0.476 | 0.271 |
| All 3 stages | 0.489 | 0.451 | 0.398 |
| Whole + stages | 0.450 | 0.369 | 0.299 |

- The neural network (64 hidden units, 5 seeds) matches ridge within the CIs; the pairwise ranker (trained on 23,462 same-vault train pairs) is lower. **With frozen features, the model is not the bottleneck; the features are.**
- My reproduction on the laptop matched exactly for ridge and pairwise; the neural network differed by ≤ 0.006 (normal run-to-run variation across library versions).
- Lesson: the first neural-network run failed (SRCC ≈ 0) because the target was not standardised and early stopping used a random *clip-level* validation split, which leaks broadcast identity. **Validation must always be split by video.**

### 2. Comparison with FineGrade (vault, Total SRCC)

| Method | Vault Total SRCC |
|---|---|
| FineGrade, full rule-consistent model (their paper) | 0.746 |
| FineGrade "direct total" ablation | 0.645 |
| Ours: frozen VideoMAE, landing stage (ridge / neural network) | 0.674 / 0.686 [CI ≈ 0.50, 0.77] |

- Not like-for-like: FineGrade is trained end-to-end and does not state its split; our test is 16 competitions never seen in training.
- Fair statement: frozen stage-wise features land in FineGrade's range, and our CI includes their full result.
- FineGrade reports only Total, not E or same-vault execution ranking. That evaluation is where our work adds something.

### 3. Experiment: pairwise model trained only on pairs from different videos

- Hypothesis (from Claude): the pairwise ranker underperforms because most same-vault pairs come from the same competition.
- Result: only **1,303 of 23,462 pairs (≈6%)** were same-video pairs. Removing them barely changed results (landing 0.271 → 0.280, all stages 0.398 → 0.405, whole + stages 0.299 → 0.319).
- **Hypothesis rejected.** Same-video pairs are not the reason the pairwise ranker is lower.

### 4. Experiment: within the same vault *and* the same competition

New hypothesis: regression models learn the absolute E level of each competition (elite meets score higher, and the features recognise the broadcast), so within-vault ranking across competitions may partly reflect *which competition* a clip is from.

Test: Spearman inside each (vault code, source video) group, which holds the competition constant (23 groups, 110 clips, ≥3 per group), plus a diagnostic that replaces each prediction with its video's average (knows only the competition).

| Features | Model | Within vault | Within vault + video |
|---|---|---|---|
| Landing | ridge | 0.476 | **0.277** [0.17, 0.48] |
| Landing | neural network | 0.476 | 0.250 [0.09, 0.39] |
| Landing | pairwise | 0.271 | 0.163 [0.02, 0.35] |
| Landing | ridge, video average only | 0.235 | n/a |
| All 3 stages | ridge | 0.489 | **0.393** [0.22, 0.56] |
| All 3 stages | neural network | 0.451 | **0.432** [0.24, 0.59] |
| All 3 stages | pairwise | 0.398 | **0.363** [0.18, 0.53] |
| All 3 stages | ridge, video average only | 0.330 [0.06, 0.49] | n/a |

Findings:

1. **There is a competition-level shortcut.** Knowing only the competition gives a within-vault SRCC of 0.24–0.33.
2. **The landing-only result relied heavily on it**: 0.476 → 0.277 once the competition is fixed. The Day 3 claim "the landing carries the most execution signal" is withdrawn.
3. **All three stages together hold up**: 0.36–0.43 within the same vault and the same competition (the neural network barely drops; pairwise drops least relative to its starting point). Competition level alone reaches 0.33 within vault, while the full model reaches 0.49 and still 0.39–0.43 with the competition fixed, so the stage features carry execution information beyond the competition level.
4. Reliability: the stricter metric uses small groups, so CIs are wide; treat it as directional evidence.
5. Implication for retrieval: a ranking model trained on absolute scores can learn to "prefer clips from elite meets". Performance-aware retrieval must control for competition level (e.g., compare within competition, or use relative/pairwise objectives).

### Current headline result

> Stage-wise features (approach, flight, landing) from a frozen video model rank execution among athletes performing the same vault at the same competition with a Spearman correlation of ≈0.4. Part of the earlier same-vault result reflected competition-level differences, which are now controlled for.

---

## Day 5 (Wed): zero-shot VLM baseline

Notebook: `notebooks/day5_vlm_zero_shot.ipynb` (Colab T4). Evaluation: `analysis/day5_eval_vlm.py` → `analysis/day5_vlm_eval.csv`. Pairwise test: own script.

Question: can a general vision-language model, never trained on gymnastics, judge execution?

Setup: `Qwen/Qwen2.5-VL-3B-Instruct`, float16, greedy decoding. 36 test clips (12 each from vault codes 5.80, 5.40, 4.60; 6–12 competitions per code). Input: 8 frames in time order (3 approach, 2 flight, 3 landing) plus the difficulty value. Prompt: act as an FIG judge, one phrase per stage, then an E score.

### 1. Absolute scoring

| | VLM zero-shot | Ridge, 3 stages (ours), same 33 clips |
|---|---|---|
| Usable answers | 33 of 36 (92%); 3 were "!!!!" (known float16 glitch of this model on T4) | n/a |
| Distinct predictions | **1** (every clip: E = 8.2) | 28 |
| Prediction spread (std) | 0.000 | 0.221 (true E std 0.214) |
| Spearman overall | undefined (constant output) | 0.358 [0.09, 0.58] |
| Spearman within vault | undefined | 0.320 [−0.03, 0.60] |
| Mean absolute error | 0.711 | 0.182 |
| Bias | −0.711 (too harsh) | −0.049 |

- Stage phrases are boilerplate ("smooth and confident / controlled and fluid / stable and balanced") on every clip; no step, hop or form fault is ever mentioned, including on lower-E clips.
- Within vault + video values on this subset are not reliable (most groups have only 2–3 clips) and are not used.

### 2. Pairwise comparison ("same vault, which athlete executed better?")

- 24 same-vault pairs, 18 valid answers, 0 errors.
- The model chose **B (the second video) in 18 of 18** pairs: pure **position bias**. The reported accuracy 0.333 is just the share of pairs where B happened to be better; it is not a measure of judging.
- The "reason" is identical for every pair and is assembled from words in the prompt ("more stable landing with fewer steps and a better chest position"): it repeats the judging criteria, not what is in the video.
- Not done (optional): presenting each pair in both orders to measure order consistency.

### Findings

1. A small zero-shot VLM does not discriminate execution: one constant score, and a constant pairwise answer.
2. Its explanations sound plausible but are **not grounded** in the video. For explainable comparison, explanations must come from measured evidence (stages, pose, detected faults), which supports the neurosymbolic direction.
3. Our frozen stage-feature model, on the same clips, ranks execution clearly better (overall 0.36–0.41) with a 4× smaller error.
4. Caveat: one 3B model, 8 frames, zero-shot. Larger models, more frames or fine-tuning may do better; the claim is "a small zero-shot VLM fails", not "VLMs cannot do this".

---

## Week summary

| Question | Answer |
|---|---|
| Are there reliable performance labels for FineGym? | Yes: FineGym-AQA official scores; 4/4 manual checks matched result sheets |
| Usable data | 961 vault clips (784 train / 177 test), split by competition; ≈12% lost to unavailable YouTube videos |
| What is the right target? | E (execution), compared within the same vault; Total and D mostly encode which vault |
| Do generic video features capture the vault? | No: they capture the broadcast (76% same-video nearest neighbours; vault identity below chance baseline) |
| Do stage-wise features capture execution? | Yes, modestly: Spearman ≈0.4 within the same vault and competition (≈0.49 within vault before controlling for competition) |
| Does a bigger model help? | No: ridge ≈ neural network; the features are the limit |
| How does it compare with FineGrade? | Vault Total 0.67–0.69 (frozen features, unseen competitions) vs FineGrade 0.746 (end-to-end, split not stated); FineGrade does not report same-vault execution |
| Can a zero-shot VLM do it? | No: constant score (8.2), always picks the second video, ungrounded explanations |
| Main pitfall found | Competition-level shortcut: models can rank by "elite meet" instead of execution; retrieval must control for it |

### Next steps

- Write the one-page summary for the mentor (in progress).
- Next phase: athlete-centred cropping or pose features to reduce scene bias; explainable stage-wise comparison grounded in measured evidence (towards neurosymbolic, rule-based fault detection); retrieval that controls for competition level; other apparatus.
- Repo housekeeping: README and `requirements.txt` (numpy, pandas, scipy, scikit-learn, yt-dlp).
