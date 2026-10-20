## Spot-check 1 — Jordyn Wieber

- Video ID: `0jqn1vxdhls`
- Time: 2:13–2:20
- Athlete: Jordyn Wieber
- D: 6.5
- E: 9.6
- Total: 16.1
- Annotation check: The annotated time range correctly corresponds to the vault performance.
- Result sheet check: The D, E and Total scores match the official result sheet.
- Observation: The clip and annotation appear consistent; no obvious issue was found.

## Spot-check 2 — Sofia Busato

- Video ID: `1rkcLEAMTpw`
- Time: 2:56–3:03
- Athlete: Sofia Busato
- D: 5.8
- E: 9.066
- Total: 14.866
- Annotation check: The annotated time range contains a vault performance.
- Result sheet check: The D, E and Total scores match the corresponding result entry.
- Observation: The video segment and annotation appear consistent; no obvious issue was found.

## Spot-check 3 — Gabrielle Douglas

- Video ID: `26Y8BsNiiL8`
- Time: 0:43–0:52
- Athlete: Gabrielle Douglas
- D: 6.5
- E: 9.5
- Total: 16.0
- Annotation check: The annotated time range contains a vault performance.
- Result sheet check: The D, E and Total scores match the corresponding result entry.
- Observation: The video segment and annotation appear consistent; no obvious issue was found.

## Spot-check 4 — Lauren Hernandez

- Video ID: `2giTb7IpJDU`
- Time: 15:02–15:09
- Athlete: Lauren Hernandez
- D: 5.8
- E: 9.0
- Total: 14.8
- Annotation check: The annotated time range contains a vault performance.
- Result sheet check: The D, E and Total scores match the corresponding Finals result entry.
- Observation: The video segment and annotation appear consistent; no obvious issue was found.


Day 2: Clip Download and Availability

- Checked all 18 source videos in the test split.
- 16 of 18 test source videos were available on YouTube.
- 2 test source videos were unavailable: GCyW1jPfWdg and iSQqsVFkQAA.
- 206 test clips were present in the manifest.
- 177 test clips were successfully downloaded and usable (86%).
- 6 test clip-level downloads failed due to FFmpeg errors.
- Downloaded a reproducible sample of 300 train clips, sampled evenly across the available train videos.
- 260 of 300 train clips were successfully downloaded and usable (87%).
- 9 source videos were unavailable across the checked train/test sources.
- 20 clip-level downloads failed due to FFmpeg errors.
- No videos hung or errored during the source-video availability check.
- No problematic/corrupted files were detected by the clip checker.
- Spot-checked downloaded clips from different source videos; the clips contained the expected vault actions and showed no obvious black, frozen, or corrupted footage.
- Generated `out/clips_available.csv` containing the usable train and test clips.
- Generated `out/download_report.txt` containing the final download and availability results.
- Copied the usable clips and latest `clips_available.csv` to Google Drive for the next-stage Colab work.


## Day 3: VideoMAE Feature Extraction and Initial Analysis

- Extracted frozen VideoMAE-Base (Kinetics-400) features on Google Colab using a T4 GPU.
- Processed 437 usable clips: 260 train and 177 test.
- 408 of 437 clips had stage features available.
- Feature extraction completed successfully with 0 errors.
- Generated whole-clip features `(437, 768)`, temporal sequence features `(437, 8, 768)`, and stage features `(437, 3, 768)`.
- Initial analysis showed that whole-clip features were strongly influenced by the source broadcast/scene rather than fine-grained vault identity.
- Nearest-neighbour vault-code accuracy was below the majority-class baseline.
- Stage features were clearly distinguishable: approach vs flight vs landing classification achieved 97.8% test accuracy.
- Landing-stage features gave the strongest execution-score prediction, with E SRCC = 0.580 overall and 0.448 within the same vault.
- The within-vault result was supported by a permutation test (p = 0.005).
- Total score was strongly associated with difficulty (D), confirming that E is the more meaningful target for performance comparison.
- Feature extraction outputs were saved to Google Drive under `capstone/features/`.
- Intermediate feature files were retained to allow extraction to resume if needed.
- Day 3 extraction completed in approximately 9.2 minutes with no errors.