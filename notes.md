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


## Day 3: VideoMAE Feature Extraction and Analysis

- Extracted frozen VideoMAE-Base (Kinetics-400) features for 961 usable clips: 784 train and 177 test.
- 886 of 961 clips contained valid stage features.
- Feature extraction completed successfully with 0 errors.
- Whole-clip features showed weak vault-identity performance: 1-NN accuracy was 0.192 compared with a 0.260 majority-class baseline.
- Strong source-video bias was observed: 1-NN retrieval within the training set selected the same source video 76.0% of the time, compared with 1.5% chance.
- Stage representations were highly separable: approach/flight/landing classification accuracy was 98.6% on the test clips.
- For execution-score prediction (E), whole-clip features achieved SRCC 0.302 overall and 0.226 within-vault.
- The landing stage achieved the strongest individual-stage result: SRCC 0.577 overall and 0.476 within-vault, with a 95% bootstrap CI of [0.29, 0.58] for within-vault SRCC.
- Stages-only features achieved SRCC 0.566 overall and 0.489 within-vault.
- Whole + stage features achieved SRCC 0.505 overall and 0.450 within-vault.
- The permutation test for whole + stages gave observed within-vault SRCC 0.450, compared with a null 95th percentile of 0.226 and p=0.000.
- The results indicate that stage-specific representations, particularly the landing stage, contain useful execution-related information beyond the whole-clip representation.
- Difficulty (D) alone remained a strong predictor of Total score, with SRCC 0.895.
- Feature files and intermediate extraction parts are stored on Google Drive under `capstone/features/`; they are not stored in Git.