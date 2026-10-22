# Action Performance-Aware Fine-Grained Composite Video Retrieval

This project focuses on the analysis and retrieval of gymnastics vault performances using deep spatio-temporal features. The primary objective is to develop a system that can distinguish between athletes performing the same vault by analyzing execution quality (E-score) across specific phases: the approach, the flight, and the landing.

## 🎯 Objective

We aim to determine if fine-grained, stage-specific representations can better predict athletic performance and execution quality than global, whole-clip features. Our research specifically targets the "competition-level shortcut," where models might mistakenly correlate broadcast quality or event prestige with athlete execution.

## 🛠️ Methodology

The project follows a rigorous pipeline from raw data acquisition to advanced model probing:

### 1. Data Acquisition & Curation
Using **FineGym-AQA**, we integrated official result-book scores (Difficulty, Execution, and Total) into the dataset.
- **Manifest Building**: Curated 1,247 scored vault clips, split by source video to prevent data leakage.
- **Pipeline**: Implemented a robust download and trim system using `yt-dlp` and `FFmpeg`, successfully processing 961 usable clips (784 train / 177 test).
- **Validation**: Performed manual spot-checks against official FIG result PDFs to ensure annotation consistency.

### 2. Feature Extraction & Modeling
We utilize a frozen **VideoMAE-Base** (Kinetics-400) to extract temporal features:
- **Global vs. Stage**: We compare whole-clip embeddings against separate embeddings for the approach, flight, and landing stages.
- **Regression Models**: Employed Ridge Regression and Neural Networks to predict execution scores.
- **Evaluation**: Used Spearman's Rank Correlation Coefficient (SRCC) and permutation tests to validate results.

### 3. VLM Zero-Shot Baseline
To test the capabilities of general-purpose AI, we evaluated **Qwen2.5-VL-3B-Instruct** in a zero-shot setting. We prompted the model to act as an FIG judge and provide both a score and a qualitative explanation for the performance.

## 📈 Key Findings

### Performance-Aware Retrieval
- **Stage-Wise Superiority**: Stage-specific features significantly outperform whole-clip features for execution prediction.
- **The Competition Shortcut**: We discovered that models often learn the absolute E-level of a competition (e.g., elite meets score higher). When controlling for the competition, stage-wise features still maintain a Spearman correlation of **≈0.4** within the same vault.
- **Feature Bottleneck**: Increasing model complexity (e.g., switching from Ridge to Neural Networks) did not significantly improve results, indicating that the quality of frozen features is the primary limiting factor.

### VLM Evaluation
- **Failure of Zero-Shot Judging**: The small VLM failed to discriminate execution, providing a constant score (8.2) for all clips and exhibiting strong position bias in pairwise comparisons.
- **Ungrounded Explanations**: VLM-generated justifications were "hallucinated" boilerplate text and not grounded in actual video evidence.

## 📂 Project Structure

```text
├── analysis/          # Probing scripts, model comparisons, and VLM evaluations
├── notebooks/         # Feature extraction and VLM zero-shot experiments
├── scripts/           # Data pipeline: manifest building, clip downloading, and verification
├── out/               # Generated manifests, availability reports, and results
└── notes.md           # Detailed development log and experimental results
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- FFmpeg (required for clip extraction)
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `yt-dlp`

### Running the Pipeline
1. **Download Clips**: Use `scripts/download_clips.py` to fetch videos from YouTube.
2. **Verify Clips**: Run `scripts/check_clips.py` to ensure data integrity.
3. **Extract Features**: Follow `notebooks/day3_feature_extraction.ipynb` to generate VideoMAE embeddings.
4. **Analyze**: Use the scripts in `analysis/` to correlate features with performance scores.

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
