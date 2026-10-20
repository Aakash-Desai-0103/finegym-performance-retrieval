# FineGym Performance Retrieval

This project explores the use of deep spatio-temporal features to analyze and retrieve gymnastics vault performances. The primary goal is to understand how different phases of a vault—the approach, the flight, and the landing—contribute to the final scoring, specifically focusing on the Execution (E) score.

## 🎯 Objective

We aim to determine if fine-grained, stage-specific representations can better predict athletic performance and execution quality than global, whole-clip features, which are often biased by the background or the identity of the source video.

## 🛠️ Methodology

The pipeline is divided into three main stages:

### 1. Data Acquisition & Curation
Using the FineGym dataset as a foundation, we implemented a custom pipeline to:
- Match annotated vault performances to their corresponding YouTube source videos.
- Download and trim clips using `FFmpeg` to isolate specific performances.
- Perform manual spot-checks on annotations to ensure the timing and scores (D and E) align with official result sheets.

### 2. Feature Extraction
We utilize a frozen **VideoMAE-Base** model (pretrained on Kinetics-400) to extract high-dimensional temporal features. 
- **Global Features**: Extracted from the entire clip.
- **Stage-Specific Features**: Extracted separately for the approach, flight, and landing phases.

### 3. Performance Analysis
To evaluate the utility of these features, we employ:
- **1-NN Retrieval**: To check if features can correctly identify the vault type.
- **SRCC (Spearman's Rank Correlation Coefficient)**: To measure the correlation between extracted features and the official Execution (E) scores.
- **Permutation Testing**: To ensure that the observed correlations are statistically significant and not due to random chance.

## 📈 Key Findings

Our initial analysis has yielded several interesting insights:
- **The "Landing" is Key**: The landing stage is the strongest individual predictor of execution quality, showing a significantly higher correlation with E-scores than the approach or flight stages.
- **Global Bias**: Whole-clip features suffer from strong source-video bias (selecting the same video ~76% of the time), making them less reliable for performance retrieval than stage-specific features.
- **High Separability**: We achieved **98.6% accuracy** in classifying the different stages of a vault, confirming that VideoMAE captures distinct temporal signatures for each phase of the movement.

## 📂 Project Structure

```text
├── analysis/          # Probing scripts for feature analysis and SRCC calculation
├── notebooks/         # Jupyter notebooks for feature extraction (Day 3)
├── scripts/           # Data pipeline: downloading clips and building manifests
├── out/               # Generated reports and availability CSVs
└── notes.md           # Detailed development log and experimental results
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- FFmpeg (required for clip extraction)
- PyTorch & HuggingFace Transformers

### Running the Pipeline
1. **Download Clips**: Use `scripts/download_clips.py` to fetch videos from YouTube based on the manifest.
2. **Verify Clips**: Run `scripts/check_clips.py` to ensure all downloaded files are usable.
3. **Extract Features**: Follow the logic in `notebooks/day3_feature_extraction.ipynb` to generate VideoMAE embeddings.
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
