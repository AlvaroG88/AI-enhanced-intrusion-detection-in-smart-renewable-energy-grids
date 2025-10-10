# AI-Enhanced Intrusion Detection in Smart Renewable Energy Grids

This project explores **AI-driven intrusion detection** for smart renewable energy grids by integrating **temporal behavior analysis** and **attack-type classification**.  
Using the *Smart Grid Intrusion Detection Dataset*, our model enhances the baseline binary detection (normal vs. attack) into a **multi-class predictive framework** capable of identifying attack types such as **DoS, Malware, Phishing, MITM, SQL Injection, and Zero-day**.

The proposed architecture combines **Random Forest** and **Autoencoder** models to achieve high detection accuracy (97.8%) while reducing false positives. Beyond classification, it focuses on **temporal intelligence** — discovering how attack occurrences vary by hour, weekday, or operational phase of the grid.

---

## Research Objectives
1. ** RQ1 (Temporal Pattern Discovery)**: Do cyber-attacks in smart grids follow identifiable time-based patterns?  
2. ** RQ2 (Predictive Modeling)**: Can temporal features (hour, day, weekday, time segment) predict the likelihood or type of the next attack?

---

## Methodology
**Step 1 – Data Preparation & Feature Engineering**  
Clean raw data, convert timestamps, and extract time-related features (`hour`, `day`, `weekday`, `time_segment`, `is_weekend`).

**Step 2 – Temporal Pattern Analysis**  
Visualize attack frequency by time (hour, weekday) using heatmaps and histograms.  
Apply statistical tests (Chi-square / ANOVA) to confirm time-dependency of attacks.

**Step 3 – Feature Ranking**  
Use feature importance analysis (Random Forest) to identify the most predictive temporal factors.

**Step 4 – Temporal Prediction Model**  
Train hybrid models (LSTM / XGBoost) to forecast attack likelihood or type based on recent temporal sequences.  
Evaluate using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

---

## Repository Structure
├── README.md
├── prepare_dataset.py # Data cleaning & time-feature extraction
├── analysis_dataset.py # Temporal pattern analysis & visualization
├── cyber_threat_preparation.ipynb # Notebook version for preprocessing
│
├── Dataset/
│ ├── Smart Grid Intrusion Detection Dataset.csv
│ ├── cyber_threats_cleaned.csv
│ └── processed_smart_grid_attacks.csv
│
└── analysis_figures2/
├── attacks_by_hour.png
├── attack_rate_by_weekday.png
├── attacktype_by_hour_heatmap.png
└── ...

---

## Expected Contributions
- **Multi-class detection** for diverse smart-grid attack types  
- **Temporal insight** into attack behaviors across operational cycles  
- **Predictive AI model** for proactive cybersecurity in energy networks
