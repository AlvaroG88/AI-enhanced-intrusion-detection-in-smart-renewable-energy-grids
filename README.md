## Project Rationale (verbatim)
> The paper uses the Smart Grid Intrusion Detection Dataset with Random Forest + Autoencoder to boost accuracy (97.8%) and cut false positives. But even though the dataset has attack types (DoS, Malware, Phishing, MITM, SQLi, Zero-day), it only does binary detection. So adding attack-type classification could be your key contribution. The input is the dataset containing timestamps and attack types. The output of our model will be: (1) Statistical evidence of time-based attack patterns (2) A predictive model estimating the likelihood of the next attack based on temporal features
>
> We have two research questions on paper:
>
> Do cyber-attacks in smart grids follow identifiable temporal patterns?
> Can we predict the likelihood of the next attack using temporal features such as time of day, day of week, and peak hours?
> Implementation (Step-by-Step):
>
> Step 1 – Data Preparation & Feature Engineering Load and clean the dataset (remove inconsistencies and missing values). Convert Timestamp to datetime and extract temporal features: hour, day, weekday, time_segment (morning/afternoon/night), and is_weekend. Output: structured dataset with time-based features.
>
> Step 2 – Temporal Pattern Analysis (Answer to RQ1) Analyze attack frequency by hour and weekday. Visualize with heatmaps and histograms (e.g., Attack_Type vs Hour). Conduct statistical tests (Chi-square / ANOVA) to check if attack occurrence depends on time. Output: visual and statistical evidence of temporal attack patterns.
>
> Step 3 – Temporal Feature Selection Evaluate which time features (hour, weekday, etc.) are most predictive of attack behavior using feature importance. Output: ranked list of most influential temporal features.
>
> Step 4 – Temporal Prediction Modeling (Answer to RQ2) Build a prediction model (e.g., LSTM, XGBoost Algorithms) using the selected temporal features. Train the model to predict the next probable attack type or occurrence based on recent time patterns. Evaluate using accuracy, precision, Recall, F1-score, and confusion matrix. Output: a trained model capable of predicting attack likelihood.·

---

## File Index

### Root
- **README.md** — You are here.
- **prepare_dataset.py** — Step 1 script. Loads raw CSV, cleans it, converts timestamps, and creates time features (`hour`, `day`, `weekday`, `time_segment`, `is_weekend`, IP validation). Outputs the processed CSV.
- **temporal_analysis.py** *(if present)* — Step 2 script. Frequency plots (hour/weekday), heatmaps, stacked bars; Chi-square tests + Cramér’s V.
- **analysis_dataset.py** — Legacy/auxiliary analysis script (kept for reference).
- **cyber_threat_preparation.ipynb** — Notebook version for data preparation/cleaning (Step 1).
- **.ipynb_checkpoints/** — Jupyter auto-generated checkpoint files (safe to ignore).

### `Dataset/`
- **Smart Grid Intrusion Detection Dataset - Copy.csv** — Raw dataset (original data source).
- **cyber_threats_cleaned.csv** — Cleaned export (intermediate).
- **processed_smart_grid_attacks\*.csv** — Processed dataset from Step 1 with standardized columns:
  `timestamp, source_ip, destination_ip, port, protocol, packet_size, attack_type, source_ip_valid, destination_ip_valid, hour, day, weekday, time_segment, is_weekend`.

### `analysis_figures2/`
- **Explanation of figures** — Brief textual summary of figure meanings.
- **attacks_by_hour.png** — Count of attacks per hour (0–23). *(count)*
- **attack_rate_by_hour.png** — Attack rate per hour (attacks / total events). *(proportion)*
- **attacks_by_weekday.png** — Count of attacks per weekday 0–6 (0=Mon … 6=Sun). *(count)*
- **attack_rate_by_weekday.png** — Attack rate per weekday (attacks / total events). *(proportion)*
- **attacktype_by_hour_heatmap.png** — Heatmap of non-benign `attack_type` counts by hour. *(heatmap)*
- **attacktype_by_weekday_heatmap.png** — Heatmap of non-benign `attack_type` counts by weekday. *(heatmap)*
- **attack_type_by_hour_bars.png** — Stacked counts of non-benign `attack_type` by hour. *(stacked counts)*
- **attack_type_by_hour_bars_normalized.png** — Stacked proportions (each bar sums to 1) of non-benign `attack_type` by hour. *(stacked proportions)*
- **attack_type_by_weekday_bars.png** — Stacked counts of non-benign `attack_type` by weekday. *(stacked counts)*
- **attack_type_by_weekday_bars_normalized.png** — Stacked proportions (each bar sums to 1) of non-benign `attack_type` by weekday. *(stacked proportions)*

---
