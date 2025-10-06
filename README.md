# AI-enhanced-intrusion-detection-in-smart-renewable-energy-grids
The paper uses the Smart Grid Intrusion Detection Dataset with Random Forest + Autoencoder to boost accuracy (97.8%) and cut false positives. But even though the dataset has attack types (DoS, Malware, Phishing, MITM, SQLi, Zero-day), it only does binary detection. So adding attack-type classification could be your key contribution.
The input is the dataset containing timestamps and attack types. The output of our model will be: 
(1) Statistical evidence of time-based attack patterns
(2) A predictive model estimating the likelihood of the next attack based on temporal features
We have two research questions on paper:
1) Do cyber-attacks in smart grids follow identifiable temporal patterns?
2) Can we predict the likelihood of the next attack using temporal features such as time of day, day of week, and peak hours?
Implementation (Step-by-Step):
Step 1 – Data Preparation & Feature Engineering
Load and clean the dataset (remove inconsistencies and missing values).
Convert Timestamp to datetime and extract temporal features: hour, day, weekday, time_segment (morning/afternoon/night), and is_weekend.
Output: structured dataset with time-based features.
Step 2 – Temporal Pattern Analysis (Answer to RQ1)
Analyze attack frequency by hour and weekday.
Visualize with heatmaps and histograms (e.g., Attack_Type vs Hour).
Conduct statistical tests (Chi-square / ANOVA) to check if attack occurrence depends on time.
Output: visual and statistical evidence of temporal attack patterns.
Step 3 – Temporal Feature Selection
Evaluate which time features (hour, weekday, etc.) are most predictive of attack behavior using feature importance.
Output: ranked list of most influential temporal features.
Step 4 – Temporal Prediction Modeling (Answer to RQ2)
Build a prediction model (e.g., LSTM, XGBoost Algorithms) using the selected temporal features.
Train the model to predict the next probable attack type or occurrence based on recent time patterns.
Evaluate using accuracy, precision, Recall, F1-score, and confusion matrix.
Output: a trained model capable of predicting attack likelihood.
