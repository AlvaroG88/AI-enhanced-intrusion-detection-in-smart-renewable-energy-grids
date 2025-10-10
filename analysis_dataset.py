# temporal_analysis.py
# ------------------------------------------------------------
# RQ1: Do cyber-attacks follow identifiable temporal patterns?
# - Loads processed_smart_grid_attacks.csv (Step 1 output)
# - Frequency analysis: hour & weekday
# - Visuals: histograms + heatmaps
# - Statistical tests: Chi-square + Cramér's V
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import math

INPUT = "processed_smart_grid_attacks.csv"
OUTDIR = Path("analysis_outputs2")
OUTDIR.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def cramer_v(chi2, n, r, c):
    """Cramér's V effect size for chi-square."""
    # bias-corrected version (Bergsma 2013) for contingency tables
    phi2 = chi2 / n
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    c_corr = c - ((c - 1) ** 2) / (n - 1)
    phi2_corr = max(0, phi2 - ((c - 1)*(r - 1)) / (n - 1))
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return 0.0
    return math.sqrt(phi2_corr / denom)

def save_bar(ax, title, xlabel, ylabel, outfile):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / outfile, dpi=150)
    plt.close()

def save_heatmap(matrix_df, title, xlabel, ylabel, outfile):
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix_df.values, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Count")
    # tick labels
    plt.xticks(ticks=range(matrix_df.shape[1]), labels=list(matrix_df.columns), rotation=0)
    plt.yticks(ticks=range(matrix_df.shape[0]), labels=list(matrix_df.index))
    plt.tight_layout()
    plt.savefig(OUTDIR / outfile, dpi=150)
    plt.close()

def chi_square_with_report(contingency, name):
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    n = contingency.values.sum()
    r, c = contingency.shape
    v = cramer_v(chi2, n, r, c)
    print(f"\n[Chi-square] {name}")
    print(f" - chi2={chi2:.3f}, dof={dof}, p-value={p:.6f}, n={n}")
    print(f" - Cramér's V={v:.3f}  (≈0.1 small, ≈0.3 medium, ≈0.5 large)")
    return chi2, p, v

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT)
# Expect columns from Step 1/your last script:
# timestamp, source_ip, destination_ip, port, protocol, packet_size,
# attack_type, source_ip_valid, destination_ip_valid,
# hour, day, weekday, time_segment, is_weekend

# Define an "is_attack" flag (if not present): benign -> 0, else 1
if "is_attack" not in df.columns:
    df["is_attack"] = (df["attack_type"].fillna("benign").str.lower() != "benign").astype(int)

# -----------------------------
# 1) Frequency Analysis
# -----------------------------

# A) Attacks by hour (count of attack events; you can also plot ratio)
attacks_by_hour = df[df["is_attack"] == 1]["hour"].value_counts().sort_index()
plt.figure(figsize=(9, 4))
attacks_by_hour.plot(kind="bar")
save_bar(plt.gca(), "Attacks by Hour", "Hour of day (0–23)", "Count", "attacks_by_hour.png")

# Optional: Attack RATE by hour (attacks / total events)
rate_by_hour = (df["is_attack"]
                .groupby(df["hour"])
                .mean()
                .reindex(range(24), fill_value=0))
plt.figure(figsize=(9, 4))
rate_by_hour.plot(kind="bar")
save_bar(plt.gca(), "Attack Rate by Hour", "Hour of day (0–23)", "Attack rate", "attack_rate_by_hour.png")

# B) Attacks by weekday
# Weekday: 0=Mon ... 6=Sun (already your encoding)
attacks_by_weekday = df[df["is_attack"] == 1]["weekday"].value_counts().sort_index()
plt.figure(figsize=(9, 4))
attacks_by_weekday.plot(kind="bar")
save_bar(plt.gca(), "Attacks by Weekday", "Weekday (0=Mon ... 6=Sun)", "Count", "attacks_by_weekday.png")

# Optional: Attack RATE by weekday
rate_by_weekday = df.groupby("weekday")["is_attack"].mean().reindex(range(7), fill_value=0)
plt.figure(figsize=(9, 4))
rate_by_weekday.plot(kind="bar")
save_bar(plt.gca(), "Attack Rate by Weekday", "Weekday (0=Mon ... 6=Sun)", "Attack rate", "attack_rate_by_weekday.png")

# C) Heatmaps: Attack_Type vs Hour / Weekday (non-benign only)
att_nonbenign = df[df["is_attack"] == 1].copy()
if not att_nonbenign.empty:
    # Hour heatmap
    ht_hour = pd.crosstab(att_nonbenign["attack_type"], att_nonbenign["hour"]).reindex(columns=range(24), fill_value=0)
    save_heatmap(ht_hour, "Attack Type vs Hour", "Hour", "Attack Type", "attacktype_by_hour_heatmap.png")

    # Weekday heatmap
    ht_wd = pd.crosstab(att_nonbenign["attack_type"], att_nonbenign["weekday"]).reindex(columns=range(7), fill_value=0)
    save_heatmap(ht_wd, "Attack Type vs Weekday", "Weekday (0=Mon ... 6=Sun)", "Attack Type", "attacktype_by_weekday_heatmap.png")
else:
    print("No non-benign attacks found; heatmaps skipped.")

# -----------------------------
# 2) Statistical Tests
# -----------------------------
print("\n=== Statistical Tests (RQ1) ===")

# (i) Is attack occurrence independent of hour?
# Build contingency: hour x is_attack (0/1)
cont_hour = pd.crosstab(df["hour"], df["is_attack"]).reindex(index=range(24), fill_value=0)
chi_square_with_report(cont_hour, "is_attack ~ hour")

# (ii) Is attack occurrence independent of weekday?
cont_wd = pd.crosstab(df["weekday"], df["is_attack"]).reindex(index=range(7), fill_value=0)
chi_square_with_report(cont_wd, "is_attack ~ weekday")

# (iii) Does attack TYPE distribution depend on hour? (non-benign only)
if not att_nonbenign.empty and att_nonbenign["attack_type"].nunique() > 1:
    cont_type_hour = pd.crosstab(att_nonbenign["hour"], att_nonbenign["attack_type"]).reindex(index=range(24), fill_value=0)
    chi_square_with_report(cont_type_hour, "attack_type ~ hour")
else:
    print("\n[Chi-square] attack_type ~ hour: skipped (not enough non-benign categories).")

# (iv) Does attack TYPE distribution depend on weekday? (non-benign only)
if not att_nonbenign.empty and att_nonbenign["attack_type"].nunique() > 1:
    cont_type_wd = pd.crosstab(att_nonbenign["weekday"], att_nonbenign["attack_type"]).reindex(index=range(7), fill_value=0)
    chi_square_with_report(cont_type_wd, "attack_type ~ weekday")
else:
    print("[Chi-square] attack_type ~ weekday: skipped (not enough non-benign categories).")

print("\nFigures saved to:", OUTDIR.resolve())
print("Done.")

# ------------------------------------------------------------
TOP_K = 6  # keep top-K attack types, group the rest as 'other'

# Work on non-benign only (so 'benign' doesn't dominate the plot)
att_nb = df[df["is_attack"] == 1].copy()

if att_nb.empty:
    print("No non-benign attacks found; stacked bar charts skipped.")
else:
    # ---- Group into top-K + 'other' ----
    type_counts = att_nb["attack_type"].fillna("unknown").value_counts()
    top_types = set(type_counts.head(TOP_K).index)

    def map_type(t):
        t = str(t) if pd.notna(t) else "unknown"
        return t if t in top_types else "other"

    att_nb["attack_type_topk"] = att_nb["attack_type"].apply(map_type)

    # ============== Attack_Type vs Hour ==============
    # counts
    ct_hour = pd.crosstab(att_nb["hour"], att_nb["attack_type_topk"]).reindex(index=range(24), fill_value=0)
    # stacked bar (counts)
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(ct_hour))
    for col in ct_hour.columns:
        plt.bar(ct_hour.index, ct_hour[col].values, bottom=bottom, label=col)
        bottom += ct_hour[col].values
    plt.title("Attack Type vs Hour (stacked counts, non-benign)")
    plt.xlabel("Hour of day (0–23)")
    plt.ylabel("Count")
    plt.legend(title="Attack Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "attack_type_by_hour_bars.png", dpi=150)
    plt.close()

    # normalized (proportions per hour)
    ct_hour_prop = ct_hour.div(ct_hour.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(ct_hour_prop))
    for col in ct_hour_prop.columns:
        plt.bar(ct_hour_prop.index, ct_hour_prop[col].values, bottom=bottom, label=col)
        bottom += ct_hour_prop[col].values
    plt.title("Attack Type vs Hour (stacked proportions, non-benign)")
    plt.xlabel("Hour of day (0–23)")
    plt.ylabel("Proportion")
    plt.legend(title="Attack Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "attack_type_by_hour_bars_normalized.png", dpi=150)
    plt.close()

    # ============== Attack_Type vs Weekday ==============
    # counts
    ct_wd = pd.crosstab(att_nb["weekday"], att_nb["attack_type_topk"]).reindex(index=range(7), fill_value=0)
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(ct_wd))
    for col in ct_wd.columns:
        plt.bar(ct_wd.index, ct_wd[col].values, bottom=bottom, label=col)
        bottom += ct_wd[col].values
    plt.title("Attack Type vs Weekday (stacked counts, non-benign)")
    plt.xlabel("Weekday (0=Mon ... 6=Sun)")
    plt.ylabel("Count")
    plt.legend(title="Attack Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "attack_type_by_weekday_bars.png", dpi=150)
    plt.close()

    # normalized (proportions per weekday)
    ct_wd_prop = ct_wd.div(ct_wd.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(ct_wd_prop))
    for col in ct_wd_prop.columns:
        plt.bar(ct_wd_prop.index, ct_wd_prop[col].values, bottom=bottom, label=col)
        bottom += ct_wd_prop[col].values
    plt.title("Attack Type vs Weekday (stacked proportions, non-benign)")
    plt.xlabel("Weekday (0=Mon ... 6=Sun)")
    plt.ylabel("Proportion")
    plt.legend(title="Attack Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "attack_type_by_weekday_bars_normalized.png", dpi=150)
    plt.close()

    print("Extra figures saved:",
          (OUTDIR / "attack_type_by_hour_bars.png").resolve(),
          (OUTDIR / "attack_type_by_weekday_bars.png").resolve(),
          (OUTDIR / "attack_type_by_hour_bars_normalized.png").resolve(),
          (OUTDIR / "attack_type_by_weekday_bars_normalized.png").resolve(),
          sep="\n  - ")

