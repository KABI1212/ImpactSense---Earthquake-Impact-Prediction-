"""
Week 1: Data Exploration & Analysis
ImpactSense - Earthquake Impact Prediction

This script performs Exploratory Data Analysis (EDA) on the raw earthquake dataset:
- Feature distributions (magnitude, depth, lat/lon, fault proximity, risk score)
- Correlation heatmap
- Geographic earthquake distribution
- Risk score analysis by feature bands
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_PATH = "data/earthquake_data.csv"
REPORTS_DIR = "analysis_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("WEEK 1: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

try:
    df = pd.read_csv(DATA_PATH)
    print(f"\n[OK] Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print(f"[ERROR] {DATA_PATH} not found.")
    raise SystemExit(1)

print(f"\nColumns      : {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}\n")
print(f"Missing values:\n{df.isnull().sum()}\n")
print(f"Basic statistics:\n{df.describe().round(3)}\n")

n_duplicates = df.duplicated().sum()
print(f"Duplicate rows: {n_duplicates}")

# Impact class balance
print(f"\nImpact_Class distribution:\n{df['Impact_Class'].value_counts()}")
print(f"  High-impact rate: {df['Impact_Class'].mean() * 100:.1f}%")

# ============================================================
# 2. Feature Distribution Plots
# ============================================================
print("\n" + "-" * 60)
print("Generating feature distribution plots...")

features = ["Magnitude", "Depth", "Latitude", "Longitude", "Fault_Proximity", "Risk_Score"]
labels = [
    "Magnitude",
    "Depth (km)",
    "Latitude (°)",
    "Longitude (°)",
    "Fault Proximity (km)",
    "Risk Score",
]
colors = ["#4A90D9", "#E87040", "#5AB76E", "#9B59B6", "#E74C3C", "#F39C12"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, col, label, color in zip(axes, features, labels, colors):
    data = df[col].dropna()
    ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean: {data.mean():.2f}")
    ax.axvline(data.median(), color="grey", linestyle=":", linewidth=1.5, label=f"Median: {data.median():.2f}")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Distribution of {label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

plt.suptitle("Feature Distributions — ImpactSense Dataset", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(REPORTS_DIR, "eda_distributions.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ============================================================
# 3. Correlation Heatmap
# ============================================================
print("  Generating correlation heatmap...")

numeric_cols = ["Magnitude", "Depth", "Latitude", "Longitude", "Fault_Proximity", "Risk_Score", "Impact_Class"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
out = os.path.join(REPORTS_DIR, "eda_correlation.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ============================================================
# 4. Geographic Earthquake Map
# ============================================================
print("  Generating geographic distribution map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: colored by Risk Score
scatter = axes[0].scatter(
    df["Longitude"], df["Latitude"],
    c=df["Risk_Score"], cmap="RdYlGn_r",
    alpha=0.35, s=6, linewidths=0
)
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label("Risk Score", fontsize=11)
axes[0].set_xlabel("Longitude (°)", fontsize=11)
axes[0].set_ylabel("Latitude (°)", fontsize=11)
axes[0].set_title("Earthquake Locations — Colored by Risk Score", fontsize=12, fontweight="bold")
axes[0].set_facecolor("#f0f4f8")
axes[0].grid(True, alpha=0.3)

# Right: colored by Impact Class
colors_class = df["Impact_Class"].map({0: "#2ecc71", 1: "#e74c3c"})
axes[1].scatter(
    df["Longitude"], df["Latitude"],
    c=colors_class, alpha=0.35, s=6, linewidths=0
)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", label="Low Impact"),
    Patch(facecolor="#e74c3c", label="High Impact"),
]
axes[1].legend(handles=legend_elements, loc="lower left", fontsize=10)
axes[1].set_xlabel("Longitude (°)", fontsize=11)
axes[1].set_ylabel("Latitude (°)", fontsize=11)
axes[1].set_title("Earthquake Locations — Colored by Impact Class", fontsize=12, fontweight="bold")
axes[1].set_facecolor("#f0f4f8")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Global Earthquake Distribution", fontsize=15, fontweight="bold")
plt.tight_layout()
out = os.path.join(REPORTS_DIR, "eda_geo_map.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ============================================================
# 5. Risk Score by Feature Band
# ============================================================
print("  Generating risk score by feature band plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Magnitude bands
mag_bins = pd.cut(df["Magnitude"], bins=[0, 3.9, 4.9, 5.9, 6.9, 10.0],
                  labels=["Minor\n≤3.9", "Light\n4–4.9", "Moderate\n5–5.9", "Strong\n6–6.9", "Major\n≥7"])
df_tmp = df.copy()
df_tmp["MagBand"] = mag_bins
order = ["Minor\n≤3.9", "Light\n4–4.9", "Moderate\n5–5.9", "Strong\n6–6.9", "Major\n≥7"]
sns.boxplot(data=df_tmp, x="MagBand", y="Risk_Score", order=order, palette="Reds", ax=axes[0])
axes[0].set_title("Risk Score by Magnitude Band", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Magnitude Band")
axes[0].set_ylabel("Risk Score")

# Depth bands
depth_bins = pd.cut(df["Depth"], bins=[0, 70, 300, 800], labels=["Shallow\n≤70 km", "Intermediate\n70–300 km", "Deep\n>300 km"])
df_tmp["DepthBand"] = depth_bins
order_d = ["Shallow\n≤70 km", "Intermediate\n70–300 km", "Deep\n>300 km"]
sns.boxplot(data=df_tmp, x="DepthBand", y="Risk_Score", order=order_d, palette="Blues", ax=axes[1])
axes[1].set_title("Risk Score by Depth Band", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Depth Band")
axes[1].set_ylabel("Risk Score")

# Fault proximity bands
fault_bins = pd.cut(df["Fault_Proximity"], bins=[0, 20, 50, 120], labels=["Near\n≤20 km", "Medium\n20–50 km", "Far\n>50 km"])
df_tmp["FaultBand"] = fault_bins
order_f = ["Near\n≤20 km", "Medium\n20–50 km", "Far\n>50 km"]
sns.boxplot(data=df_tmp, x="FaultBand", y="Risk_Score", order=order_f, palette="Greens", ax=axes[2])
axes[2].set_title("Risk Score by Fault Proximity", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Fault Proximity Band")
axes[2].set_ylabel("Risk Score")

plt.suptitle("Risk Score Distribution by Feature Bands", fontsize=14, fontweight="bold")
plt.tight_layout()
out = os.path.join(REPORTS_DIR, "eda_risk_by_band.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("WEEK 1 EDA COMPLETE")
print("=" * 60)
print(f"  Dataset rows  : {len(df):,}")
print(f"  Dataset cols  : {df.shape[1]}")
print(f"  Duplicates    : {n_duplicates}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  High-impact % : {df['Impact_Class'].mean() * 100:.1f}%")
print(f"  Magnitude range: {df['Magnitude'].min():.1f} – {df['Magnitude'].max():.1f}")
print(f"  Depth range    : {df['Depth'].min():.1f} – {df['Depth'].max():.1f} km")
print(f"\n  Saved to analysis_reports/:")
print(f"    - eda_distributions.png")
print(f"    - eda_correlation.png")
print(f"    - eda_geo_map.png")
print(f"    - eda_risk_by_band.png")
print("=" * 60)
