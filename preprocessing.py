"""
Week 2: Preprocessing & Feature Engineering
ImpactSense - Earthquake Impact Prediction

This script handles:
- Data loading and duplicate removal
- Missing value handling
- Feature engineering (geospatial clusters, interaction features, depth/magnitude bins)
- Label encoding of categorical columns
- Normalization/scaling of numeric features
- Saving the cleaned and preprocessed dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_PATH = 'data/earthquake_data.csv'
OUTPUT_DIR = 'data'
REPORTS_DIR = 'analysis_reports'
MODELS_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("WEEK 2: PREPROCESSING & FEATURE ENGINEERING")
print("=" * 60)

try:
    df = pd.read_csv(DATA_PATH)
    print(f"\n[OK] Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"[ERROR] {DATA_PATH} not found. Run data_generator.py first.")
    exit(1)

# Display initial info
print(f"\nColumns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}\n")

# ============================================================
# 2. Handle Duplicates
# ============================================================
print("-" * 60)
print("Step 1: Handling Duplicates")
print("-" * 60)

num_duplicates = df.duplicated().sum()
print(f"  Duplicate rows found: {num_duplicates}")

if num_duplicates > 0:
    df = df.drop_duplicates()
    print(f"  Duplicates removed. New shape: {df.shape}")
else:
    print("  No duplicate rows found. Data is clean.")

# ============================================================
# 3. Handle Missing Values
# ============================================================
print("\n" + "-" * 60)
print("Step 2: Handling Missing Values")
print("-" * 60)

missing = df.isnull().sum()
total_missing = missing.sum()
print(f"  Total missing values: {total_missing}")

if total_missing > 0:
    print("  Missing values per column:")
    for col, count in missing[missing > 0].items():
        print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    Filled '{col}' with median: {median_val:.4f}")
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"    Filled '{col}' with mode: {mode_val}")
    
    print(f"  Missing values after handling: {df.isnull().sum().sum()}")
else:
    print("  No missing values found. Data is complete.")

# ============================================================
# 4. Feature Engineering
# ============================================================
print("\n" + "-" * 60)
print("Step 3: Feature Engineering")
print("-" * 60)

# 4b. Depth Bins (Shallow, Intermediate, Deep)
print("\n  [4b] Creating Depth Category Bins...")
df['Depth_Category'] = pd.cut(
    df['Depth'],
    bins=[0, 70, 300, 700],
    labels=['Shallow', 'Intermediate', 'Deep']
)
print(f"    Depth categories:\n{df['Depth_Category'].value_counts().to_string()}")

# Encode depth category
le_depth = LabelEncoder()
df['Depth_Category_Encoded'] = le_depth.fit_transform(df['Depth_Category'])
joblib.dump(le_depth, os.path.join(MODELS_DIR, 'depth_encoder.pkl'))

# 4c. Magnitude Bins (Low, Moderate, Strong, Major, Great)
print("\n  [4c] Creating Magnitude Category Bins...")
df['Magnitude_Category'] = pd.cut(
    df['Magnitude'],
    bins=[0, 3.9, 4.9, 5.9, 6.9, 10.0],
    labels=['Minor', 'Light', 'Moderate', 'Strong', 'Major']
)
print(f"    Magnitude categories:\n{df['Magnitude_Category'].value_counts().to_string()}")

le_mag = LabelEncoder()
df['Magnitude_Category_Encoded'] = le_mag.fit_transform(df['Magnitude_Category'])
joblib.dump(le_mag, os.path.join(MODELS_DIR, 'magnitude_encoder.pkl'))

# 4d. Interaction Features
print("\n  [4d] Creating Interaction Features...")
df['Mag_Depth_Ratio'] = df['Magnitude'] / (df['Depth'] + 1)  # +1 to avoid div by zero
df['Mag_Depth_Product'] = df['Magnitude'] * df['Depth']
df['Energy_Proxy'] = 10 ** (1.5 * df['Magnitude'] + 4.8)  # Approximate seismic energy
df['Log_Depth'] = np.log1p(df['Depth'])
df['Mag_Squared'] = df['Magnitude'] ** 2
print("    Created: Mag_Depth_Ratio, Mag_Depth_Product, Energy_Proxy, Log_Depth, Mag_Squared")

# 4e. Geospatial Clustering (Location Risk Zones)
print("\n  [4e] Creating Geospatial Clusters (Location Risk Zones)...")
geo_features = df[['Latitude', 'Longitude']].values

# Use KMeans to create geographical clusters
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Geo_Cluster'] = kmeans.fit_predict(geo_features)
print(f"    Created {n_clusters} geographical clusters")
print(f"    Cluster distribution:\n{df['Geo_Cluster'].value_counts().sort_index().to_string()}")
joblib.dump(kmeans, os.path.join(MODELS_DIR, 'geo_kmeans.pkl'))

# 4f. Location Risk Score based on cluster average risk
print("\n  [4f] Computing Location Risk Score...")
cluster_risk = df.groupby('Geo_Cluster')['Risk_Score'].mean()
df['Location_Risk_Score'] = df['Geo_Cluster'].map(cluster_risk)
print(f"    Average risk per cluster:")
for cluster_id, risk in cluster_risk.sort_values(ascending=False).items():
    print(f"      Cluster {cluster_id}: {risk:.2f}")

# 4g. Fault Proximity Category
print("\n  [4g] Creating Fault Proximity Categories...")
df['Fault_Category'] = pd.cut(
    df['Fault_Proximity'],
    bins=[0, 20, 50, 100],
    labels=['Near', 'Medium', 'Far']
)
le_fault = LabelEncoder()
df['Fault_Category_Encoded'] = le_fault.fit_transform(df['Fault_Category'])
print(f"    Fault proximity categories:\n{df['Fault_Category'].value_counts().to_string()}")

# ============================================================
# 5. Normalize / Scale Numeric Features
# ============================================================
print("\n" + "-" * 60)
print("Step 4: Feature Scaling / Normalization")
print("-" * 60)

# Features to be used for model training
feature_columns = [
    'Magnitude', 'Depth', 'Latitude', 'Longitude',
    'Fault_Proximity',
    'Depth_Category_Encoded', 'Magnitude_Category_Encoded',
    'Mag_Depth_Ratio', 'Mag_Depth_Product',
    'Log_Depth', 'Mag_Squared',
    'Geo_Cluster', 'Location_Risk_Score',
    'Fault_Category_Encoded'
]

print(f"\n  Feature columns for modeling ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"    {i:2d}. {col}")

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print("\n  StandardScaler fitted and saved to models/scaler.pkl")

# Create scaled DataFrame for reference
df_scaled = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in feature_columns])

# ============================================================
# 6. Visualize Engineered Features
# ============================================================
print("\n" + "-" * 60)
print("Step 5: Generating Feature Engineering Visualizations")
print("-" * 60)

sns.set(style="whitegrid")

# Plot 1: Depth Category vs Impact Class
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(x='Depth_Category', hue='Impact_Class', data=df, palette='coolwarm', ax=axes[0, 0])
axes[0, 0].set_title('Depth Category vs Impact Class')
axes[0, 0].legend(title='Impact', labels=['Low', 'High'])

sns.countplot(x='Magnitude_Category', hue='Impact_Class', data=df, palette='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('Magnitude Category vs Impact Class')
axes[0, 1].legend(title='Impact', labels=['Low', 'High'])
axes[0, 1].tick_params(axis='x', rotation=30)

sns.boxplot(x='Geo_Cluster', y='Risk_Score', data=df, palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Risk Score Distribution by Geo Cluster')

sns.countplot(x='Fault_Category', hue='Impact_Class', data=df, palette='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Fault Proximity Category vs Impact Class')
axes[1, 1].legend(title='Impact', labels=['Low', 'High'])

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'feature_engineering_plots.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/feature_engineering_plots.png")

# Plot 2: Correlation matrix of all engineered features
plt.figure(figsize=(16, 12))
corr_cols = feature_columns + ['Risk_Score', 'Impact_Class']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', fmt='.2f',
            center=0, linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix (Engineered Features)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'engineered_correlation_matrix.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/engineered_correlation_matrix.png")

# Plot 3: Scatter plot - Magnitude vs Depth colored by Impact
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Magnitude'], df['Depth'], c=df['Impact_Class'],
                      cmap='RdYlGn_r', alpha=0.4, s=10)
plt.colorbar(scatter, label='Impact Class')
plt.xlabel('Magnitude')
plt.ylabel('Depth (km)')
plt.title('Magnitude vs Depth (Colored by Impact Class)')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'magnitude_vs_depth_impact.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/magnitude_vs_depth_impact.png")

# Plot 4: Geo clusters on a map-like scatter
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['Geo_Cluster'],
                      cmap='tab10', alpha=0.4, s=10)
plt.colorbar(scatter, label='Geo Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geospatial Clusters')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'geo_clusters.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/geo_clusters.png")

# ============================================================
# 7. Save Preprocessed Dataset
# ============================================================
print("\n" + "-" * 60)
print("Step 6: Saving Preprocessed Dataset")
print("-" * 60)

# Save feature columns list for model training
feature_config = {
    'feature_columns': feature_columns,
    'target_column': 'Impact_Class',
    'regression_target': 'Risk_Score'
}
joblib.dump(feature_config, os.path.join(MODELS_DIR, 'feature_config.pkl'))
print("  Feature configuration saved to models/feature_config.pkl")

# Save preprocessed data
output_path = os.path.join(OUTPUT_DIR, 'earthquake_preprocessed.csv')
df.to_csv(output_path, index=False)
print(f"  Preprocessed dataset saved to {output_path}")
print(f"  Final shape: {df.shape}")
print(f"  Total features for modeling: {len(feature_columns)}")

# Summary
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE - SUMMARY")
print("=" * 60)
print(f"  Original samples: 10000 | After cleaning: {len(df)}")
print(f"  Original features: 8 | Engineered features: {len(feature_columns)}")
print(f"  New features created:")
print(f"    - Depth_Category (Shallow/Intermediate/Deep)")
print(f"    - Magnitude_Category (Minor/Light/Moderate/Strong/Major)")
print(f"    - Mag_Depth_Ratio, Mag_Depth_Product, Log_Depth, Mag_Squared")
print(f"    - Geo_Cluster (KMeans, {n_clusters} zones)")
print(f"    - Location_Risk_Score (cluster-based)")
print(f"    - Fault_Category (Near/Medium/Far)")
print(f"\n  Saved artifacts:")
print(f"    - data/earthquake_preprocessed.csv")
print(f"    - models/scaler.pkl, label_encoder.pkl")
print(f"    - models/depth_encoder.pkl, magnitude_encoder.pkl")
print(f"    - models/geo_kmeans.pkl, feature_config.pkl")
print(f"    - analysis_reports/feature_engineering_plots.png")
print(f"    - analysis_reports/engineered_correlation_matrix.png")
print(f"    - analysis_reports/magnitude_vs_depth_impact.png")
print(f"    - analysis_reports/geo_clusters.png")
print("=" * 60)
