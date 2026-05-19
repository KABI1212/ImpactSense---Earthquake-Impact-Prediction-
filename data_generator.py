
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 10000

# Generate synthetic data
data = {
    'Magnitude': np.random.uniform(2.5, 9.0, num_samples),
    'Depth': np.random.uniform(5, 700, num_samples),  # Depth in km
    'Latitude': np.random.uniform(-90, 90, num_samples),
    'Longitude': np.random.uniform(-180, 180, num_samples),
    'Fault_Proximity': np.random.uniform(0, 100, num_samples) # Distance to nearest fault line in km
}

df = pd.DataFrame(data)

# Simulate Impact Score calculation (simplified physics-based logic)
# Higher magnitude -> higher impact
# Shallower depth -> higher impact
def calculate_risk(row):
    mag_factor = row['Magnitude'] ** 1.5
    depth_factor = 100 / (row['Depth'] + 10) # Shallower is more dangerous
    
    risk_score = (mag_factor * depth_factor) - (row['Fault_Proximity'] * 0.05)
    return risk_score

df['Risk_Score'] = df.apply(calculate_risk, axis=1)

# Normalize Risk Score to 0-100 range for readability (optional, but good for analysis)
df['Risk_Score'] = (df['Risk_Score'] - df['Risk_Score'].min()) / (df['Risk_Score'].max() - df['Risk_Score'].min()) * 100

# Define Target Variable: Impact Class (0: Low, 1: High)
# Threshold set at median for balanced classes initially
threshold = df['Risk_Score'].median()
df['Impact_Class'] = (df['Risk_Score'] > threshold).astype(int)

# Introduce some noise to make it realistic for ML model
noise = np.random.normal(0, 5, num_samples)
df['Risk_Score'] += noise

# Recalculate class after noise (optional, but keeps consistency)
df['Impact_Class'] = (df['Risk_Score'] > threshold).astype(int)

# Save to CSV
output_file = 'data/earthquake_data.csv'
# Ensure directory exists
import os
os.makedirs('data', exist_ok=True)
df.to_csv(output_file, index=False)

print(f"Synthetic dataset generated: {output_file} with {num_samples} samples.")
print(df.head())
