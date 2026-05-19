
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

try:
    df = pd.read_csv('data/earthquake_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: data/earthquake_data.csv not found.")
    exit(1)

print("\n" + "="*50)
print(f"Dataset Overview (Total Samples: {len(df)})")
print("="*50)

# Check for duplicates
duplicates = df[df.duplicated()]
num_duplicates = len(duplicates)

print(f"\nDuplicate Rows Found: {num_duplicates}")
if num_duplicates > 0:
    print("-" * 30)
    print(duplicates.head(20))
    print("-" * 30)
else:
    print("No identical duplicate rows found.")

print("\n" + "="*50)
print("Detailed Data Inspection (First 50 Rows)")
print("Columns: Magnitude, Latitude, Longitude, Depth, Soil_Type")
print("="*50)

# Select specific columns for inspection as requested
inspection_df = df[['Magnitude', 'Latitude', 'Longitude', 'Depth', 'Soil_Type']]
print(inspection_df.head(50))

print("\n" + "="*50)
print("Statistical Summary of Inspection Columns")
print("="*50)
print(inspection_df.describe())
