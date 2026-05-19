
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create directories for reports
os.makedirs('analysis_reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Data
try:
    df = pd.read_csv('data/earthquake_data.csv')
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print("Error: data/earthquake_data.csv not found.")
    exit(1)

# 2. EDA & Visualization

# Set style
sns.set(style="whitegrid")

# Magnitude Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Magnitude'], bins=30, kde=True, color='blue')
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.savefig('analysis_reports/magnitude_distribution.png')
plt.close()

# Depth vs Risk Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Depth', y='Risk_Score', hue='Impact_Class', data=df, palette='coolwarm')
plt.title('Depth vs Risk Score (Colored by Impact Class)')
plt.savefig('analysis_reports/depth_vs_risk.png')
plt.close()

# Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.savefig('analysis_reports/correlation_matrix.png')
plt.close()

print("EDA plots saved to 'analysis_reports/' directory.")

# Terminal Visualizations (ASCII)
def print_ascii_histogram(data, title, bins=10):
    counts, bin_edges = np.histogram(data, bins=bins)
    print(f"\n{title}")
    print("-" * 50)
    max_count = max(counts) if len(counts) > 0 else 1
    for i in range(len(counts)):
        bar_length = int(40 * counts[i] / max_count)
        bar = '#' * bar_length
        print(f"{bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar:<40} ({counts[i]})")
    print("-" * 50)

print_ascii_histogram(df['Magnitude'], "Magnitude Distribution (ASCII Histogram)")
print_ascii_histogram(df['Risk_Score'], "Risk Score Distribution (ASCII Histogram)")

# 3. Preprocessing
features = ['Magnitude', 'Depth', 'Latitude', 'Longitude', 'Fault_Proximity']
X = df[features]
y = df['Impact_Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessors
joblib.dump(scaler, 'models/scaler.pkl')

# 4. Model Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

# Save Metrics
with open('analysis_reports/metrics.txt', 'w') as f:
    f.write(f"Model: Random Forest Classifier\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    
# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('analysis_reports/confusion_matrix.png')
plt.close()

# 6. Save Model
joblib.dump(clf, 'models/impact_model.pkl')
print("Model saved and analysis complete.")
