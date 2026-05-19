import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuration
MODELS_DIR = 'models'

print("=" * 70)
print("DEMONSTRATION: TUNING LOGISTIC REGRESSION AND DECISION TREE")
print("=" * 70)

# 1. Load Data
try:
    df = pd.read_csv('data/earthquake_preprocessed.csv')
    feature_config = joblib.load(os.path.join(MODELS_DIR, 'feature_config.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    feature_columns = feature_config['feature_columns']
    target_col = feature_config['target_column']
    
    X = df[feature_columns].values
    y = df[target_col].values
    X_scaled = scaler.transform(df[feature_columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- Model 1: Logistic Regression Tuning ---
print("\n" + "-" * 50)
print("Tuning Model 1: Logistic Regression")
print("-" * 50)

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [500, 1000]
}

print("Running GridSearchCV for Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
lr_grid.fit(X_train, y_train)

print(f"Best Parameters: {lr_grid.best_params_}")
print(f"Best CV Score:   {lr_grid.best_score_:.4f}")
print(f"Test Accuracy:   {accuracy_score(y_test, lr_grid.predict(X_test)):.4f}")

# --- Model 2: Decision Tree Tuning ---
print("\n" + "-" * 50)
print("Tuning Model 2: Decision Tree")
print("-" * 50)

dt_param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

print("Running GridSearchCV for Decision Tree...")
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
dt_grid.fit(X_train, y_train)

print(f"Best Parameters: {dt_grid.best_params_}")
print(f"Best CV Score:   {dt_grid.best_score_:.4f}")
print(f"Test Accuracy:   {accuracy_score(y_test, dt_grid.predict(X_test)):.4f}")

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
