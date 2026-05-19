"""
Milestone 2 - Weeks 3 & 4: Model Training & Evaluation
ImpactSense - Earthquake Impact Prediction

Week 3: Baseline Models (Logistic Regression, Decision Tree)
Week 4: Advanced Models (Random Forest, Gradient Boosting, XGBoost)
        Cross-validation, Hyperparameter Tuning, Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("[WARNING] XGBoost not installed. Skipping XGBoost model.")
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("[WARNING] SHAP not installed. Skipping SHAP analysis.")
    HAS_SHAP = False

# ============================================================
# Configuration
# ============================================================
REPORTS_DIR = 'analysis_reports'
MODELS_DIR = 'models'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set(style="whitegrid")

# ============================================================
# 1. Load Preprocessed Data
# ============================================================
print("=" * 70)
print("MILESTONE 2: MODEL TRAINING & EVALUATION")
print("=" * 70)

try:
    df = pd.read_csv('data/earthquake_preprocessed.csv')
    feature_config = joblib.load(os.path.join(MODELS_DIR, 'feature_config.pkl'))
    print(f"\n[OK] Preprocessed data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("[ERROR] Preprocessed data not found. Run preprocessing.py first.")
    exit(1)

feature_columns = [col for col in feature_config['feature_columns'] if 'soil' not in col.lower()]
target_col = feature_config['target_column']

print(f"  Features ({len(feature_columns)}): {feature_columns}")
print(f"  Target: {target_col}")

# ============================================================
# 2. Prepare Data
# ============================================================
X = df[feature_columns].values
y = df[target_col].values

# Load the scaler fitted during preprocessing
# Fit a new scaler on the filtered features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train set: {X_train.shape[0]} samples")
print(f"  Test set:  {X_test.shape[0]} samples")
print(f"  Class distribution (train): {np.bincount(y_train)}")
print(f"  Class distribution (test):  {np.bincount(y_test)}")

# ============================================================
# WEEK 3: Baseline Models
# ============================================================
print("\n" + "=" * 70)
print("WEEK 3: BASELINE MODEL TRAINING")
print("=" * 70)

results = {}

# --- Model 1: Logistic Regression (with GridSearchCV) ---
print("\n" + "-" * 50)
print("Model 1: Logistic Regression (with GridSearchCV)")
print("-" * 50)

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [500, 1000]
}

print("  Running GridSearchCV...")
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=0
)
lr_grid.fit(X_train, y_train)

lr_model = lr_grid.best_estimator_
lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

results['Logistic Regression'] = {
    'model': lr_model, 'accuracy': lr_accuracy,
    'precision': lr_precision, 'recall': lr_recall,
    'f1': lr_f1, 'predictions': lr_pred
}

print(f"  Best Parameters: {lr_grid.best_params_}")
print(f"  Best CV Score:   {lr_grid.best_score_ * 100:.2f}%")
print(f"  Accuracy:  {lr_accuracy * 100:.2f}%")
print(f"  Precision: {lr_precision * 100:.2f}%")
print(f"  Recall:    {lr_recall * 100:.2f}%")
print(f"  F1-Score:  {lr_f1 * 100:.2f}%")
print(f"\n  Classification Report:\n{classification_report(y_test, lr_pred)}")

# --- Model 2: Decision Tree (with GridSearchCV) ---
print("\n" + "-" * 50)
print("Model 2: Decision Tree (with GridSearchCV)")
print("-" * 50)

dt_param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

print("  Running GridSearchCV...")
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=0
)
dt_grid.fit(X_train, y_train)

dt_model = dt_grid.best_estimator_
dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

results['Decision Tree'] = {
    'model': dt_model, 'accuracy': dt_accuracy,
    'precision': dt_precision, 'recall': dt_recall,
    'f1': dt_f1, 'predictions': dt_pred
}

print(f"  Best Parameters: {dt_grid.best_params_}")
print(f"  Best CV Score:   {dt_grid.best_score_ * 100:.2f}%")
print(f"  Accuracy:  {dt_accuracy * 100:.2f}%")
print(f"  Precision: {dt_precision * 100:.2f}%")
print(f"  Recall:    {dt_recall * 100:.2f}%")
print(f"  F1-Score:  {dt_f1 * 100:.2f}%")
print(f"\n  Classification Report:\n{classification_report(y_test, dt_pred)}")

# ============================================================
# WEEK 4: Advanced Model Training
# ============================================================
print("\n" + "=" * 70)
print("WEEK 4: ADVANCED MODEL TRAINING")
print("=" * 70)

# --- Model 3: Random Forest with Hyperparameter Tuning ---
print("\n" + "-" * 50)
print("Model 3: Random Forest (with GridSearchCV)")
print("-" * 50)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("  Running GridSearchCV (this may take a minute)...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

results['Random Forest'] = {
    'model': rf_model, 'accuracy': rf_accuracy,
    'precision': rf_precision, 'recall': rf_recall,
    'f1': rf_f1, 'predictions': rf_pred
}

print(f"  Best Parameters: {rf_grid.best_params_}")
print(f"  Best CV Score:   {rf_grid.best_score_ * 100:.2f}%")
print(f"  Accuracy:  {rf_accuracy * 100:.2f}%")
print(f"  Precision: {rf_precision * 100:.2f}%")
print(f"  Recall:    {rf_recall * 100:.2f}%")
print(f"  F1-Score:  {rf_f1 * 100:.2f}%")
print(f"\n  Classification Report:\n{classification_report(y_test, rf_pred)}")

# --- Model 4: Gradient Boosting ---
print("\n" + "-" * 50)
print("Model 4: Gradient Boosting")
print("-" * 50)

gb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

print("  Running GridSearchCV...")
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=0
)
gb_grid.fit(X_train, y_train)

gb_model = gb_grid.best_estimator_
gb_pred = gb_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

results['Gradient Boosting'] = {
    'model': gb_model, 'accuracy': gb_accuracy,
    'precision': gb_precision, 'recall': gb_recall,
    'f1': gb_f1, 'predictions': gb_pred
}

print(f"  Best Parameters: {gb_grid.best_params_}")
print(f"  Best CV Score:   {gb_grid.best_score_ * 100:.2f}%")
print(f"  Accuracy:  {gb_accuracy * 100:.2f}%")
print(f"  Precision: {gb_precision * 100:.2f}%")
print(f"  Recall:    {gb_recall * 100:.2f}%")
print(f"  F1-Score:  {gb_f1 * 100:.2f}%")
print(f"\n  Classification Report:\n{classification_report(y_test, gb_pred)}")

# --- Model 5: XGBoost ---
if HAS_XGBOOST:
    print("\n" + "-" * 50)
    print("Model 5: XGBoost")
    print("-" * 50)

    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    print("  Running GridSearchCV...")
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        xgb_param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X_train, y_train)

    xgb_model = xgb_grid.best_estimator_
    xgb_pred = xgb_model.predict(X_test)

    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)

    results['XGBoost'] = {
        'model': xgb_model, 'accuracy': xgb_accuracy,
        'precision': xgb_precision, 'recall': xgb_recall,
        'f1': xgb_f1, 'predictions': xgb_pred
    }

    print(f"  Best Parameters: {xgb_grid.best_params_}")
    print(f"  Best CV Score:   {xgb_grid.best_score_ * 100:.2f}%")
    print(f"  Accuracy:  {xgb_accuracy * 100:.2f}%")
    print(f"  Precision: {xgb_precision * 100:.2f}%")
    print(f"  Recall:    {xgb_recall * 100:.2f}%")
    print(f"  F1-Score:  {xgb_f1 * 100:.2f}%")
    print(f"\n  Classification Report:\n{classification_report(y_test, xgb_pred)}")

# ============================================================
# 3. Model Comparison
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1']
    }
    for name, r in results.items()
}).T

comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
print("\n", comparison_df.to_string())

# Find best model
best_model_name = comparison_df.index[0]
best_model = results[best_model_name]['model']
print(f"\n  Best Model: {best_model_name} (Accuracy: {comparison_df.iloc[0]['Accuracy'] * 100:.2f}%)")

# ============================================================
# 4. Visualizations
# ============================================================
print("\n" + "=" * 70)
print("GENERATING EVALUATION VISUALIZATIONS")
print("=" * 70)

# --- Plot 1: Model Comparison Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy Comparison
metrics_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in metrics_names]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

bars = axes[0].barh(metrics_names, accuracies, color=colors[:len(metrics_names)], edgecolor='black')
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 1)
for bar, acc in zip(bars, accuracies):
    axes[0].text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc * 100:.2f}%',
                 va='center', fontweight='bold')

# F1 Score Comparison
f1_scores = [results[m]['f1'] for m in metrics_names]
bars = axes[1].barh(metrics_names, f1_scores, color=colors[:len(metrics_names)], edgecolor='black')
axes[1].set_xlabel('F1 Score')
axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 1)
for bar, f1 in zip(bars, f1_scores):
    axes[1].text(f1 + 0.01, bar.get_y() + bar.get_height()/2, f'{f1 * 100:.2f}%',
                 va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: analysis_reports/model_comparison.png")

# --- Plot 2: Confusion Matrices for All Models ---
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
if n_models == 1:
    axes = [axes]

for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    ax.set_title(f'{name}\nAcc: {r["accuracy"] * 100:.2f}%', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'all_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: analysis_reports/all_confusion_matrices.png")

# --- Plot 3: ROC Curves ---
plt.figure(figsize=(10, 8))
for name, r in results.items():
    model = r['model']
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'roc_curves.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/roc_curves.png")

# --- Plot 4: Feature Importance (Best Model) ---
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    importances = best_model.feature_importances_
    feature_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=True)

    colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_imp)))
    feature_imp.plot(kind='barh', color=colors_imp, edgecolor='black')
    plt.title(f'Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    print("  Saved: analysis_reports/feature_importance.png")

    print(f"\n  Top 5 Most Important Features ({best_model_name}):")
    for feat, imp in feature_imp.sort_values(ascending=False).head(5).items():
        print(f"    {feat}: {imp:.4f}")

# --- Plot 5: Learning Curves (Best Model) ---
print("\n  Generating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_scaled, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(10, 6))
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', linewidth=2, label='Training Accuracy')
plt.plot(train_sizes, val_mean, 'o-', color='orange', linewidth=2, label='Validation Accuracy')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'Learning Curves ({best_model_name})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'learning_curves.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/learning_curves.png")

# --- Plot 6: Cross-Validation Scores Box Plot ---
print("\n  Computing cross-validation scores for all models...")
cv_results = {}
for name, r in results.items():
    cv_scores = cross_val_score(r['model'], X_scaled, y, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores
    print(f"    {name}: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")

plt.figure(figsize=(10, 6))
cv_df = pd.DataFrame(cv_results)
cv_df.boxplot(vert=True, patch_artist=True)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Cross-Validation Accuracy Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'cv_boxplot.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/cv_boxplot.png")

# ============================================================
# 5. SHAP Analysis (if available)
# ============================================================
if HAS_SHAP and hasattr(best_model, 'feature_importances_'):
    print("\n" + "-" * 50)
    print("SHAP Feature Importance Analysis")
    print("-" * 50)

    try:
        # Use a sample for SHAP to avoid long computation
        X_sample = X_test[:200]
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_sample,
                              feature_names=feature_columns, show=False)
        else:
            shap.summary_plot(shap_values, X_sample,
                              feature_names=feature_columns, show=False)
        plt.title(f'SHAP Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'shap_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: analysis_reports/shap_importance.png")
    except Exception as e:
        print(f"  [WARNING] SHAP analysis failed: {e}")

# ============================================================
# 6. Regression Metrics (on Risk_Score)
# ============================================================
print("\n" + "=" * 70)
print("REGRESSION METRICS (Risk Score Prediction)")
print("=" * 70)

# Train best model type on Risk_Score regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

y_reg = df['Risk_Score'].values
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

reg_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred_r)
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\n  Gradient Boosting Regressor:")
print(f"  MAE:  {mae:.4f}")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# Regression visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
axes[0].scatter(y_test_r, y_pred_r, alpha=0.3, s=10, color='steelblue')
axes[0].plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Risk Score')
axes[0].set_ylabel('Predicted Risk Score')
axes[0].set_title(f'Actual vs Predicted Risk Score\nR² = {r2:.4f}', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual Plot
residuals = y_test_r - y_pred_r
axes[1].scatter(y_pred_r, residuals, alpha=0.3, s=10, color='coral')
axes[1].axhline(y=0, color='black', linewidth=1.5)
axes[1].set_xlabel('Predicted Risk Score')
axes[1].set_ylabel('Residuals')
axes[1].set_title(f'Residual Plot\nMAE = {mae:.4f}, RMSE = {rmse:.4f}', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'regression_results.png'), dpi=150)
plt.close()
print("  Saved: analysis_reports/regression_results.png")

# Save regression model
joblib.dump(reg_model, os.path.join(MODELS_DIR, 'risk_regressor.pkl'))

# ============================================================
# 7. Save Best Classification Model & Full Report
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODELS & REPORT")
print("=" * 70)

# Save best model
joblib.dump(best_model, os.path.join(MODELS_DIR, 'impact_model.pkl'))
print(f"  Best model ({best_model_name}) saved to models/impact_model.pkl")

# Save all models
for name, r in results.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(r['model'], os.path.join(MODELS_DIR, f'{safe_name}_model.pkl'))
    print(f"  {name} model saved to models/{safe_name}_model.pkl")

# Generate comprehensive report
report_path = os.path.join(REPORTS_DIR, 'model_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ImpactSense - Model Training Report\n")
    f.write("Milestone 2: Weeks 3 & 4\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET INFO\n")
    f.write(f"  Total Samples: {len(df)}\n")
    f.write(f"  Features: {len(feature_columns)}\n")
    f.write(f"  Train/Test Split: 80/20\n\n")

    f.write("CLASSIFICATION RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
    f.write("-" * 70 + "\n")
    for name, r in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        f.write(f"{name:<25} {r['accuracy'] * 100:>10.2f}% {r['precision'] * 100:>10.2f}% "
                f"{r['recall'] * 100:>10.2f}% {r['f1'] * 100:>10.2f}%\n")
    f.write("-" * 70 + "\n")
    f.write(f"\nBest Model: {best_model_name}\n\n")

    f.write("REGRESSION RESULTS (Risk Score)\n")
    f.write(f"  Model: Gradient Boosting Regressor\n")
    f.write(f"  MAE:  {mae:.4f}\n")
    f.write(f"  MSE:  {mse:.4f}\n")
    f.write(f"  RMSE: {rmse:.4f}\n")
    f.write(f"  R²:   {r2:.4f}\n\n")

    if hasattr(best_model, 'feature_importances_'):
        f.write("TOP FEATURES (by importance):\n")
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_columns))):
            f.write(f"  {i+1}. {feature_columns[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}\n")

    f.write("\nCROSS-VALIDATION RESULTS (5-fold)\n")
    for name, scores in cv_results.items():
        f.write(f"  {name}: {scores.mean() * 100:.2f}% (+/- {scores.std()*200:.2f}%)\n")

    f.write("\n\nDETAILED CLASSIFICATION REPORTS\n")
    f.write("=" * 70 + "\n")
    for name, r in results.items():
        f.write(f"\n{name}:\n")
        f.write(classification_report(y_test, r['predictions']))
        f.write("\n")

print(f"  Full report saved to {report_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("MILESTONE 2 COMPLETE - FINAL SUMMARY")
print("=" * 70)
print(f"\n  Models Trained: {len(results)}")
print(f"  Best Model: {best_model_name}")
print(f"  Best Accuracy: {results[best_model_name]['accuracy'] * 100:.2f}%")
print(f"  Best F1-Score: {results[best_model_name]['f1'] * 100:.2f}%")
print(f"\n  Regression (Risk Score):")
print(f"    R² Score: {r2:.4f}")
print(f"    MAE: {mae:.4f}")
print(f"\n  Saved Visualizations:")
print(f"    - model_comparison.png")
print(f"    - all_confusion_matrices.png")
print(f"    - roc_curves.png")
print(f"    - feature_importance.png")
print(f"    - learning_curves.png")
print(f"    - cv_boxplot.png")
print(f"    - regression_results.png")
if HAS_SHAP:
    print(f"    - shap_importance.png")
print(f"    - model_report.txt")
print("=" * 70)
