import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import time
warnings.filterwarnings('ignore')


print("=" * 80)
print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("Sri Lanka Tourism Experience Prediction System")
print("=" * 80)



# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "─" * 80)
print("LOADING DATA")
print("─" * 80)

df = pd.read_csv('../data/processed_dataset.csv')
print(f"\n   Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

print("\n   Target Variable Distribution:")
print("   " + "-" * 35)
for cat in ['Excellent', 'Good', 'Average', 'Poor']:
    count = (df['experience_category'] == cat).sum()
    pct = count / len(df) * 100
    print(f"   {cat:<12}: {count:>4} ({pct:>5.1f}%)")



# =============================================================================
# PREPARE FEATURES AND TARGET
# =============================================================================

print("\n" + "─" * 80)
print("FEATURE PREPARATION")
print("─" * 80)

# Define feature columns (exclude identifiers and targets)
exclude_cols = ['destination', 'district', 'experience_score', 
                'experience_category', 'experience_class']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n   Number of features: {len(feature_cols)}")

# Prepare data
X = df[feature_cols].copy()
y_class = df['experience_class'].copy()    # For classification (0-3)
y_score = df['experience_score'].copy()    # For regression (0-1)

print(f"   Feature matrix shape: {X.shape}")



# =============================================================================
# TRAIN/VALIDATION/TEST SPLIT
# =============================================================================

print("\n" + "─" * 80)
print("DATA SPLITTING (60/20/20)")
print("─" * 80)

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp_class, y_test_class, y_temp_score, y_test_score = train_test_split(
    X, y_class, y_score,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# Second split: 75% train, 25% validation (of the 80%)
X_train, X_val, y_train_class, y_val_class, y_train_score, y_val_score = train_test_split(
    X_temp, y_temp_class, y_temp_score,
    test_size=0.25,
    random_state=42,
    stratify=y_temp_class
)

print(f"""
   DATA SPLIT SUMMARY:
   -------------------
   *  Training set:   {X_train.shape[0]:>4} samples ({X_train.shape[0]/len(X)*100:.1f}%)
   *  Validation set: {X_val.shape[0]:>4} samples ({X_val.shape[0]/len(X)*100:.1f}%)
   *  Test set:       {X_test.shape[0]:>4} samples ({X_test.shape[0]/len(X)*100:.1f}%)
   *  Total:          {len(X):>4} samples
""")



# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

print("\n" + "─" * 80)
print("HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("─" * 80)

# Define hyperparameter search space
param_distributions = {
    'learning_rate': uniform(0.01, 0.29),      # 0.01 to 0.30
    'max_iter': randint(100, 400),             # 100 to 400
    'max_depth': randint(3, 12),               # 3 to 12
    'min_samples_leaf': randint(2, 30),        # 2 to 30
    'l2_regularization': uniform(0.0, 1.0),    # 0.0 to 1.0
    'max_bins': [63, 127, 255],                # Common bin sizes
}

print("""
   HYPERPARAMETER SEARCH SPACE:
   ----------------------------
   *  learning_rate:     [0.01, 0.30] (uniform)
   *  max_iter:          [100, 400] (integer)
   *  max_depth:         [3, 12] (integer)
   *  min_samples_leaf:  [2, 30] (integer)
   *  l2_regularization: [0.0, 1.0] (uniform)
   *  max_bins:          [63, 127, 255]
""")

# Create base model
base_clf = HistGradientBoostingClassifier(
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=0
)

# Perform randomized search
print("   Running RandomizedSearchCV (50 iterations, 5-fold CV)...")
print("   This may take a few minutes...\n")

start_time = time.time()

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_distributions,
    n_iter=50,              # Number of parameter combinations to try
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1,
    return_train_score=True
)

random_search.fit(X_train, y_train_class)

tuning_time = time.time() - start_time
print(f"\n   * Hyperparameter tuning completed in {tuning_time:.1f} seconds")



# =============================================================================
# BEST HYPERPARAMETERS
# =============================================================================

print("\n" + "─" * 80)
print("BEST HYPERPARAMETERS FOUND")
print("─" * 80)

best_params = random_search.best_params_

print(f"""
   OPTIMAL HYPERPARAMETERS:
   ------------------------
   *  learning_rate:     {best_params['learning_rate']:.4f}
   *  max_iter:          {best_params['max_iter']}
   *  max_depth:         {best_params['max_depth']}
   *  min_samples_leaf:  {best_params['min_samples_leaf']}
   *  l2_regularization: {best_params['l2_regularization']:.4f}
   *  max_bins:          {best_params['max_bins']}
   
   Best CV Accuracy:     {random_search.best_score_:.4f}
""")

# Save hyperparameter tuning results
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df = cv_results_df.sort_values('rank_test_score')
cv_results_df.to_csv('../outputs/hyperparameter_tuning_results.csv', index=False)
print("   * Saved: ../outputs/hyperparameter_tuning_results.csv")



# =============================================================================
# TRAIN FINAL MODELS WITH BEST PARAMETERS
# =============================================================================

print("\n" + "─" * 80)
print("TRAINING FINAL MODELS")
print("─" * 80)

# Classification model with best parameters
print("\n   Training HistGradientBoostingClassifier with tuned parameters...")
clf = HistGradientBoostingClassifier(
    **best_params,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=0
)
clf.fit(X_train, y_train_class)
print(f"   * Classifier trained (iterations used: {clf.n_iter_})")

# Regression model with best parameters (adapted)
print("\n   Training HistGradientBoostingRegressor with tuned parameters...")
reg_params = {k: v for k, v in best_params.items()}
reg = HistGradientBoostingRegressor(
    **reg_params,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=0
)
reg.fit(X_train, y_train_score)
print(f"   * Regressor trained (iterations used: {reg.n_iter_})")



# =============================================================================
# MODEL EVALUATION
# =============================================================================

print("\n" + "─" * 80)
print("MODEL EVALUATION")
print("─" * 80)


# Calculate and display classification metrics
def evaluate_classification(y_true, y_pred, set_name):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    print(f"\n   {set_name} SET (Classification):")
    print("   " + "-" * 40)
    print(f"   *  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"   *  F1 Score (Macro):    {metrics['f1_macro']:.4f}")
    print(f"   *  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"   *  Precision:           {metrics['precision']:.4f}")
    print(f"   *  Recall:              {metrics['recall']:.4f}")
    
    return metrics

# Calculate and display regression metrics
def evaluate_regression(y_true, y_pred, set_name):
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    
    print(f"\n   {set_name} SET (Regression):")
    print("   " + "-" * 40)
    print(f"   *  R² Score:            {metrics['r2']:.4f}")
    print(f"   *  RMSE:                {metrics['rmse']:.4f}")
    print(f"   *  MAE:                 {metrics['mae']:.4f}")
    
    return metrics


# Make predictions
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

y_train_pred_reg = reg.predict(X_train)
y_val_pred_reg = reg.predict(X_val)
y_test_pred_reg = reg.predict(X_test)

# Evaluate
train_clf_metrics = evaluate_classification(y_train_class, y_train_pred, "TRAINING")
val_clf_metrics = evaluate_classification(y_val_class, y_val_pred, "VALIDATION")
test_clf_metrics = evaluate_classification(y_test_class, y_test_pred, "TEST")

train_reg_metrics = evaluate_regression(y_train_score, y_train_pred_reg, "TRAINING")
val_reg_metrics = evaluate_regression(y_val_score, y_val_pred_reg, "VALIDATION")
test_reg_metrics = evaluate_regression(y_test_score, y_test_pred_reg, "TEST")

# Cross-validation on full training data
print("\n   CROSS-VALIDATION (5-Fold Stratified):")
print("   " + "-" * 40)
cv_scores = cross_val_score(clf, X_temp, y_temp_class, cv=5, scoring='accuracy')
print(f"   *  Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   *  Mean Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std()*2:.4f})")

# Detailed classification report
print("\n" + "─" * 80)
print("   DETAILED CLASSIFICATION REPORT (Test Set)")
print("─" * 80)
class_labels = ['Poor', 'Average', 'Good', 'Excellent']
print(classification_report(y_test_class, y_test_pred, target_names=class_labels, zero_division=0))



# =============================================================================
# SAVE MODELS AND RESULTS
# =============================================================================

print("\n" + "─" * 80)
print("SAVING MODELS AND ARTIFACTS")
print("─" * 80)

# Save models
with open('../models/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("   * Saved: ../models/classifier.pkl")

with open('../models/regressor.pkl', 'wb') as f:
    pickle.dump(reg, f)
print("   * Saved: ../models/regressor.pkl")

with open('../models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("   * Saved: ../models/feature_names.pkl")

# Save best hyperparameters
with open('../models/best_hyperparameters.pkl', 'wb') as f:
    pickle.dump(best_params, f)
print("   * Saved: ../models/best_hyperparameters.pkl")

# Save metrics
metrics_summary = {
    'classification': {
        'train': train_clf_metrics,
        'validation': val_clf_metrics,
        'test': test_clf_metrics,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    },
    'regression': {
        'train': train_reg_metrics,
        'validation': val_reg_metrics,
        'test': test_reg_metrics
    },
    'hyperparameters': best_params,
    'tuning_time_seconds': tuning_time
}

with open('../models/metrics.pkl', 'wb') as f:
    pickle.dump(metrics_summary, f)
print("   * Saved: ../models/metrics.pkl")



# =============================================================================
# CREATE VISUALIZATIONS
# =============================================================================

print("\n" + "─" * 80)
print("GENERATING VISUALIZATIONS")
print("─" * 80)

plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Confusion Matrix
print("\n   Creating confusion matrix plots...")
cm = confusion_matrix(y_test_class, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].set_ylabel('Actual Label', fontsize=11)
axes[0].set_title('Confusion Matrix (Test Set)\nRaw Counts', fontsize=12)

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels, ax=axes[1],
            cbar_kws={'label': 'Percentage'})
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].set_ylabel('Actual Label', fontsize=11)
axes[1].set_title('Confusion Matrix (Test Set)\nNormalized', fontsize=12)

plt.tight_layout()
plt.savefig('../outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   * Saved: ../outputs/confusion_matrix.png")
plt.close()


# Figure 2: Performance Metrics Comparison
print("   Creating performance comparison plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classification metrics
metrics_plot = pd.DataFrame({
    'Train': [train_clf_metrics['accuracy'], train_clf_metrics['f1_weighted'], 
              train_clf_metrics['precision'], train_clf_metrics['recall']],
    'Validation': [val_clf_metrics['accuracy'], val_clf_metrics['f1_weighted'], 
                   val_clf_metrics['precision'], val_clf_metrics['recall']],
    'Test': [test_clf_metrics['accuracy'], test_clf_metrics['f1_weighted'], 
             test_clf_metrics['precision'], test_clf_metrics['recall']]
}, index=['Accuracy', 'F1 Score', 'Precision', 'Recall'])

metrics_plot.plot(kind='bar', ax=axes[0], color=['#27ae60', '#3498db', '#e74c3c'],
                  edgecolor='white', width=0.8)
axes[0].set_title('Classification Performance', fontsize=12)
axes[0].set_ylabel('Score', fontsize=11)
axes[0].set_ylim(0, 1.15)
axes[0].legend(loc='upper right')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.3f', fontsize=8, padding=2)

# Regression metrics
reg_plot = pd.DataFrame({
    'Train': [train_reg_metrics['rmse'], train_reg_metrics['mae']],
    'Validation': [val_reg_metrics['rmse'], val_reg_metrics['mae']],
    'Test': [test_reg_metrics['rmse'], test_reg_metrics['mae']]
}, index=['RMSE', 'MAE'])

reg_plot.plot(kind='bar', ax=axes[1], color=['#27ae60', '#3498db', '#e74c3c'],
              edgecolor='white', width=0.8)
axes[1].set_title('Regression Performance', fontsize=12)
axes[1].set_ylabel('Error', fontsize=11)
axes[1].legend(loc='upper right')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.3f', fontsize=8, padding=2)

plt.tight_layout()
plt.savefig('../outputs/model_performance.png', dpi=150, bbox_inches='tight')
print("   * Saved: ../outputs/model_performance.png")
plt.close()


# Figure 3: Actual vs Predicted (Regression)
print("   Creating regression scatter plot...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test_score, y_test_pred_reg, alpha=0.6, c='#3498db',
           edgecolors='white', s=80, label='Predictions')
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
ax.fill_between([0, 1], [0-0.1, 1-0.1], [0+0.1, 1+0.1], alpha=0.1, color='green',
                label='±0.1 Error Band')
ax.set_xlabel('Actual Experience Score', fontsize=12)
ax.set_ylabel('Predicted Experience Score', fontsize=12)
ax.set_title(f'Actual vs Predicted\n(Test Set, R² = {test_reg_metrics["r2"]:.4f})', fontsize=14)
ax.legend(loc='upper left')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("   * Saved: ../outputs/actual_vs_predicted.png")
plt.close()


# Figure 4: Cross-Validation Scores
print("   Creating cross-validation plot...")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(1, 6), cv_scores, color='#3498db', edgecolor='white', alpha=0.8)
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', lw=2,
           label=f'Mean: {cv_scores.mean():.4f}')
ax.fill_between([0, 6], cv_scores.mean() - cv_scores.std(), cv_scores.mean() + cv_scores.std(),
                alpha=0.2, color='red', label=f'±1 Std: {cv_scores.std():.4f}')
ax.set_xlabel('Fold Number', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('5-Fold Stratified Cross-Validation', fontsize=14)
ax.set_xticks(range(1, 6))
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')

for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{score:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('../outputs/cross_validation.png', dpi=150, bbox_inches='tight')
print("   * Saved: ../outputs/cross_validation.png")
plt.close()


# Figure 5: Hyperparameter Tuning Progress
print("   Creating hyperparameter tuning visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top 20 parameter combinations
top_20 = cv_results_df.head(20)
axes[0].barh(range(20), top_20['mean_test_score'], color='#3498db', alpha=0.8)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels([f"Config {i+1}" for i in range(20)])
axes[0].set_xlabel('Mean CV Accuracy', fontsize=11)
axes[0].set_title('Top 20 Hyperparameter Configurations', fontsize=12)
axes[0].invert_yaxis()

# Learning rate vs accuracy
axes[1].scatter(cv_results_df['param_learning_rate'], 
                cv_results_df['mean_test_score'],
                c=cv_results_df['param_max_depth'], cmap='viridis',
                alpha=0.6, s=50)
axes[1].set_xlabel('Learning Rate', fontsize=11)
axes[1].set_ylabel('Mean CV Accuracy', fontsize=11)
axes[1].set_title('Learning Rate vs Accuracy\n(Color = Max Depth)', fontsize=12)
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Max Depth')

plt.tight_layout()
plt.savefig('../outputs/hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
print("   * Saved: ../outputs/hyperparameter_tuning.png")
plt.close()



# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)

print(f"""
MODEL RESULTS SUMMARY
=====================

Algorithm: HistGradientBoostingClassifier
-----------------------------------------
*  Uses histogram-based gradient boosting
*  Native support for missing values
*  Efficient for medium-sized datasets

Hyperparameter Tuning:
----------------------
*  Method: RandomizedSearchCV
*  Iterations: 50
*  CV Folds: 5
*  Tuning Time: {tuning_time:.1f} seconds

Best Hyperparameters:
---------------------
*  learning_rate:     {best_params['learning_rate']:.4f}
*  max_iter:          {best_params['max_iter']}
*  max_depth:         {best_params['max_depth']}
*  min_samples_leaf:  {best_params['min_samples_leaf']}
*  l2_regularization: {best_params['l2_regularization']:.4f}
*  max_bins:          {best_params['max_bins']}

Classification Results (Test Set):
----------------------------------
*  Accuracy:     {test_clf_metrics['accuracy']:.4f} ({test_clf_metrics['accuracy']*100:.2f}%)
*  F1 Score:     {test_clf_metrics['f1_weighted']:.4f}
*  Precision:    {test_clf_metrics['precision']:.4f}
*  Recall:       {test_clf_metrics['recall']:.4f}

Regression Results (Test Set):
------------------------------
*  R² Score:     {test_reg_metrics['r2']:.4f}
*  RMSE:         {test_reg_metrics['rmse']:.4f}
*  MAE:          {test_reg_metrics['mae']:.4f}

Cross-Validation:
-----------------
*  Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})
""")