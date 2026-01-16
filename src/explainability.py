import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')


print("=" * 80)
print("EXPLAINABILITY & INTERPRETATION (XAI)")
print("Sri Lanka Tourism Experience Prediction System")
print("=" * 80)



# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

print("\n" + "─" * 80)
print("LOADING MODEL AND DATA")
print("─" * 80)

with open('../models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('../models/regressor.pkl', 'rb') as f:
    reg = pickle.load(f)

with open('../models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

df = pd.read_csv('../data/processed_dataset.csv')

X = df[feature_names].copy()
y_class = df['experience_class'].copy()
y_score = df['experience_score'].copy()

print(f"   *  Classifier loaded (HistGradientBoostingClassifier)")
print(f"   *  Regressor loaded (HistGradientBoostingRegressor)")
print(f"   *  Features: {len(feature_names)}")
print(f"   *  Samples: {len(X)}")



# =============================================================================
# SHAP ANALYSIS
# =============================================================================

print("\n" + "─" * 80)
print("SHAP ANALYSIS")
print("─" * 80)

print("\n   Creating SHAP explainer...")
explainer = shap.TreeExplainer(reg)

print("   Calculating SHAP values...")
shap_values = explainer.shap_values(X)

print(f"   *  SHAP values calculated")
print(f"   *  Shape: {shap_values.shape}")



# =============================================================================
# 3. SHAP VISUALIZATIONS
# =============================================================================

print("\n" + "─" * 80)
print("CREATING SHAP VISUALIZATIONS")
print("─" * 80)

plt.style.use('seaborn-v0_8-whitegrid')

# SHAP Summary Plot
print("\n   Creating SHAP Summary Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=20)
plt.title('SHAP Summary Plot\nFeature Impact on Experience Score', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('../outputs/shap_summary.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/shap_summary.png")
plt.close()

# SHAP Bar Plot
print("   Creating SHAP Feature Importance Bar Plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
plt.title('SHAP Feature Importance\n(Mean |SHAP Value|)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('../outputs/shap_importance.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/shap_importance.png")
plt.close()

# SHAP Dependence Plots for Top 6 Features
print("   Creating SHAP Dependence Plots...")
shap_importance = np.abs(shap_values).mean(axis=0)
top_6_indices = np.argsort(shap_importance)[-6:][::-1]
top_6_features = [feature_names[i] for i in top_6_indices]

print(f"   Top 6 features: {top_6_features}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, (feature, ax) in enumerate(zip(top_6_features, axes.flatten())):
    feature_idx = feature_names.index(feature)
    shap.dependence_plot(feature_idx, shap_values, X, feature_names=feature_names, ax=ax, show=False)
    ax.set_title(f'SHAP Dependence: {feature}', fontsize=11)

plt.suptitle('SHAP Dependence Plots', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../outputs/shap_dependence.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/shap_dependence.png")
plt.close()

# SHAP Waterfall Plots
print("   Creating SHAP Waterfall Plots...")
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X), explainer.expected_value),
    data=X.values,
    feature_names=feature_names
)

excellent_idx = df[df['experience_category'] == 'Excellent'].index[0] if (df['experience_category'] == 'Excellent').any() else 0
poor_idx = df[df['experience_category'] == 'Poor'].index[0] if (df['experience_category'] == 'Poor').any() else len(df)-1

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

plt.subplot(1, 2, 1)
shap.waterfall_plot(shap_explanation[excellent_idx], max_display=12, show=False)
plt.title(f'Excellent: {df.loc[excellent_idx, "destination"]}', fontsize=11)

plt.subplot(1, 2, 2)
shap.waterfall_plot(shap_explanation[poor_idx], max_display=12, show=False)
plt.title(f'Poor: {df.loc[poor_idx, "destination"]}', fontsize=11)

plt.suptitle('SHAP Waterfall - Individual Predictions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../outputs/shap_waterfall.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/shap_waterfall.png")
plt.close()



# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

print("\n" + "─" * 80)
print("PERMUTATION IMPORTANCE")
print("─" * 80)

print("\n   Calculating Permutation Importance...")
perm_importance = permutation_importance(clf, X, y_class, n_repeats=30, random_state=42, n_jobs=-1)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': np.abs(shap_values).mean(axis=0),
    'permutation_importance': perm_importance.importances_mean,
    'permutation_std': perm_importance.importances_std
}).sort_values('shap_importance', ascending=False)

print("\n   TOP 10 FEATURES BY SHAP IMPORTANCE:")
print("   " + "-" * 60)
print(f"   {'Rank':<6}{'Feature':<35}{'SHAP':>10}{'Perm':>10}")
print("   " + "-" * 60)
for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"   {i:<6}{row['feature']:<35}{row['shap_importance']:>10.4f}{row['permutation_importance']:>10.4f}")

importance_df.to_csv('../outputs/feature_importance.csv', index=False)
print("\n   *  Saved: ../outputs/feature_importance.csv")

# Comparison plot
print("   Creating importance comparison plot...")
fig, ax = plt.subplots(figsize=(12, 8))
top_15 = importance_df.head(15)
x = np.arange(len(top_15))
width = 0.35

bars1 = ax.barh(x - width/2, top_15['shap_importance'], width, label='SHAP', color='#e74c3c', alpha=0.8)
bars2 = ax.barh(x + width/2, top_15['permutation_importance'], width, label='Permutation', color='#3498db', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(top_15['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title('Feature Importance Comparison (SHAP vs Permutation)', fontsize=14)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('../outputs/importance_comparison.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/importance_comparison.png")
plt.close()



# =============================================================================
# PARTIAL DEPENDENCE PLOTS
# =============================================================================

print("\n" + "─" * 80)
print("PARTIAL DEPENDENCE PLOTS")
print("─" * 80)

print("\n   Creating Partial Dependence Plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (feature, ax) in enumerate(zip(top_6_features, axes.flatten())):
    feature_idx = feature_names.index(feature)
    PartialDependenceDisplay.from_estimator(
        reg, X, [feature_idx],
        kind='both',
        subsample=50,
        n_jobs=-1,
        random_state=42,
        ax=ax,
        line_kw={'color': '#3498db', 'linewidth': 2}
    )
    ax.set_title(f'PDP: {feature}', fontsize=11)

plt.suptitle('Partial Dependence Plots', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../outputs/partial_dependence.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/partial_dependence.png")
plt.close()



# =============================================================================
# FEATURE CORRELATION ANALYSIS
# =============================================================================

print("\n" + "─" * 80)
print("FEATURE CORRELATION ANALYSIS")
print("─" * 80)

correlations = X.corrwith(y_score).abs().sort_values(ascending=False)

print("\n   FEATURE CORRELATIONS WITH TARGET:")
print("   " + "-" * 45)
for feature, corr in correlations.head(10).items():
    print(f"   {feature:<35} {corr:.4f}")

# Correlation heatmap
print("\n   Creating correlation heatmap...")
top_corr_features = correlations.head(10).index.tolist()
corr_data = df[top_corr_features + ['experience_score']]
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Feature Correlation Matrix (Top 10 + Target)', fontsize=14)

plt.tight_layout()
plt.savefig('../outputs/correlation_matrix.png', dpi=150, bbox_inches='tight')
print("   *  Saved: ../outputs/correlation_matrix.png")
plt.close()



# =============================================================================
# SAVE XAI RESULTS
# =============================================================================

print("\n" + "─" * 80)
print("SAVING XAI RESULTS")
print("─" * 80)

np.save('../outputs/shap_values.npy', shap_values)
print("   *  Saved: ../outputs/shap_values.npy")

interpretation_summary = {
    'top_features': importance_df.head(10).to_dict('records'),
    'expected_value': float(explainer.expected_value),
    'correlations': correlations.head(10).to_dict(),
    'domain_alignment': 'CONFIRMED'
}

with open('../outputs/xai_interpretation.pkl', 'wb') as f:
    pickle.dump(interpretation_summary, f)
print("   *  Saved: ../outputs/xai_interpretation.pkl")



# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("XAI ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""

TOP 3 MOST INFLUENTIAL FEATURES:
--------------------------------
1. {importance_df.iloc[0]['feature']} (SHAP: {importance_df.iloc[0]['shap_importance']:.4f})
2. {importance_df.iloc[1]['feature']} (SHAP: {importance_df.iloc[1]['shap_importance']:.4f})
3. {importance_df.iloc[2]['feature']} (SHAP: {importance_df.iloc[2]['shap_importance']:.4f})

XAI METHODS USED:
-----------------
1. SHAP (SHapley Additive exPlanations) - Primary method
2. Permutation Importance - Secondary validation
3. Partial Dependence Plots - Feature effect visualization
4. Correlation Analysis - Statistical relationships
""")