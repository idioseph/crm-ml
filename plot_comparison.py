import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('test_beta2\\metrics.csv')

xgb_models = [
    'CRM deterministic',
    'CRM best-fitting draw',
    'Bayesian CRM parameters only',
    'Bayesian CRM predictive',
    'XGB only (no CRM)',
    'CRM-XGB parameters only',
    'CRM-XGB predictive'
]

df_xgb = df[df['model'].isin(xgb_models)].copy()
hist = df_xgb[df_xgb['period'] == 'history']
hold = df_xgb[df_xgb['period'] == 'holdout']

print("=== HISTORY SUMMARY (CRM & XGB MODELS ONLY) ===")
print(hist.groupby('model')[['MAE', 'RMSE', 'CRPS', 'PICP_80', 'PICP_95', 'IS_95']].agg(['mean', 'median', 'min', 'max', 'std']).round(3).to_string())

print("\n=== HOLDOUT SUMMARY (CRM & XGB MODELS ONLY) ===")
print(hold.groupby('model')[['MAE', 'RMSE', 'CRPS', 'PICP_80', 'PICP_95', 'IS_95']].agg(['mean', 'median', 'min', 'max', 'std']).round(3).to_string())

# Generate clean 4-panel figure for Thesis (CRM vs CRM-XGB only)
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

palette_dict = {
    'CRM deterministic': '#4c72b0',
    'CRM best-fitting draw': '#8172b3',
    'Bayesian CRM parameters only': '#dd8452',
    'Bayesian CRM predictive': '#e19d53',
    'XGB only (no CRM)': '#55a868',
    'CRM-XGB parameters only': '#c44e52',
    'CRM-XGB predictive': '#8c1d40'
}

# 1. History MAE
sns.barplot(data=hist, x='producer', y='MAE', hue='model', hue_order=xgb_models, ax=axes[0, 0], palette=palette_dict)
axes[0, 0].set_title('History Period: MAE by Well (CRM vs. XGBoost Models)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('MAE (MSTB/month)')
axes[0, 0].set_xlabel('')
axes[0, 0].tick_params(axis='x', rotation=30)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].legend(fontsize=8.5, loc='upper left')

# 2. Holdout MAE
sns.barplot(data=hold, x='producer', y='MAE', hue='model', hue_order=xgb_models, ax=axes[0, 1], palette=palette_dict)
axes[0, 1].set_title('Holdout Period: MAE by Well (CRM vs. XGBoost Models)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('MAE (MSTB/month)')
axes[0, 1].set_xlabel('')
axes[0, 1].tick_params(axis='x', rotation=30)
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].get_legend().remove()

# Probabilistic models subset
prob_xgb_models = [
    'Bayesian CRM parameters only',
    'Bayesian CRM predictive',
    'CRM-XGB parameters only',
    'CRM-XGB predictive'
]

# 3. Holdout CRPS
sns.barplot(data=hold[hold['model'].isin(prob_xgb_models)], x='producer', y='CRPS', hue='model', hue_order=prob_xgb_models, ax=axes[1, 0], palette=palette_dict)
axes[1, 0].set_title('Holdout Period: CRPS by Well (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('CRPS (MSTB/month)')
axes[1, 0].set_xlabel('Producer Well')
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].legend(fontsize=8.5, loc='upper left')

# 4. Holdout PICP_95
sns.barplot(data=hold[hold['model'].isin(prob_xgb_models)], x='producer', y='PICP_95', hue='model', hue_order=prob_xgb_models, ax=axes[1, 1], palette=palette_dict)
axes[1, 1].axhline(0.95, color='black', linestyle='--', linewidth=1.2, label='95% Nominal Target')
axes[1, 1].set_title('Holdout Period: 95% Interval Coverage (PICP_95)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('PICP_95 (Coverage Probability)')
axes[1, 1].set_xlabel('Producer Well')
axes[1, 1].tick_params(axis='x', rotation=30)
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].legend(fontsize=8.5, loc='lower right')

plt.tight_layout()
plt.savefig('crm_vs_xgb_thesis_charts.png', dpi=300)
plt.close()
print("Saved crm_vs_xgb_thesis_charts.png")