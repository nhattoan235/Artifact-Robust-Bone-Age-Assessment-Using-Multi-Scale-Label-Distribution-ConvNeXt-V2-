import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ---- PATHS ----
gt_path = r'C:\Projetos\hand-bone\boneage_predictions.csv'
pred_mult_path = r'C:\Projetos\hand-bone\cleaned_boneage_predictions_mult.csv'
save_path = r'C:\Projetos\hand-bone\figs\mult_original_vs_calibrated.png'

# ---- LOAD DATA ----
gt_df = pd.read_csv(gt_path)
gt_df.columns = [c.strip() for c in gt_df.columns]
pred_df = pd.read_csv(pred_mult_path)
pred_df.columns = [c.strip() for c in pred_df.columns]

# ---- MERGE GROUND TRUTH ----
if 'Ground truth bone age (months)' not in pred_df.columns:
    pred_df = pred_df.merge(
        gt_df[['Case ID', 'Predicted Months']].rename(columns={'Predicted Months': 'Ground truth bone age (months)'}),
        on='Case ID', how='left'
    )

# ---- CALIBRATION ----
y_true = pred_df['Ground truth bone age (months)']
y_pred_orig = pred_df['Predicted Months']
slope, intercept, *_ = stats.linregress(y_pred_orig, y_true)
pred_df['Calibrated'] = intercept + slope * y_pred_orig
y_pred_cal = pred_df['Calibrated']

# ---- METRICS ----
mae_orig = mean_absolute_error(y_true, y_pred_orig)
mae_cal  = mean_absolute_error(y_true, y_pred_cal)
rmse_orig = np.sqrt(mean_squared_error(y_true, y_pred_orig))
rmse_cal  = np.sqrt(mean_squared_error(y_true, y_pred_cal))
bias_orig = np.mean(y_pred_orig - y_true)
bias_cal  = np.mean(y_pred_cal - y_true)

print("Performance Metrics:")
print(f"Original   → MAE: {mae_orig:.2f}, RMSE: {rmse_orig:.2f}, Bias: {bias_orig:+.2f}")
print(f"Calibrated → MAE: {mae_cal:.2f}, RMSE: {rmse_cal:.2f}, Bias: {bias_cal:+.2f}")

# ---- PLOT ----
sns.set_theme(style="whitegrid", font_scale=1.25)
fig, axes = plt.subplots(2, 1, figsize=(12, 18), sharex=True)
plt.subplots_adjust(hspace=0.10)

bright_blue = '#2196f3'

for idx, col in enumerate(['Predicted Months', 'Calibrated']):
    ax = axes[idx]
    sns.boxplot(
        x='Ground truth bone age (months)',
        y=col,
        data=pred_df,
        color=bright_blue,
        showfliers=False,
        width=0.85,
        linewidth=2.2,
        ax=ax
    )

    lims = [
        min(pred_df['Ground truth bone age (months)'].min(), pred_df[col].min()),
        max(pred_df['Ground truth bone age (months)'].max(), pred_df[col].max())
    ]
    
    ax.plot(
        lims, lims,
        ls='--',
        color='gray',
        linewidth=2,
        alpha=0.7,
        label='Perfect Agreement (y = x)'
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_ylabel(('Prediction' if idx == 0 else 'Calibrated Prediction') + ' (months)',
                  fontsize=16, fontweight='semibold', color='black')
    ax.set_title('Before Calibration' if idx == 0 else 'After Calibration',
                 fontsize=20, weight='bold', color='black')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([])
    ax.set_xlabel('')

    if idx == 0:
        medians = pred_df.groupby('Ground truth bone age (months)')[col].median().reset_index()
        x_median = medians['Ground truth bone age (months)'].values
        y_median = medians[col].values
        slope_m, intercept_m, *_ = stats.linregress(x_median, y_median)
        y_fit = intercept_m + slope_m * x_median
        ax.plot(
            x_median, y_fit,
            color='#0d47a1',
            linewidth=2.5,
            linestyle='--',
            alpha=1.0,
            label='Regression line'
        )
    ax.legend(loc='upper left', fontsize=14, frameon=True, facecolor='white',
              edgecolor='black', borderpad=1.1, labelcolor='black')

plt.tight_layout(rect=[0, 0.045, 1, 1])
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=350)
plt.show()
