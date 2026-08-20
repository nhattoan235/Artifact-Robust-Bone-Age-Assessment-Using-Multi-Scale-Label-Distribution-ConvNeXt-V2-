import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

CSV_GT = r"C:\Projetos\hand-bone\boneage-test-dataset.csv"
CSV_CLEAN = r"C:\Projetos\hand-bone\ensemble_gender_predictions_mult.csv"
CSV_ORIG = r"C:\Projetos\hand-bone\original_gender_predictions.csv"
SAVE_PATH = r"C:\Projetos\hand-bone\figs\confusion_matrix_original_vs_majorityvote.png"

df_gt = pd.read_csv(CSV_GT)
df_clean = pd.read_csv(CSV_CLEAN)
df_orig = pd.read_csv(CSV_ORIG)

df_gt['id'] = df_gt['Case ID'].astype(str)
df_gt['male'] = (df_gt['Sex'].str.upper().str.strip() == 'M').astype(int)

df_clean['id'] = df_clean['id'].astype(str)
df_clean['predicted_label'] = df_clean['predicted_label'].astype(int)

df_orig['id'] = df_orig['id'].astype(str)
df_orig['predicted_label'] = df_orig['predicted_label'].astype(int)


df_orig = pd.merge(df_orig, df_gt[['id', 'male']], on='id')
df_clean = pd.merge(df_clean, df_gt[['id', 'male']], on='id')

majority_votes = (
    df_clean.groupby('id')['predicted_label']
    .agg(lambda x: int(x.mean() >= 0.5))
    .reset_index()
    .rename(columns={'predicted_label': 'majority_vote'})
)

majority_votes = pd.merge(majority_votes, df_gt[['id', 'male']], on='id')
cm_majority = confusion_matrix(majority_votes['male'], majority_votes['majority_vote'])
cm_original = confusion_matrix(df_orig['male'], df_orig['predicted_label'])


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

disp_left = ConfusionMatrixDisplay(confusion_matrix=cm_majority, display_labels=["Female", "Male"])
disp_left.plot(ax=axes[0], cmap='Blues', colorbar=False, text_kw={'fontsize': 16})
axes[0].set_title("Majority Vote on Inpainted Dataset")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Ground Truth")

disp_right = ConfusionMatrixDisplay(confusion_matrix=cm_original, display_labels=["Female", "Male"])
disp_right.plot(ax=axes[1], cmap='Blues', colorbar=False, text_kw={'fontsize': 16})
axes[1].set_title("Original Dataset")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Ground Truth")

plt.suptitle("Confusion Matrices: Majority Vote vs. Original Dataset Predictions", fontsize=14, y=1.05)
plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.show()
