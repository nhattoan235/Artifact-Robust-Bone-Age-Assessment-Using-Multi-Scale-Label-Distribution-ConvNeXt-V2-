import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.utils import resample
import pprint

def compute_metrics_with_ci(preds_single_path, preds_multi_path, ground_truth_path, ground_truth_col="Ground truth bone age (months)", n_bootstrap=1000, alpha=0.05):
    preds_single = pd.read_csv(preds_single_path)
    preds_multi = pd.read_csv(preds_multi_path)
    ground_truth = pd.read_excel(ground_truth_path)
    ground_truth = ground_truth.rename(columns={ground_truth_col: "True Months"})
    single_merged = pd.merge(preds_single, ground_truth, on="Case ID")
    multi_merged = pd.merge(preds_multi, ground_truth, on="Case ID")

    multi_avg = multi_merged.groupby("Case ID").agg({
        "Predicted Months": "mean",
        "True Months": "first"
    }).reset_index()


    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        return mae, rmse

    def bootstrap_ci(y_true, y_pred, metric_fn):
        stats = [
            metric_fn(*resample(y_true, y_pred))
            for _ in range(n_bootstrap)
        ]
        lower = np.percentile(stats, 100 * alpha / 2)
        upper = np.percentile(stats, 100 * (1 - alpha / 2))
        return lower, upper

    results = {}
    for label, df in zip(["single", "multi"], [single_merged, multi_avg]):
        y_true = df["True Months"]
        y_pred = df["Predicted Months"]
        mae, rmse = compute_metrics(y_true, y_pred)
        mae_ci = bootstrap_ci(y_true, y_pred, mean_absolute_error)
        rmse_ci = bootstrap_ci(y_true, y_pred, lambda y1, y2: root_mean_squared_error(y1, y2))
        results[label] = {
            "MAE": mae, "MAE_CI": mae_ci,
            "RMSE": rmse, "RMSE_CI": rmse_ci
        }

    return results


def compute_auc_with_ci(
    preds_single_path,
    preds_multi_path,
    ground_truth_path,
    n_bootstrap=1000,
    alpha=0.05
):
    preds_single = pd.read_csv(preds_single_path)
    preds_multi = pd.read_csv(preds_multi_path)
    ground_truth = pd.read_csv(ground_truth_path)

    ground_truth["True Label"] = ground_truth["Sex"].map({"M": 1, "F": 0})

    merged_single = pd.merge(preds_single, ground_truth[["Case ID", "True Label"]], left_on="id", right_on="Case ID")
    merged_multi = pd.merge(preds_multi, ground_truth[["Case ID", "True Label"]], left_on="id", right_on="Case ID")

    multi_avg = merged_multi.groupby("id").agg({
        "prob_male": "mean",
        "True Label": "first"
    }).reset_index()

    def bootstrap_auc_ci(y_true, y_score):
        stats = []
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) > 1:
                stats.append(roc_auc_score(y_true[idx], y_score[idx]))
        return np.percentile(stats, 100 * alpha / 2), np.percentile(stats, 100 * (1 - alpha / 2))

    results = {}
    for label, df in zip(["single", "multi"], [merged_single, multi_avg]):
        y_true = df["True Label"]
        y_score = df["prob_male"]
        auc = roc_auc_score(y_true, y_score)
        auc_ci = bootstrap_auc_ci(y_true, y_score)
        results[label] = {"AUC": auc, "AUC_CI": auc_ci}

    return results


regression_results = compute_metrics_with_ci(
    r"C:\Projetos\hand-bone\boneage_predictions.csv",
    r"C:\Projetos\hand-bone\cleaned_boneage_predictions_mult.csv",
    r"C:\Projetos\hand-bone\Bone age ground truth.xlsx"
)


classification_results = compute_auc_with_ci(
    preds_single_path=r"C:\Projetos\hand-bone\original_gender_predictions.csv",
    preds_multi_path=r"C:\Projetos\hand-bone\ensemble_gender_predictions_mult.csv",
    ground_truth_path=r"C:\Projetos\hand-bone\boneage-test-dataset.csv"
)
pprint.pprint(regression_results)
pprint.pprint(classification_results)
