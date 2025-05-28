import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

ROUND = 4


def compute_metrics(y_true, y_pred, y_prob=None, average=None):
    # Determine if task is binary or multiclass
    n_classes = len(np.unique(y_true))
    average = average or ("binary" if n_classes == 2 else "weighted")

    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), ROUND),
        "Precision": round(precision_score(y_true, y_pred, average=average), ROUND),
        "Recall": round(recall_score(y_true, y_pred, average=average), ROUND),
        "F1": round(f1_score(y_true, y_pred, average=average), ROUND),
        "CohenKappa": round(cohen_kappa_score(y_true, y_pred).item(), ROUND),
        "BalancedAccuracy": round(balanced_accuracy_score(y_true, y_pred).item(), ROUND),
    }

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "TruePositive": tp,
            "FalsePositive": fp,
            "TrueNegative": tn,
            "FalseNegative": fn,
        })

    # Compute ROC AUC and Log Loss
    if y_prob is not None:
        if n_classes == 2:
            # y_prob should be shape (n_samples, 2)
            metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_prob[:, 1]), ROUND)
        else:
            # Use multiclass ROC AUC
            metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_prob, multi_class="ovr"), ROUND)

        metrics["AireSousCourbeROC"] = metrics["ROC_AUC"]
        metrics["LogLoss"] = round(log_loss(y_true, y_prob), ROUND)

    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names))


def compute_roc_auc(y_true, y_prob, multi_class="ovr"):
    return roc_auc_score(y_true, y_prob, multi_class=multi_class)
