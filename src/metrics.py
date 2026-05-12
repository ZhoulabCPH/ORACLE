import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def multiclass_auc_ovr(probs: torch.Tensor, labels: torch.Tensor) -> float:
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return -1
    try:
        return roc_auc_score(labels.numpy(), probs.numpy(), multi_class="ovr")
    except Exception:
        return -1


def sensitivity_specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    per_class = []
    sens_list, spec_list = [], []

    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class.append({
            "cls": cls,
            "sens": sens,
            "spec": spec,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })
        sens_list.append(sens)
        spec_list.append(spec)

    macro = {
        "sens_macro": float(np.mean(sens_list)) if sens_list else 0.0,
        "spec_macro": float(np.mean(spec_list)) if spec_list else 0.0,
    }
    return per_class, macro
