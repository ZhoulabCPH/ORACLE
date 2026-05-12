import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def namespace_to_dict(ns):
    if hasattr(ns, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    return ns


def plot_metrics_from_csv(metrics_csv_path: str, output_root: str):
    if not os.path.exists(metrics_csv_path):
        print(f"[plot] CSV not found: {metrics_csv_path}")
        return

    dfm = pd.read_csv(metrics_csv_path)

    try:
        plt.figure()
        for col, label in [
            ("train_loss", "Train Loss"),
            ("val_loss", "Val Loss"),
            ("train_auc", "Train AUC"),
            ("val_auc", "Val AUC"),
        ]:
            if col in dfm.columns:
                plt.plot(dfm["epoch"], dfm[col], label=label)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Loss & AUC over Epochs (No TTA)")
        plt.legend()
        plt.savefig(os.path.join(output_root, "loss_auc.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[plot] Failed to plot loss_auc.png: {e}")

    try:
        plt.figure()
        for col, label in [
            ("train_sens_macro", "Train Sensitivity (macro)"),
            ("train_spec_macro", "Train Specificity (macro)"),
            ("val_sens_macro", "Val Sensitivity (macro)"),
            ("val_spec_macro", "Val Specificity (macro)"),
        ]:
            if col in dfm.columns:
                plt.plot(dfm["epoch"], dfm[col], label=label)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Sensitivity & Specificity over Epochs (No TTA)")
        plt.legend()
        plt.savefig(os.path.join(output_root, "sens_spec.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[plot] Failed to plot sens_spec.png: {e}")
