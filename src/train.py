import argparse
import copy
import os

# Determinism environment variables should be set before importing torch.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    precision_score,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import load_config
from src.dataset import CTDataset
from src.evaluate import eval_epoch_no_tta
from src.losses import FocalLoss
from src.metrics import multiclass_auc_ovr, sensitivity_specificity_per_class
from src.model import create_model, load_pretrained_if_available
from src.seed import seed_everything, worker_init_fn
from src.utils import ensure_dir, namespace_to_dict, plot_metrics_from_csv, save_json


def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg, scheduler=None, step_per_batch=False):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    accum_steps = cfg.training.accum_steps

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp):
            logits = model(x)
            loss = criterion(logits, y) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step_per_batch and scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0) * accum_steps
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, torch.cat(all_logits), torch.cat(all_labels)


def build_scheduler(optimizer, cfg, total_steps):
    if cfg.scheduler.use_onecycle:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.scheduler.onecycle_max_lr,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        )
        return scheduler, True

    if cfg.scheduler.use_warmup_cosine:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        warmup_steps = max(1, int(total_steps * cfg.scheduler.warmup_steps_fraction))
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(10, total_steps - warmup_steps),
            eta_min=cfg.scheduler.min_lr,
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return scheduler, True

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=cfg.scheduler.min_lr,
        threshold=0.001,
        threshold_mode="rel",
    )
    return scheduler, False


def main():
    parser = argparse.ArgumentParser(description="Train 3D CT-ROI ResNet34 classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = cfg.paths.output_root
    model_save_dir = os.path.join(output_root, "models")
    ensure_dir(output_root)
    ensure_dir(model_save_dir)

    df = pd.read_csv(cfg.paths.csv_file)

    ct_col = cfg.columns.ct_col
    roi_col = cfg.columns.roi_col
    label_col = cfg.columns.label_col

    df[ct_col] = df[ct_col].apply(lambda x: x if str(x).endswith(".gz") else str(x) + ".gz")
    df[roi_col] = df[roi_col].apply(lambda x: x if str(x).endswith(".gz") else str(x) + ".gz")

    missing_ct = df[ct_col].apply(lambda x: not os.path.exists(x))
    missing_roi = df[roi_col].apply(lambda x: not os.path.exists(x))

    if missing_ct.any() or missing_roi.any():
        print(f"Warning: missing CT files = {missing_ct.sum()}, missing ROI files = {missing_roi.sum()}")
        df = df[~missing_ct]
        print(f"Dataset size after removing missing CT files: {len(df)}")

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.data.val_size,
        random_state=cfg.seed,
        stratify=df[label_col],
    )

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    print(f"Train label distribution:\n{train_df[label_col].value_counts()}")
    print(f"Val label distribution:\n{val_df[label_col].value_counts()}")

    train_df.to_csv(os.path.join(output_root, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_root, "val_split.csv"), index=False)
    save_json(namespace_to_dict(cfg), os.path.join(output_root, "hyperparams.json"))

    train_dataset = CTDataset(train_df, cfg, augment=cfg.augmentation.enabled)
    val_dataset = CTDataset(val_df, cfg, augment=False)

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        generator=torch_gen,
        drop_last=False,
        worker_init_fn=worker_init_fn if cfg.training.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if cfg.training.num_workers > 0 else None,
    )

    classes = np.unique(train_df[label_col])
    class_weights = compute_class_weight("balanced", classes=classes, y=train_df[label_col])
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Class weights: {class_weights}")

    model = create_model(cfg, device)
    model = load_pretrained_if_available(model, cfg, device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.base_lr,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = cfg.training.epochs * max(1, len(train_loader))
    scheduler, scheduler_step_per_batch = build_scheduler(optimizer, cfg, total_steps)

    if cfg.training.use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.training.label_smoothing,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)

    best_auc = -float("inf")
    best_auc_weights = None
    best_auc_epoch = 0
    best_score = -float("inf")
    early_stop_counter = 0

    metrics_csv = os.path.join(output_root, "epoch_metrics.csv")
    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\n{'=' * 30}\nEpoch {epoch}/{cfg.training.epochs}\n{'=' * 30}")

        train_loss, train_logits, train_labels = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            cfg,
            scheduler=scheduler,
            step_per_batch=scheduler_step_per_batch,
        )

        val_loss, val_probs, val_labels = eval_epoch_no_tta(
            model,
            val_loader,
            criterion,
            device,
            use_amp=cfg.training.use_amp,
        )

        with torch.no_grad():
            train_probs = torch.softmax(train_logits, dim=1)

        train_pred = train_probs.argmax(dim=1)
        val_pred = val_probs.argmax(dim=1)

        train_auc = multiclass_auc_ovr(train_probs, train_labels)
        try:
            val_auc = roc_auc_score(val_labels.numpy(), val_probs.numpy(), multi_class="ovr")
        except Exception:
            val_auc = -1

        val_acc = accuracy_score(val_labels.numpy(), val_pred.numpy())
        val_f1 = f1_score(val_labels.numpy(), val_pred.numpy(), average="macro", zero_division=0)
        val_recall = recall_score(val_labels.numpy(), val_pred.numpy(), average="macro", zero_division=0)
        val_precision = precision_score(val_labels.numpy(), val_pred.numpy(), average="macro", zero_division=0)

        train_cls_stats, train_macro = sensitivity_specificity_per_class(
            train_labels.numpy(),
            train_pred.numpy(),
            cfg.model.num_classes,
        )
        val_cls_stats, val_macro = sensitivity_specificity_per_class(
            val_labels.numpy(),
            val_pred.numpy(),
            cfg.model.num_classes,
        )

        val_score = 0.5 * (val_auc if val_auc != -1 else 0) + 0.5 * val_f1

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"Train AUC: {train_auc:.4f} | Val AUC (No TTA): {val_auc:.4f}")
        print(f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Recall: {val_recall:.4f} | Precision: {val_precision:.4f}")
        print(f"Train Sens: {train_macro['sens_macro']:.4f} | Spec: {train_macro['spec_macro']:.4f}")
        print(f"Val Sens: {val_macro['sens_macro']:.4f} | Spec: {val_macro['spec_macro']:.4f}")

        print("\nPer-class validation Sens/Spec:")
        for s in val_cls_stats:
            print(
                f"Class {s['cls']}: Sens={s['sens']:.4f}, Spec={s['spec']:.4f}, "
                f"TP={s['tp']}, FP={s['fp']}, TN={s['tn']}, FN={s['fn']}"
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "train_sens_macro": train_macro["sens_macro"],
            "train_spec_macro": train_macro["spec_macro"],
            "val_sens_macro": val_macro["sens_macro"],
            "val_spec_macro": val_macro["spec_macro"],
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
        }

        pd.DataFrame([row]).to_csv(
            metrics_csv,
            mode=("a" if os.path.exists(metrics_csv) else "w"),
            header=not os.path.exists(metrics_csv),
            index=False,
        )

        if val_auc > best_auc and val_auc != -1:
            best_auc = float(val_auc)
            best_auc_epoch = int(epoch)
            best_auc_weights = copy.deepcopy(model.state_dict())

            best_auc_filename = f"best_resnet34_epoch_{best_auc_epoch:03d}_auc_{best_auc:.4f}_noTTA.pth"
            best_auc_path = os.path.join(model_save_dir, best_auc_filename)
            torch.save(best_auc_weights, best_auc_path)
            print(f"New best AUC model saved: {best_auc_path}")

        if val_score > best_score + cfg.training.early_stop_min_delta:
            best_score = float(val_score)
            early_stop_counter = 0
            print(f"New best combined score: {val_score:.4f}")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{cfg.training.early_stop_patience}")

        if not scheduler_step_per_batch and scheduler is not None:
            scheduler.step(val_auc if val_auc != -1 else 0.0)

        if early_stop_counter >= cfg.training.early_stop_patience:
            print(f"Early stopped at epoch {epoch}.")
            break

    print("\nTraining finished.")

    if best_auc_weights is not None:
        model.load_state_dict(best_auc_weights)
        print(f"Loaded best AUC model from epoch {best_auc_epoch}, AUC={best_auc:.4f}")

        val_loss, val_probs, val_labels = eval_epoch_no_tta(
            model,
            val_loader,
            criterion,
            device,
            use_amp=cfg.training.use_amp,
        )

        true_labels = val_labels.numpy()
        pred_labels = val_probs.argmax(dim=1).numpy()

        results = pd.DataFrame({
            "true_label": true_labels,
            "pred_label": pred_labels,
            "prob_class0": val_probs[:, 0].numpy(),
            "prob_class1": val_probs[:, 1].numpy(),
            "prob_class2": val_probs[:, 2].numpy(),
        })
        pred_csv_path = os.path.join(output_root, "val_predictions_noTTA.csv")
        results.to_csv(pred_csv_path, index=False)

        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Validation Confusion Matrix (Three-class, No TTA)")
        cm_png_path = os.path.join(output_root, "val_confusion_matrix_noTTA.png")
        plt.savefig(cm_png_path, bbox_inches="tight")
        plt.close()

        try:
            final_auc = roc_auc_score(true_labels, val_probs.numpy(), multi_class="ovr")
        except Exception:
            final_auc = -1

        _, val_macro_end = sensitivity_specificity_per_class(
            true_labels,
            pred_labels,
            cfg.model.num_classes,
        )

        print("\nFinal validation performance (No TTA):")
        print(f"AUC: {final_auc:.4f}")
        print(f"Sensitivity macro: {val_macro_end['sens_macro']:.4f}")
        print(f"Specificity macro: {val_macro_end['spec_macro']:.4f}")

    plot_metrics_from_csv(metrics_csv, output_root)


if __name__ == "__main__":
    main()
