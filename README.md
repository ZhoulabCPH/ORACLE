# ORACLE

ORACLE (Opportunistic Risk Assessment and Classification of Lesions in the Esophagus) is a two-stage deep learning framework for opportunistic screening of esophageal lesions on non-contrast chest CT images.

The framework first localizes the esophageal region using a 3D nnU-Net segmentation model, and then classifies lesions as benign, early-stage ESCC, or advanced-stage ESCC using a 3D ResNet-based classifier.

By leveraging routinely acquired CT scans, ORACLE provides a non-invasive and scalable auxiliary tool for identifying individuals who may require further endoscopic evaluation.

# 3D CT-ROI ResNet34 Classification

This repository provides a PyTorch/MONAI implementation for three-class classification using 3D CT patches and corresponding ROI masks.

## Features

- 3D ResNet-34 backbone based on MONAI
- Two-channel input: CT image and ROI mask
- Three-class classification
- Deterministic training setup
- Class-balanced loss
- Mixed precision training
- Gradient accumulation
- Warmup + cosine learning rate schedule
- Strict no-TTA validation and model selection
- CSV logging and confusion matrix visualization

## Project Structure

```text
ct_roi_resnet34_classification/
├── configs/
│   └── default.yaml
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── seed.py
│   ├── train.py
│   └── utils.py
├── scripts/
│   └── train.sh
├── data/
├── checkpoints/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Format

The training CSV should contain at least the following columns:

| Column | Description |
| --- | --- |
| `ct_path` | Path to the CT NIfTI patch |
| `roi_path` | Path to the ROI NIfTI mask |
| `earlylabel` | Class label encoded as `0`, `1`, or `2` |

Example:

```csv
ct_path,roi_path,earlylabel
/path/to/case001_ct.nii.gz,/path/to/case001_roi.nii.gz,0
/path/to/case002_ct.nii.gz,/path/to/case002_roi.nii.gz,1
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, please install the PyTorch version that matches your CUDA environment.

## Training

```bash
python -m src.train --config configs/default.yaml
```

Or:

```bash
bash scripts/train.sh
```

## Outputs

The training script saves results under the configured output directory:

- `train_split.csv`
- `val_split.csv`
- `hyperparams.json`
- `epoch_metrics.csv`
- `models/best_resnet34_epoch_xxx_auc_xxxx_noTTA.pth`
- `val_predictions_noTTA.csv`
- `val_confusion_matrix_noTTA.png`
- `loss_auc.png`
- `sens_spec.png`

