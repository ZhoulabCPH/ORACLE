import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import affine_transform


class CTDataset(Dataset):
    def __init__(self, df, cfg, augment: bool = False):
        self.data = df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment

        self.patch_shape = tuple(cfg.data.patch_shape)
        self.ct_col = cfg.columns.ct_col
        self.roi_col = cfg.columns.roi_col
        self.label_col = cfg.columns.label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ct_path = row[self.ct_col]
        roi_path = row[self.roi_col]
        label = int(row[self.label_col])

        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"CT file not found: {ct_path}")

        ct_img = nib.load(ct_path).get_fdata().astype(np.float32)

        roi_exists = os.path.exists(roi_path)
        if not roi_exists:
            if self.cfg.augmentation.roi_missing_as_zero:
                roi_img = np.zeros(self.patch_shape, dtype=np.float32)
            else:
                raise FileNotFoundError(f"ROI file not found: {roi_path}")
        else:
            try:
                roi_img = nib.load(roi_path).get_fdata().astype(np.float32)
            except Exception:
                if self.cfg.augmentation.roi_missing_as_zero:
                    roi_img = np.zeros(self.patch_shape, dtype=np.float32)
                    print(f"[Dataset] Failed to load ROI, using zeros: {roi_path}")
                else:
                    raise

        roi_img = (roi_img > 0).astype(np.float32)

        ct_img = np.clip(ct_img, -125, 225)
        ct_img = (ct_img + 125) / 350.0

        if ct_img.shape != self.patch_shape:
            raise ValueError(f"Invalid CT shape: expected {self.patch_shape}, got {ct_img.shape} @ {ct_path}")
        if roi_img.shape != self.patch_shape:
            raise ValueError(f"Invalid ROI shape: expected {self.patch_shape}, got {roi_img.shape} @ {roi_path}")

        if self.augment and np.random.rand() < self.cfg.augmentation.augment_prob:
            ct_img, roi_img = self._random_affine(ct_img, roi_img)

        ct_tensor = torch.from_numpy(ct_img).unsqueeze(0)
        roi_tensor = torch.from_numpy(roi_img).unsqueeze(0)

        if self.augment and self.cfg.augmentation.add_ct_noise:
            noise = torch.randn_like(ct_tensor) * self.cfg.augmentation.ct_noise_std
            ct_tensor = (ct_tensor + noise).clamp_(0.0, 1.0)

        image = torch.cat([ct_tensor, roi_tensor], dim=0)

        if self.augment and self.cfg.augmentation.roi_channel_dropout_p > 0:
            if np.random.rand() < self.cfg.augmentation.roi_channel_dropout_p:
                image[1].zero_()

        return image, label

    def _random_affine(self, ct_img, roi_img):
        rotation_range = self.cfg.augmentation.rotation_range
        scale_range = self.cfg.augmentation.scale_range

        angle_x = np.random.uniform(-rotation_range, rotation_range)
        angle_y = np.random.uniform(-rotation_range, rotation_range)
        angle_z = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(scale_range[0], scale_range[1])

        rot_mat = np.eye(3)
        rot_mat = rot_mat @ np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle_x)), -np.sin(np.radians(angle_x))],
            [0, np.sin(np.radians(angle_x)),  np.cos(np.radians(angle_x))]
        ])
        rot_mat = rot_mat @ np.array([
            [np.cos(np.radians(angle_y)), 0, np.sin(np.radians(angle_y))],
            [0, 1, 0],
            [-np.sin(np.radians(angle_y)), 0, np.cos(np.radians(angle_y))]
        ])
        rot_mat = rot_mat @ np.array([
            [np.cos(np.radians(angle_z)), -np.sin(np.radians(angle_z)), 0],
            [np.sin(np.radians(angle_z)),  np.cos(np.radians(angle_z)), 0],
            [0, 0, 1]
        ])
        rot_mat = scale * rot_mat

        ct_img = self.apply_transform(ct_img, rot_mat, order=3)
        roi_img = self.apply_transform(roi_img, rot_mat, order=0)
        roi_img = (roi_img > 0.5).astype(np.float32)
        return ct_img, roi_img

    @staticmethod
    def apply_transform(img, affine, order=3):
        center = np.array(img.shape) / 2 - 0.5
        offset = center - affine.dot(center)
        return affine_transform(
            img, affine, offset=offset, output_shape=img.shape,
            order=order, mode="nearest"
        )
