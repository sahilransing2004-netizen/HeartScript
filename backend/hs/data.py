"""PyTorch dataset: clean rendered images + on-the-fly (train) or seeded (val/test) photo degradation."""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import degrade
from .labels import SUPERCLASSES
from .preprocess import preprocess_bgr


class EcgImageDataset(Dataset):
    """manifest columns: ecg_id, path (relative to image_dir), NORM, MI, STTC, CD, HYP.
    train=True: random severity each call, drawn with `train_weights` over severities 0..3.
    train=False: deterministic per record; severity 'mixed' picks 1..3 per record, or pass 0..3."""

    def __init__(self, manifest, image_dir, img_hw=(288, 384), train=False, severity="mixed",
                 train_weights=(0.15, 0.35, 0.30, 0.20), rot90=False, seed=0):
        self.ids = manifest["ecg_id"].to_numpy()
        self.paths = manifest["path"].to_numpy()
        self.Y = manifest[SUPERCLASSES].to_numpy(np.float32)
        self.image_dir, self.img_hw = image_dir, tuple(img_hw)
        self.train, self.severity, self.rot90, self.seed = train, severity, rot90, seed
        w = np.asarray(train_weights, np.float64)
        self.w = w / w.sum()

    def __len__(self):
        return len(self.ids)

    def degraded_bgr(self, i, out_size=None):
        img = cv2.imread(os.path.join(self.image_dir, self.paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(os.path.join(self.image_dir, self.paths[i]))
        if self.train:
            rng = np.random.default_rng()
            sev = int(rng.choice(4, p=self.w))
        else:
            rng = np.random.default_rng(int(self.ids[i]) * 1_000_003 + self.seed)
            sev = int(rng.integers(1, 4)) if self.severity == "mixed" else int(self.severity)
        return degrade(img, rng, sev, out_size=out_size, allow_rot90=self.rot90)

    def __getitem__(self, i):
        H, W = self.img_hw
        img = self.degraded_bgr(i, out_size=(W, H))
        return torch.from_numpy(preprocess_bgr(img, self.img_hw)), torch.from_numpy(self.Y[i])
