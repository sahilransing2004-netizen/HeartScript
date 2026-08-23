"""
Stage 2: SimCLR contrastive pretraining for ECG.

Two augmented views of the same window are pulled together, all other
in-batch samples pushed apart (NT-Xent loss). Augmentations are ECG-specific:
lead dropout, gaussian jitter, gain scaling, and time masking.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ECGEncoder


class ECGAugment:
    def __init__(self, jitter_std=0.05, lead_dropout_p=0.1, gain_range=(0.8, 1.2), time_mask_frac=0.1):
        self.jitter_std = jitter_std
        self.lead_dropout_p = lead_dropout_p
        self.gain_range = gain_range
        self.time_mask_frac = time_mask_frac

    def __call__(self, x):
        # x: (batch, leads, time)
        x = x.clone()

        # 1. gaussian jitter
        x = x + torch.randn_like(x) * self.jitter_std

        # 2. random gain scaling per-sample
        gain = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*self.gain_range)
        x = x * gain

        # 3. random lead dropout (zero out a few leads entirely)
        lead_mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.lead_dropout_p).float()
        x = x * lead_mask

        # 4. random contiguous time mask
        t = x.shape[-1]
        mask_len = int(t * self.time_mask_frac)
        if mask_len > 0:
            for i in range(x.shape[0]):
                start = random.randint(0, t - mask_len)
                x[i, :, start : start + mask_len] = 0

        return x


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, in_channels=12, embed_dim=128, depth=4, proj_dim=64, temperature=0.2):
        super().__init__()
        self.encoder = ECGEncoder(in_channels, embed_dim, depth)
        self.proj_head = ProjectionHead(embed_dim, proj_dim)
        self.augment = ECGAugment()
        self.temperature = temperature

    def forward(self, x):
        view1 = self.augment(x)
        view2 = self.augment(x)

        z1 = F.normalize(self.proj_head(self.encoder(view1)), dim=-1)
        z2 = F.normalize(self.proj_head(self.encoder(view2)), dim=-1)
        return z1, z2

    def nt_xent_loss(self, z1, z2):
        # standard NT-Xent (normalized temperature-scaled cross entropy)
        b = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2b, proj_dim)
        sim = torch.mm(z, z.t()) / self.temperature  # (2b, 2b)

        # mask out self-similarity
        mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # positive pairs: (i, i+b) and (i+b, i)
        targets = torch.arange(2 * b, device=z.device)
        targets = (targets + b) % (2 * b)

        return F.cross_entropy(sim, targets)
