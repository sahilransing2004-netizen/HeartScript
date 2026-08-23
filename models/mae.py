"""
Stage 1: Masked Autoencoder pretraining for ECG.

We mask random contiguous time-patches of the raw signal, encode the visible
signal (masked regions zeroed out), and train a small decoder to reconstruct
the full signal. Loss is MSE only over the masked patches (standard MAE recipe).
"""
import torch
import torch.nn as nn

from models.encoder import ECGEncoder


class MAEDecoder(nn.Module):
    """Small conv-transpose decoder mapping encoder features back to raw signal length."""

    def __init__(self, embed_dim, out_channels, depth=4):
        super().__init__()
        channels = [embed_dim // (2 ** i) for i in range(depth)]
        channels = [max(c, 16) for c in channels] + [out_channels]

        layers = []
        for i in range(depth):
            layers.append(
                nn.ConvTranspose1d(
                    channels[i], channels[i + 1], kernel_size=8, stride=2, padding=3
                )
            )
            if i < depth - 1:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MAEModel(nn.Module):
    def __init__(self, in_channels=12, embed_dim=128, depth=4, patch_size=25, mask_ratio=0.5):
        super().__init__()
        self.encoder = ECGEncoder(in_channels, embed_dim, depth)
        self.decoder = MAEDecoder(embed_dim, in_channels, depth)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def random_mask(self, x):
        # x: (batch, leads, time). Mask whole time-patches (same mask across all leads).
        b, c, t = x.shape
        n_patches = t // self.patch_size
        n_mask = max(1, int(n_patches * self.mask_ratio))

        mask = torch.zeros(b, n_patches, device=x.device)
        for i in range(b):
            mask_idx = torch.randperm(n_patches)[:n_mask]
            mask[i, mask_idx] = 1

        # upsample patch-mask to full time length
        full_mask = mask.repeat_interleave(self.patch_size, dim=1)  # (b, n_patches*patch_size)
        if full_mask.shape[1] < t:
            pad = t - full_mask.shape[1]
            full_mask = torch.nn.functional.pad(full_mask, (0, pad))
        full_mask = full_mask.unsqueeze(1)  # (b, 1, t)

        x_masked = x * (1 - full_mask)
        return x_masked, full_mask

    def forward(self, x):
        x_masked, mask = self.random_mask(x)
        feats = self.encoder.forward_features(x_masked)  # (b, embed_dim, t')
        recon = self.decoder(feats)  # (b, leads, t'')

        # decoder output length may not exactly match input due to stride rounding — crop/pad
        t_in = x.shape[-1]
        t_out = recon.shape[-1]
        if t_out > t_in:
            recon = recon[..., :t_in]
        elif t_out < t_in:
            recon = torch.nn.functional.pad(recon, (0, t_in - t_out))

        return recon, mask

    def loss(self, x, recon, mask):
        # reconstruction loss computed ONLY on masked regions
        mse = (recon - x) ** 2
        mse = mse * mask  # zero out loss on visible (unmasked) regions
        denom = mask.sum() * x.shape[1] + 1e-8  # normalize by number of masked elements
        return mse.sum() / denom
