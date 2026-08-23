"""
Shared encoder backbone. Kept as a lightweight 1D CNN so it trains on CPU
in reasonable time. Swap to a small transformer later once you're on GPU.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ECGEncoder(nn.Module):
    """
    Input:  (batch, n_leads, time)
    Output: (batch, embed_dim)  -- global-pooled embedding
            also exposes forward_features() for (batch, embed_dim, time') sequence output,
            used by the MAE decoder.
    """

    def __init__(self, in_channels=12, embed_dim=128, depth=4):
        super().__init__()
        hidden = [max(embed_dim // (2 ** (depth - i - 1)), 16) for i in range(depth)]
        hidden[-1] = embed_dim
        channels = [in_channels] + hidden

        blocks = []
        for i in range(depth):
            blocks.append(ConvBlock(channels[i], channels[i + 1]))
        self.blocks = nn.Sequential(*blocks)
        self.embed_dim = embed_dim

    def forward_features(self, x):
        # x: (batch, leads, time) -> (batch, embed_dim, time')
        return self.blocks(x)

    def forward(self, x):
        feats = self.forward_features(x)  # (batch, embed_dim, time')
        pooled = feats.mean(dim=-1)  # global average pool -> (batch, embed_dim)
        return pooled
