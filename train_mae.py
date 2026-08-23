"""
Stage 1: MAE pretraining.

Usage:
    python train_mae.py --config configs/config.yaml
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PTBXLDataset
from models.mae import MAEModel


def main(cfg):
    device = torch.device(cfg["mae"]["device"] if torch.cuda.is_available() or cfg["mae"]["device"] == "cpu" else "cpu")
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)

    train_ds = PTBXLDataset(
        root=cfg["data"]["ptbxl_root"],
        sampling_rate=cfg["data"]["sampling_rate"],
        window_seconds=cfg["data"]["window_seconds"],
        subset_fraction=cfg["data"]["subset_fraction"],
        split="train",
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        seed=cfg["data"]["seed"],
        return_labels=False,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["mae"]["batch_size"], shuffle=True, num_workers=2)

    model = MAEModel(
        in_channels=cfg["encoder"]["in_channels"],
        embed_dim=cfg["encoder"]["embed_dim"],
        depth=cfg["encoder"]["depth"],
        patch_size=cfg["mae"]["patch_size"],
        mask_ratio=cfg["mae"]["mask_ratio"],
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["mae"]["lr"])

    for epoch in range(cfg["mae"]["epochs"]):
        model.train()
        total_loss = 0.0
        for x in tqdm(train_loader, desc=f"MAE epoch {epoch+1}/{cfg['mae']['epochs']}"):
            x = x.to(device).float()
            recon, mask = model(x)
            loss = model.loss(x, recon, mask)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: recon_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], "mae_encoder.pt")
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"Saved pretrained encoder -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
