"""
Stage 2: SimCLR contrastive pretraining, initialized from MAE encoder weights.

Usage:
    python train_simclr.py --config configs/config.yaml --init_from checkpoints/mae_encoder.pt
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PTBXLDataset
from models.simclr import SimCLRModel


def main(cfg, init_from):
    device = torch.device(cfg["simclr"]["device"] if torch.cuda.is_available() or cfg["simclr"]["device"] == "cpu" else "cpu")
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
    train_loader = DataLoader(train_ds, batch_size=cfg["simclr"]["batch_size"], shuffle=True, num_workers=2)

    model = SimCLRModel(
        in_channels=cfg["encoder"]["in_channels"],
        embed_dim=cfg["encoder"]["embed_dim"],
        depth=cfg["encoder"]["depth"],
        proj_dim=cfg["simclr"]["proj_dim"],
        temperature=cfg["simclr"]["temperature"],
    ).to(device)

    if init_from and os.path.exists(init_from):
        model.encoder.load_state_dict(torch.load(init_from, map_location=device))
        print(f"Initialized encoder from {init_from}")
    else:
        print("No MAE checkpoint found — training SimCLR encoder from scratch.")

    optim = torch.optim.Adam(model.parameters(), lr=cfg["simclr"]["lr"])

    for epoch in range(cfg["simclr"]["epochs"]):
        model.train()
        total_loss = 0.0
        for x in tqdm(train_loader, desc=f"SimCLR epoch {epoch+1}/{cfg['simclr']['epochs']}"):
            x = x.to(device).float()
            z1, z2 = model(x)
            loss = model.nt_xent_loss(z1, z2)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: contrastive_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], "simclr_encoder.pt")
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"Saved contrastive-tuned encoder -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--init_from", default="checkpoints/mae_encoder.pt")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.init_from)
