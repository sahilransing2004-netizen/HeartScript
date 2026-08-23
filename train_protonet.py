"""
Stage 3: Prototypical Network episodic training on top of the SimCLR encoder.

Usage:
    python train_protonet.py --config configs/config.yaml --init_from checkpoints/simclr_encoder.pt
"""
import argparse
import os

import torch
import yaml
from tqdm import tqdm

from data.dataset import PTBXLDataset
from models.encoder import ECGEncoder
from models.protonet import ProtoNet, sample_episode


def build_class_indices(dataset):
    class_indices = {}
    for i, label in enumerate(dataset.labels):
        class_indices.setdefault(label, []).append(i)
    return class_indices


def main(cfg, init_from):
    device = torch.device(cfg["protonet"]["device"] if torch.cuda.is_available() or cfg["protonet"]["device"] == "cpu" else "cpu")
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
        return_labels=True,
    )
    val_ds = PTBXLDataset(
        root=cfg["data"]["ptbxl_root"],
        sampling_rate=cfg["data"]["sampling_rate"],
        window_seconds=cfg["data"]["window_seconds"],
        subset_fraction=cfg["data"]["subset_fraction"],
        split="val",
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        seed=cfg["data"]["seed"],
        return_labels=True,
    )

    train_classes = build_class_indices(train_ds)
    val_classes = build_class_indices(val_ds)

    encoder = ECGEncoder(
        in_channels=cfg["encoder"]["in_channels"],
        embed_dim=cfg["encoder"]["embed_dim"],
        depth=cfg["encoder"]["depth"],
    )
    if init_from and os.path.exists(init_from):
        encoder.load_state_dict(torch.load(init_from, map_location=device))
        print(f"Initialized encoder from {init_from}")
    else:
        print("No SimCLR checkpoint found — training ProtoNet encoder from scratch.")

    model = ProtoNet(encoder=encoder).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["protonet"]["lr"])

    n_way = cfg["protonet"]["n_way"]
    k_shot = cfg["protonet"]["k_shot"]
    n_query = cfg["protonet"]["n_query"]
    episodes_per_epoch = cfg["protonet"]["episodes_per_epoch"]

    for epoch in range(cfg["protonet"]["epochs"]):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for _ in tqdm(range(episodes_per_epoch), desc=f"ProtoNet epoch {epoch+1}/{cfg['protonet']['epochs']}"):
            support_x, support_y, query_x, query_y = sample_episode(
                train_ds, n_way, k_shot, n_query, train_classes
            )
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            loss, acc = model.loss_and_acc(support_x, support_y, query_x, query_y, n_way)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_acc += acc.item()

        print(
            f"Epoch {epoch+1}: train_loss={total_loss/episodes_per_epoch:.4f} "
            f"train_acc={total_acc/episodes_per_epoch:.4f}"
        )

        # quick validation pass
        model.eval()
        val_acc = 0.0
        n_val_episodes = 20
        with torch.no_grad():
            for _ in range(n_val_episodes):
                support_x, support_y, query_x, query_y = sample_episode(
                    val_ds, n_way, k_shot, n_query, val_classes
                )
                support_x, support_y = support_x.to(device), support_y.to(device)
                query_x, query_y = query_x.to(device), query_y.to(device)
                _, acc = model.loss_and_acc(support_x, support_y, query_x, query_y, n_way)
                val_acc += acc.item()
        print(f"  val_acc={val_acc/n_val_episodes:.4f}")

    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], "protonet_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved final ProtoNet model -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--init_from", default="checkpoints/simclr_encoder.pt")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.init_from)
