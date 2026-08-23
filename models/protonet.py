"""
Stage 3: Prototypical Networks on top of the pretrained encoder.

For each episode: sample K support examples + Q query examples per class.
Compute class prototypes = mean embedding of support examples.
Classify queries by (negative) squared euclidean distance to prototypes.

This is what lets you get decent anomaly separation even with PTB-XL's
class imbalance — you don't need thousands of labeled anomaly examples,
just a handful of good support sets per episode.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ECGEncoder


class ProtoNet(nn.Module):
    def __init__(self, encoder: ECGEncoder = None, in_channels=12, embed_dim=128, depth=4):
        super().__init__()
        self.encoder = encoder if encoder is not None else ECGEncoder(in_channels, embed_dim, depth)

    def compute_prototypes(self, support_x, support_y, n_way):
        # support_x: (n_way*k_shot, leads, time), support_y: (n_way*k_shot,)
        embeddings = self.encoder(support_x)  # (n_way*k_shot, embed_dim)
        prototypes = []
        for c in range(n_way):
            class_embeds = embeddings[support_y == c]
            prototypes.append(class_embeds.mean(dim=0))
        return torch.stack(prototypes)  # (n_way, embed_dim)

    def forward(self, support_x, support_y, query_x, n_way):
        prototypes = self.compute_prototypes(support_x, support_y, n_way)
        query_embeds = self.encoder(query_x)  # (n_query_total, embed_dim)

        # negative squared euclidean distance -> logits
        dists = torch.cdist(query_embeds, prototypes, p=2) ** 2  # (n_query_total, n_way)
        logits = -dists
        return logits

    def loss_and_acc(self, support_x, support_y, query_x, query_y, n_way):
        logits = self.forward(support_x, support_y, query_x, n_way)
        loss = F.cross_entropy(logits, query_y)
        preds = logits.argmax(dim=-1)
        acc = (preds == query_y).float().mean()
        return loss, acc


def sample_episode(dataset, n_way, k_shot, n_query, class_indices):
    """
    class_indices: dict {class_label: list of dataset indices for that class}
    Returns support_x, support_y, query_x, query_y (tensors, unbatched collate needed by caller)
    """
    import random

    support_items, query_items = [], []
    for c in range(n_way):
        idxs = class_indices[c]
        chosen = random.sample(idxs, k_shot + n_query)
        support_idxs, query_idxs = chosen[:k_shot], chosen[k_shot:]

        for idx in support_idxs:
            x, _ = dataset[idx]
            support_items.append((x, c))
        for idx in query_idxs:
            x, _ = dataset[idx]
            query_items.append((x, c))

    support_x = torch.stack([torch.tensor(x, dtype=torch.float32) for x, _ in support_items])
    support_y = torch.tensor([y for _, y in support_items], dtype=torch.long)
    query_x = torch.stack([torch.tensor(x, dtype=torch.float32) for x, _ in query_items])
    query_y = torch.tensor([y for _, y in query_items], dtype=torch.long)

    return support_x, support_y, query_x, query_y
