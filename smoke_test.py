"""Quick shape/sanity check on synthetic ECG-shaped tensors — no PTB-XL needed."""
import torch

from models.encoder import ECGEncoder
from models.mae import MAEModel
from models.simclr import SimCLRModel
from models.protonet import ProtoNet

B, LEADS, T = 8, 12, 500  # batch, leads, 5s @ 100Hz

x = torch.randn(B, LEADS, T)

print("== Encoder ==")
enc = ECGEncoder(in_channels=LEADS, embed_dim=128, depth=4)
out = enc(x)
print("embedding shape:", out.shape)
assert out.shape == (B, 128)

print("== MAE ==")
mae = MAEModel(in_channels=LEADS, embed_dim=128, depth=4, patch_size=25, mask_ratio=0.5)
recon, mask = mae(x)
loss = mae.loss(x, recon, mask)
print("recon shape:", recon.shape, "mask shape:", mask.shape, "loss:", loss.item())
assert recon.shape == x.shape

print("== SimCLR ==")
simclr = SimCLRModel(in_channels=LEADS, embed_dim=128, depth=4, proj_dim=64, temperature=0.2)
z1, z2 = simclr(x)
cl_loss = simclr.nt_xent_loss(z1, z2)
print("z1 shape:", z1.shape, "loss:", cl_loss.item())
assert z1.shape == (B, 64)

print("== ProtoNet ==")
proto = ProtoNet(encoder=ECGEncoder(in_channels=LEADS, embed_dim=128, depth=4))
n_way, k_shot, n_query = 2, 5, 15
support_x = torch.randn(n_way * k_shot, LEADS, T)
support_y = torch.tensor([c for c in range(n_way) for _ in range(k_shot)])
query_x = torch.randn(n_way * n_query, LEADS, T)
query_y = torch.tensor([c for c in range(n_way) for _ in range(n_query)])
p_loss, p_acc = proto.loss_and_acc(support_x, support_y, query_x, query_y, n_way)
print("protonet loss:", p_loss.item(), "acc:", p_acc.item())

print("\nALL SMOKE TESTS PASSED")
