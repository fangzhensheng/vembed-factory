import torch
import torch.nn as nn
import torch.nn.functional as F

N = 128
temperature = 0.05
q = torch.ones(N, 1536)
p = torch.ones(N, 1536)

# add small noise to prevent NaN during backward, but wait, they are identical
mrl_dims = [1536, 1024, 768, 512, 256, 128, 64]
total_loss = 0
for dim in mrl_dims:
    q_dim = F.normalize(q[:, :dim], p=2, dim=1)
    p_dim = F.normalize(p[:, :dim], p=2, dim=1)
    logits_dim = q_dim @ p_dim.T / temperature
    loss_dim = nn.CrossEntropyLoss()(logits_dim, torch.arange(N))
    total_loss += loss_dim.item()

print(f"MRL N=128: {total_loss}")

total_loss_256 = 0
for dim in mrl_dims:
    q_dim = F.normalize(torch.ones(256, 1536)[:, :dim], p=2, dim=1)
    p_dim = F.normalize(torch.ones(256, 1536)[:, :dim], p=2, dim=1)
    logits_dim = q_dim @ p_dim.T / temperature
    loss_dim = nn.CrossEntropyLoss()(logits_dim, torch.arange(256))
    total_loss_256 += loss_dim.item()
print(f"MRL N=256: {total_loss_256}")

total_loss_512 = 0
for dim in mrl_dims:
    q_dim = F.normalize(torch.ones(512, 1536)[:, :dim], p=2, dim=1)
    p_dim = F.normalize(torch.ones(512, 1536)[:, :dim], p=2, dim=1)
    logits_dim = q_dim @ p_dim.T / temperature
    loss_dim = nn.CrossEntropyLoss()(logits_dim, torch.arange(512))
    total_loss_512 += loss_dim.item()
print(f"MRL N=512: {total_loss_512}")
