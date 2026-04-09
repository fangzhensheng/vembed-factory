import torch
import torch.nn as nn
import torch.nn.functional as F

N = 128
temperature = 0.05
q = torch.ones(N, 1536)
p = torch.ones(N, 1536)

q = F.normalize(q, p=2, dim=1)
p = F.normalize(p, p=2, dim=1)

logits = q @ p.T / temperature
target = torch.arange(N)
loss = nn.CrossEntropyLoss()(logits, target)
print(f"Single dim InfoNCE (identical embeddings, N={N}): {loss.item()}")

# MRL
mrl_dims = [1536, 1024, 768, 512, 256, 128, 64]
total_loss = 0
for dim in mrl_dims:
    q_dim = F.normalize(q[:, :dim], p=2, dim=1)
    p_dim = F.normalize(p[:, :dim], p=2, dim=1)
    logits_dim = q_dim @ p_dim.T / temperature
    loss_dim = nn.CrossEntropyLoss()(logits_dim, target)
    total_loss += loss_dim.item()

print(f"MRL InfoNCE (identical embeddings, N={N}, 7 dims): {total_loss}")

# MRL with 64
N2 = 64
q2 = torch.ones(N2, 1536)
p2 = torch.ones(N2, 1536)
q2 = F.normalize(q2, p=2, dim=1)
p2 = F.normalize(p2, p=2, dim=1)
target2 = torch.arange(N2)
total_loss2 = 0
for dim in mrl_dims:
    q_dim = F.normalize(q2[:, :dim], p=2, dim=1)
    p_dim = F.normalize(p2[:, :dim], p=2, dim=1)
    logits_dim = q_dim @ p_dim.T / temperature
    loss_dim = nn.CrossEntropyLoss()(logits_dim, target2)
    total_loss2 += loss_dim.item()
print(f"MRL InfoNCE (identical embeddings, N={N2}, 7 dims): {total_loss2}")
