import torch

from vembed.losses.functions.cosent import CoSENTLoss
from vembed.losses.functions.infonce import InfoNCELoss
from vembed.losses.functions.matryoshka import MatryoshkaLoss
from vembed.losses.functions.triplet import TripletMarginLoss


def _make_embeds(batch=4, dim=16):
    torch.manual_seed(0)
    q = torch.randn(batch, dim)
    p = q + 0.1 * torch.randn(batch, dim)
    n = torch.randn(batch * 2, dim)
    return q, p, n


def test_infonce_basic():
    q, p, n = _make_embeds()
    loss_fn = InfoNCELoss({"temperature": 0.1})
    loss = loss_fn(q, p, n)
    assert torch.isfinite(loss), "InfoNCE loss should be finite"
    assert loss.item() > 0, "InfoNCE loss should be positive"


def test_triplet_margin_basic():
    q, p, n = _make_embeds()
    loss_fn = TripletMarginLoss({"triplet_margin": 0.5})
    loss = loss_fn(q, p, n)
    assert torch.isfinite(loss), "Triplet loss should be finite"


def test_cosent_basic():
    q, p, n = _make_embeds()
    loss_fn = CoSENTLoss({"cosent_scale": 20.0})
    loss = loss_fn(q, p, n)
    assert torch.isfinite(loss), "CoSENT loss should be finite"


def test_matryoshka_wrapper():
    q, p, n = _make_embeds(dim=32)
    base = InfoNCELoss({"temperature": 0.1})
    mrl = MatryoshkaLoss(base, dims=[32, 16, 8])
    loss = mrl(q, p, n)
    assert torch.isfinite(loss), "MRL wrapped loss should be finite"
