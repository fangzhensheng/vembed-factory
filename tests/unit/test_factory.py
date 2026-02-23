from vembed.losses.factory import LossFactory


def test_factory_infonce():
    cfg = {"loss_type": "infonce", "temperature": 0.07}
    loss = LossFactory.create(cfg)
    assert loss is not None


def test_factory_triplet_mrl():
    cfg = {
        "loss_type": "triplet",
        "triplet_margin": 0.3,
        "use_mrl": True,
        "mrl_dims": [64, 32],
    }
    loss = LossFactory.create(cfg)
    # MatryoshkaLoss has attribute 'dims'
    assert hasattr(loss, "dims")
    assert loss.dims == [64, 32]
