import pytest


def test_import_vembed():
    """Test that vembed top-level package can be imported."""
    import vembed

    assert hasattr(vembed, "__version__")
    assert hasattr(vembed, "Trainer")
    assert hasattr(vembed, "Predictor")
    assert hasattr(vembed, "VEmbedFactoryTrainer")
    assert hasattr(vembed, "VEmbedFactoryPredictor")


def test_import_core_modules():
    """Test that core sub-packages can be imported."""
    from vembed.inference import VEmbedFactoryPredictor  # noqa: F401
    from vembed.losses.factory import LossFactory  # noqa: F401
    from vembed.model.modeling import VisualRetrievalModel  # noqa: F401
    from vembed.trainer import VEmbedFactoryTrainer  # noqa: F401


def test_import_grad_cache():
    """Test that gradient cache module can be imported."""
    try:
        from vembed.training.gradient_cache import GradientCache  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import GradientCache: {e}")


def test_version_format():
    """Test that __version__ follows semver pattern."""
    import re

    import vembed

    pattern = r"^\d+\.\d+\.\d+"
    assert re.match(
        pattern, vembed.__version__
    ), f"Version '{vembed.__version__}' does not match semver pattern"
