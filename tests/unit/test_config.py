"""Unit tests for configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from vembed.config import load_base_config, merge_configs
from vembed.training.config import get_distributed_config


class TestBaseConfig:
    """Test base configuration loading."""

    def test_load_base_config_structure(self):
        """Test that base config has required structure."""
        config = load_base_config()

        assert isinstance(config, dict)
        assert "model_name" in config or "batch_size" in config

    def test_load_base_config_defaults(self):
        """Test common default values."""
        config = load_base_config()

        # Common defaults should exist
        assert config.get("batch_size", 0) > 0
        assert config.get("learning_rate", 0) > 0
        assert config.get("epochs", 0) > 0


class TestConfigMerging:
    """Test configuration merging logic."""

    def test_merge_simple(self):
        """Test simple config merging."""
        base = {"a": 1, "b": 2}
        user = {"b": 3}

        result = merge_configs(base, user)

        assert result["a"] == 1
        assert result["b"] == 3

    def test_merge_new_keys(self):
        """Test merging with new keys."""
        base = {"a": 1}
        user = {"b": 2, "c": 3}

        result = merge_configs(base, user)

        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3

    def test_merge_nested(self):
        """Test merging nested configurations."""
        base = {"model": {"name": "clip", "dim": 256}}
        user = {"model": {"dim": 512}}

        result = merge_configs(base, user)

        assert result["model"]["name"] == "clip"
        assert result["model"]["dim"] == 512

    def test_merge_empty(self):
        """Test merging with empty dicts."""
        base = {"a": 1}

        result1 = merge_configs(base, {})
        assert result1["a"] == 1

        result2 = merge_configs({}, base)
        assert result2["a"] == 1


class TestDistributedConfig:
    """Test distributed training configuration."""

    def test_gradient_cache_disabled(self):
        """Test when gradient cache is disabled."""
        config = {
            "use_gradient_cache": False,
            "use_gradient_checkpointing": False,
        }

        grad_ckpt, grad_cache, find_unused = get_distributed_config(config)

        assert grad_cache is False
        assert find_unused is False

    def test_gradient_cache_enabled(self):
        """Test when gradient cache is enabled."""
        config = {
            "use_gradient_cache": True,
            "use_gradient_checkpointing": False,
        }

        _, grad_cache, find_unused = get_distributed_config(config)

        assert grad_cache is True
        # With gradient cache, find_unused should be True
        assert find_unused is True

    def test_gradient_checkpointing_enabled(self):
        """Test gradient checkpointing configuration."""
        config = {
            "use_gradient_cache": False,
            "use_gradient_checkpointing": True,
        }

        grad_ckpt, grad_cache, find_unused = get_distributed_config(config)

        assert grad_ckpt is True
        # Without gradient cache, find_unused should be False
        assert find_unused is False

    def test_both_optimizations_enabled(self):
        """Test when both gradient cache and checkpointing are enabled."""
        config = {
            "use_gradient_cache": True,
            "use_gradient_checkpointing": True,
        }

        grad_ckpt, grad_cache, find_unused = get_distributed_config(config)

        assert grad_ckpt is True
        assert grad_cache is True
        # find_unused should be True with gradient cache
        assert find_unused is True

    def test_default_config_values(self):
        """Test default values when keys are missing."""
        config = {}

        grad_ckpt, grad_cache, find_unused = get_distributed_config(config)

        # Should not raise, should use defaults
        assert isinstance(grad_ckpt, bool)
        assert isinstance(grad_cache, bool)
        assert isinstance(find_unused, bool)


class TestConfigValidation:
    """Test configuration validation."""

    def test_required_fields(self):
        """Test that required fields are present."""
        config = load_base_config()

        required_fields = ["batch_size", "learning_rate", "epochs"]
        for field in required_fields:
            assert field in config or config.get(field) is not None

    def test_numeric_fields_are_numeric(self):
        """Test that numeric fields are actually numeric."""
        config = load_base_config()

        numeric_fields = [
            "batch_size",
            "learning_rate",
            "epochs",
            "warmup_ratio",
            "weight_decay",
        ]

        for field in numeric_fields:
            if field in config:
                assert isinstance(config[field], (int, float)), f"{field} should be numeric"

    def test_string_fields_are_strings(self):
        """Test that string fields are strings."""
        config = load_base_config()

        string_fields = ["scheduler_type", "loss_type", "output_dir"]

        for field in string_fields:
            if field in config:
                assert isinstance(config[field], str), f"{field} should be string"


class TestYAMLConfig:
    """Test YAML configuration loading."""

    def test_yaml_loading(self):
        """Test loading YAML config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            config_data = {
                "model_name": "test_model",
                "batch_size": 32,
                "learning_rate": 1e-5,
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config["model_name"] == "test_model"
            assert loaded_config["batch_size"] == 32
            assert loaded_config["learning_rate"] == 1e-5

    def test_yaml_with_lists(self):
        """Test YAML with list values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            config_data = {
                "model_name": "test",
                "mrl_dims": [1024, 512, 256],
                "target_modules": ["q_proj", "v_proj"],
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config["mrl_dims"] == [1024, 512, 256]
            assert loaded_config["target_modules"] == ["q_proj", "v_proj"]


class TestConfigOverrides:
    """Test configuration override logic."""

    def test_cli_override(self):
        """Test CLI override parsing."""
        base = {"batch_size": 32, "learning_rate": 1e-5}

        overrides = {"batch_size": "64", "learning_rate": "5e-5"}

        # Simulate override merging
        for key, value in overrides.items():
            if key in base:
                if isinstance(base[key], (int, float)):
                    base[key] = float(value)
                else:
                    base[key] = value

        assert base["batch_size"] == 64.0
        assert base["learning_rate"] == 5e-5

    def test_override_priority(self):
        """Test override priority: CLI > YAML > Defaults."""
        defaults = {"batch_size": 32, "epochs": 3}
        yaml_config = {"batch_size": 64}
        cli_override = {"epochs": 5}

        # Merge in priority order
        result = {**defaults, **yaml_config, **cli_override}

        assert result["batch_size"] == 64  # From YAML
        assert result["epochs"] == 5  # From CLI


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
