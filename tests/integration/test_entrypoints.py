"""Integration tests for vembed entrypoints (CLI commands)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from vembed.entrypoints.validate_data import validate_data_command


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with sample JSONL and images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create images
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        img = Image.new("RGB", (100, 100), color="red")
        for img_name in ["cat.jpg", "dog.jpg", "bird.jpg"]:
            img.save(img_dir / img_name)

        # Create data file
        data = [
            {
                "query": "a cat",
                "positive": "cat.jpg",
                "negatives": ["dog.jpg"],
            },
            {
                "query": "a dog",
                "positive": "dog.jpg",
                "negatives": ["cat.jpg"],
            },
            {
                "query": "a bird",
                "positive": "bird.jpg",
                "negatives": [],
            },
        ]

        data_path = tmpdir / "data.jsonl"
        with open(data_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        yield {
            "root": tmpdir,
            "data_path": str(data_path),
            "image_dir": str(img_dir),
        }


@pytest.fixture
def mock_args_validate():
    """Create mock args for validate_data_command."""
    args = MagicMock()
    args.sample = 100
    args.check_images = True
    args.column_mapping = None
    return args


class TestValidateDataCommand:
    """Test validate_data_command entrypoint."""

    @patch("vembed.entrypoints.validate_data.load_data")
    @patch("vembed.entrypoints.validate_data.validate_dataset")
    @patch("vembed.entrypoints.validate_data.print_data_validation_report")
    def test_validate_data_success(self, mock_print, mock_validate, mock_load, mock_args_validate, temp_data_dir):
        """Test successful data validation."""
        mock_load.return_value = [{"query": "test", "positive": "test.jpg"}]
        mock_validate.return_value = {
            "total_records": 1,
            "total_issues": 0,
            "issues": {},
        }

        mock_args_validate.data_path = temp_data_dir["data_path"]
        mock_args_validate.image_root = temp_data_dir["image_dir"]

        result = validate_data_command(mock_args_validate)

        assert result is None or result == 0
        mock_load.assert_called_once()
        mock_validate.assert_called_once()

    @patch("vembed.entrypoints.validate_data.load_data")
    def test_validate_data_missing_file(self, mock_load, mock_args_validate):
        """Test validation with missing data file."""
        mock_load.side_effect = FileNotFoundError("Data file not found")
        mock_args_validate.data_path = "/nonexistent/file.jsonl"

        result = validate_data_command(mock_args_validate)

        assert result == 1

    @patch("vembed.entrypoints.validate_data.load_data")
    @patch("vembed.entrypoints.validate_data.validate_dataset")
    @patch("vembed.entrypoints.validate_data.validate_column_mapping")
    @patch("vembed.entrypoints.validate_data.print_data_validation_report")
    @patch("vembed.entrypoints.validate_data.print_validation_report")
    def test_validate_data_with_column_mapping(
        self, mock_col_print, mock_print, mock_col_validate, mock_validate, mock_load, mock_args_validate
    ):
        """Test validation with column mapping."""
        mock_load.return_value = [{"query": "test", "positive": "test.jpg"}]
        mock_validate.return_value = {"total_records": 1, "total_issues": 0, "issues": {}}
        mock_col_validate.return_value = {"valid": True}

        mock_args_validate.data_path = "data.jsonl"
        mock_args_validate.column_mapping = ["query=text", "positive=image"]
        mock_args_validate.image_root = ""

        result = validate_data_command(mock_args_validate)

        assert result is None or result == 0
        mock_col_validate.assert_called_once()


class TestTrainEntrypoint:
    """Test train entrypoint (basic integration)."""

    @patch("vembed.entrypoints.train.main")
    def test_train_import_succeeds(self, mock_main):
        """Test that train module can be imported."""
        from vembed.entrypoints import train

        assert hasattr(train, "main")
        assert callable(train.main)

    def test_train_module_has_required_functions(self):
        """Test that train module has required functions."""
        from vembed.entrypoints import train

        required_functions = ["main"]
        for func_name in required_functions:
            assert hasattr(train, func_name), f"Missing function: {func_name}"


class TestEntrypointsBasicFunctionality:
    """Test basic entrypoint imports and structure."""

    def test_all_entrypoints_importable(self):
        """Test that all entrypoint modules can be imported."""
        import vembed.entrypoints as entrypoints
        import vembed.entrypoints.train as train
        import vembed.entrypoints.validate_data as validate_data

        # All submodules should be importable
        assert all([entrypoints, train, validate_data])

    def test_entrypoints_package_structure(self):
        """Test entrypoints package structure."""
        from vembed.entrypoints import __init__

        # Package should exist and be importable
        assert __init__ is not None


class TestValidateDataIntegration:
    """Integration tests for validate_data with real data."""

    @patch("vembed.entrypoints.validate_data.print_data_validation_report")
    @patch("vembed.entrypoints.validate_data.load_data")
    @patch("vembed.entrypoints.validate_data.validate_dataset")
    def test_validate_data_with_real_args(self, mock_validate, mock_load, mock_print):
        """Test validate_data_command with realistic arguments."""
        # Mock data
        mock_load.return_value = [
            {"query": "cat", "positive": "cat.jpg"},
            {"query": "dog", "positive": "dog.jpg"},
        ]

        mock_validate.return_value = {
            "total_records": 2,
            "total_issues": 0,
            "issues": {},
        }

        # Create args
        args = MagicMock()
        args.data_path = "data.jsonl"
        args.sample = 100
        args.check_images = True
        args.image_root = "images"
        args.column_mapping = None

        # Execute
        result = validate_data_command(args)

        # Verify
        assert result is None or result == 0
        mock_load.assert_called_once_with(args.data_path)


class TestEntrypointErrorHandling:
    """Test error handling in entrypoints."""

    @patch("vembed.entrypoints.validate_data.load_data")
    def test_validate_data_handles_corrupted_file(self, mock_load):
        """Test validate_data handles corrupted data file."""
        mock_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        args = MagicMock()
        args.data_path = "corrupted.jsonl"

        result = validate_data_command(args)

        assert result == 1

    @patch("vembed.entrypoints.validate_data.load_data")
    def test_validate_data_handles_permission_error(self, mock_load):
        """Test validate_data handles permission errors."""
        mock_load.side_effect = PermissionError("Access denied")

        args = MagicMock()
        args.data_path = "restricted.jsonl"

        result = validate_data_command(args)

        assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
