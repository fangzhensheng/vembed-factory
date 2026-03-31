"""Data cleaning and validation utilities for vembed-factory."""

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning strategies."""

    skip_missing_images: bool = True
    """Skip records where images cannot be loaded from disk."""

    skip_empty_fields: bool = True
    """Skip records with empty or whitespace-only text fields."""

    skip_invalid_paths: bool = True
    """Skip records with image paths that don't exist."""

    min_text_length: int = 1
    """Minimum required text length for query/positive fields."""

    max_text_length: int = 100000
    """Maximum allowed text length (sanity check for corrupted data)."""

    valid_image_extensions: tuple = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
    )
    """Valid image file extensions."""

    use_multiprocessing: bool = False
    """Use HF Dataset.filter() with multiprocessing for large datasets (>5M records)."""

    num_workers: int = 4
    """Number of workers for multiprocessing filtering."""


def validate_and_clean_data(
    data: Dataset | list[dict],
    config: DataCleaningConfig | None = None,
    image_root: str = "",
) -> tuple[Dataset | list[dict], dict]:
    """Validate dataset and optionally clean invalid records.

    Args:
        data: HF Dataset or list of dicts
        config: DataCleaningConfig with cleaning strategy
        image_root: Root directory for relative image paths

    Returns:
        Tuple of (cleaned_data, report) where report contains:
        - total: Total records before cleaning
        - valid: Valid records after cleaning
        - invalid: Invalid records removed
        - issues: Dict of issue type -> count
    """
    if config is None:
        config = DataCleaningConfig()

    if isinstance(data, list):
        # Convert list to Dataset for consistent handling
        if data:
            data = Dataset.from_dict({k: [d.get(k) for d in data] for k in data[0]})
        else:
            data = Dataset.from_dict({})

    report = {
        "total": len(data),
        "valid": 0,
        "invalid": 0,
        "issues": {},
    }

    def is_valid_record(record: dict) -> bool:
        """Check if a record meets cleaning criteria."""
        if config.skip_empty_fields:
            query = str(record.get("query", "")).strip()
            if not query or len(query) < config.min_text_length:
                report["issues"]["empty_query"] = report["issues"].get("empty_query", 0) + 1
                return False

            if len(query) > config.max_text_length:
                report["issues"]["query_too_long"] = report["issues"].get("query_too_long", 0) + 1
                return False

            positive = str(record.get("positive", "")).strip()
            if not positive:
                report["issues"]["empty_positive"] = report["issues"].get("empty_positive", 0) + 1
                return False

        # Skip URLs and data URIs; only check local file paths
        positive = str(record.get("positive", ""))
        if config.skip_invalid_paths and not positive.startswith(("http", "data:")):
            img_path = Path(image_root) / positive if image_root else Path(positive)
            if not img_path.exists():
                report["issues"]["missing_image"] = report["issues"].get("missing_image", 0) + 1
                return False

        return True

    dataset_size = len(data)
    should_use_multiprocessing = config.use_multiprocessing and dataset_size > 5_000_000

    if should_use_multiprocessing:
        # Use multiprocessing for million-scale datasets to avoid Arrow deserialization overhead
        logger.info(
            "Using multiprocessing for cleaning %d records (%d workers)...",
            dataset_size,
            config.num_workers,
        )

        def filter_fn(record):
            """Filter function for multiprocessing path (local validation without report mutation)."""
            if config.skip_empty_fields:
                query = str(record.get("query", "")).strip()
                if not query or len(query) < config.min_text_length:
                    return False
                if len(query) > config.max_text_length:
                    return False

                positive = str(record.get("positive", "")).strip()
                if not positive:
                    return False

            positive = str(record.get("positive", ""))
            if config.skip_invalid_paths and not positive.startswith(("http", "data:")):
                img_path = Path(image_root) / positive if image_root else Path(positive)
                if not img_path.exists():
                    return False

            return True

        cleaned = data.filter(filter_fn, num_proc=config.num_workers)
        report["valid"] = len(cleaned)
        report["invalid"] = dataset_size - len(cleaned)

    else:
        valid_indices = []
        for i in range(len(data)):
            record = data[i]
            if is_valid_record(record):
                valid_indices.append(i)
            else:
                report["invalid"] += 1

        cleaned = data.select(valid_indices) if valid_indices else data.select([])

        report["valid"] = len(valid_indices)

    return cleaned, report


def print_cleaning_report(report: dict) -> None:
    """Print a formatted cleaning report."""
    print("\n" + "=" * 70)
    print("DATA CLEANING REPORT")
    print("=" * 70)

    print("\nRecord Statistics:")
    print(f"  Total records: {report['total']}")
    print(f"  Valid records: {report['valid']} ({report['valid']*100/max(report['total'],1):.1f}%)")
    print(
        f"  Invalid records: {report['invalid']} ({report['invalid']*100/max(report['total'],1):.1f}%)"
    )

    if report["issues"]:
        print("\nWARNING: Issues Found:")
        for issue_type, count in sorted(report["issues"].items(), key=lambda x: -x[1]):
            print(f"  • {issue_type}: {count}")
    else:
        print("\nSUCCESS: No issues found")

    print("\n" + "=" * 70 + "\n")
