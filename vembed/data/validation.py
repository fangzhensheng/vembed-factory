"""Data validation utilities for vembed-factory datasets."""

import json
import logging
from pathlib import Path
from typing import Any

from .loading import load_data

logger = logging.getLogger(__name__)


def validate_dataset(
    data_source: str | list[dict[str, Any]],
    column_mapping: dict[str, str] | None = None,
    sample_size: int = 100,
    skip_invalid: bool = False,
) -> dict[str, Any]:
    """
    Validate dataset quality and return statistics.

    Args:
        data_source: File path or loaded data.
        column_mapping: Optional column name mapping.
        sample_size: Number of records to sample for analysis.
        skip_invalid: If True, filter out records with missing required fields.

    Returns:
        Dictionary with validation statistics including:
        - total_records: Total count
        - sample_size: Actual sample size used
        - text_lengths: Distribution of text lengths
        - image_count: Number of records with images
        - negative_count: Number of records with negatives
        - missing_fields: Dict of field -> count of missing records
        - required_fields_present: Boolean for each required field
    """
    data = load_data(data_source) if isinstance(data_source, str | Path) else data_source
    total = len(data)

    sample_size = min(sample_size, total)
    sample = data[:sample_size]

    col_mapping = column_mapping or {}
    query_col = col_mapping.get("query", "query")
    positive_col = col_mapping.get("positive", "positive")
    negatives_col = col_mapping.get("negatives", "negatives")

    # Fallback to common aliases if mapped column not found
    common_fallbacks = {
        "query": ["query", "caption", "text", "question"],
        "positive": ["positive", "image", "answer", "content"],
        "negatives": ["negatives", "negative_samples", "hard_negatives"],
    }

    def find_column(record, col_key, fallback_list):
        """Find column in record with fallback to aliases."""
        if col_key in record:
            return col_key
        for alias in fallback_list:
            if alias in record:
                return alias
        return None

    query_lengths = []
    image_count = 0
    negative_count = 0
    missing_required = {"query": 0, "positive": 0}
    has_query_image = False
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")

    for record in sample:
        # Find actual columns
        actual_query_col = find_column(record, query_col, common_fallbacks["query"])
        actual_positive_col = find_column(record, positive_col, common_fallbacks["positive"])
        actual_negatives_col = find_column(record, negatives_col, common_fallbacks["negatives"])

        query_val = str(record.get(actual_query_col, "")).strip() if actual_query_col else ""
        if not query_val:
            missing_required["query"] += 1
        else:
            query_lengths.append(len(query_val))

        pos_val = str(record.get(actual_positive_col, "")).strip() if actual_positive_col else ""
        if not pos_val:
            missing_required["positive"] += 1
        else:
            pos_val_lower = pos_val.lower()
            if (
                any(pos_val_lower.endswith(ext) for ext in image_extensions)
                or "/" in pos_val_lower
                or "\\" in pos_val_lower
            ):
                image_count += 1

        if actual_negatives_col and record.get(actual_negatives_col):
            negative_count += 1

        if "query_image" in record or "source_image" in record:
            has_query_image = True

    stats = {
        "total_records": total,
        "sample_size": sample_size,
        "has_query_image": has_query_image,
        "text_stats": {
            "min_length": min(query_lengths) if query_lengths else 0,
            "max_length": max(query_lengths) if query_lengths else 0,
            "avg_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
            "missing": missing_required["query"],
        },
        "image_count": image_count,
        "image_ratio": image_count / sample_size if sample_size > 0 else 0,
        "negative_count": negative_count,
        "negative_ratio": negative_count / sample_size if sample_size > 0 else 0,
        "required_fields": {
            "query": missing_required["query"] == 0,
            "positive": missing_required["positive"] == 0,
        },
        "missing_fields_detail": missing_required,
    }

    if skip_invalid:
        valid_count = total - sum(missing_required.values())
        stats["valid_records"] = valid_count
        stats["invalid_records"] = total - valid_count

    return stats


def print_validation_report(stats: dict[str, Any]) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print("\nRecord Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Analyzed sample: {stats['sample_size']}")

    if "valid_records" in stats:
        print(f"  Valid records: {stats['valid_records']}")
        print(f"  Invalid records: {stats['invalid_records']}")

    print("\nText Statistics:")
    text_stats = stats.get("text_stats", {})
    print(f"  Min length: {text_stats.get('min_length', 0)}")
    print(f"  Max length: {text_stats.get('max_length', 0)}")
    print(f"  Avg length: {text_stats.get('avg_length', 0):.1f}")
    print(f"  Missing: {text_stats.get('missing', 0)}/{stats['sample_size']}")

    print("\nImage Statistics:")
    print(
        f"  Image samples: {stats['image_count']}/{stats['sample_size']} ({stats['image_ratio']*100:.1f}%)"
    )
    print(f"  Has query images: {stats.get('has_query_image', False)}")

    print("\nNegative Statistics:")
    print(
        f"  Negative samples: {stats['negative_count']}/{stats['sample_size']} ({stats['negative_ratio']*100:.1f}%)"
    )

    print("\nRequired Fields:")
    for field, present in stats.get("required_fields", {}).items():
        status = "✓" if present else "✗"
        print(f"  {status} {field}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m vembed.data.validation <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    stats = validate_dataset(data_path, sample_size=100)
    print_validation_report(stats)
    print("Raw stats:", json.dumps(stats, indent=2))
