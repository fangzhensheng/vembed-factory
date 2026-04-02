"""Column mapping validation utilities for early detection of data issues."""

import logging
from typing import Any

from datasets import Dataset

from .loading import load_data

logger = logging.getLogger(__name__)

# Default column aliases (from dataset.py COLUMN_ALIASES)
DEFAULT_COLUMN_ALIASES = {
    "query": ["query", "caption", "text", "question", "instruction", "prompt", "query_text"],
    "positive": ["positive", "image", "answer", "content", "document", "paragraph", "pos_text"],
    "negatives": ["negatives", "negative_samples", "hard_negatives", "distractors"],
    "query_image": ["query_image", "source_image"],
}


def validate_column_mapping(
    data: Dataset | list[dict],
    column_mapping: dict[str, str] | None = None,
    sample_size: int = 100,
) -> dict[str, Any]:
    """Validate column mapping against sample data.

    Early detection of column mapping issues that would cause silent data loss.

    Args:
        data: HF Dataset or list of dicts
        column_mapping: Explicit column name mapping (e.g., {"query": "question"})
        sample_size: Number of records to sample for validation

    Returns:
        Report dict with:
        - valid: Boolean indicating if all columns are valid
        - issues: List of detected issues
        - column_presence: Dict of column -> presence ratio
        - column_stats: Detailed stats per column
    """
    if isinstance(data, list):
        sample = data[:sample_size]
    else:
        sample_size = min(sample_size, len(data))
        sample_indices = range(sample_size)
        sample = [data[i] for i in sample_indices]

    report: dict[str, Any] = {
        "valid": True,
        "issues": [],
        "column_presence": {},
        "column_stats": {},
    }

    # Determine columns to validate
    columns_to_check = column_mapping or {"query": "query", "positive": "positive"}

    for col_semantic_key, col_name in columns_to_check.items():
        presence = 0
        non_empty = 0
        empty_values = []

        for idx, record in enumerate(sample):
            if col_name in record:
                presence += 1
                value = record[col_name]
                if value and str(value).strip():
                    non_empty += 1
                else:
                    empty_values.append(idx)

        presence_ratio = presence / len(sample) if sample else 0
        non_empty_ratio = non_empty / presence if presence > 0 else 0

        report["column_presence"][col_name] = presence_ratio
        report["column_stats"][col_name] = {
            "presence": presence,
            "non_empty": non_empty,
            "presence_ratio": presence_ratio,
            "non_empty_ratio": non_empty_ratio,
            "empty_indices": empty_values[:5],
        }

        if presence < len(sample):
            report["valid"] = False
            missing_count = len(sample) - presence
            report["issues"].append(
                f"Column '{col_name}' (semantic: '{col_semantic_key}') not found in "
                f"{missing_count}/{len(sample)} records"
            )

        if presence > 0 and non_empty == 0:
            report["valid"] = False
            report["issues"].append(
                f"Column '{col_name}' is present in {presence} records but all values are empty"
            )

        if non_empty_ratio < 0.5:
            report["issues"].append(
                f"Column '{col_name}': Only {non_empty}/{presence} records have non-empty values "
                f"({non_empty_ratio*100:.1f}%)"
            )

    return report


def early_validate_dataset(
    data_source: str | Dataset | list[dict],
    column_mapping: dict[str, str] | None = None,
    raise_on_error: bool = True,
) -> bool:
    """Perform early validation of dataset before training.

    Should be called during dataset initialization to catch issues immediately.

    Args:
        data_source: File path, HF Dataset, or list of dicts
        column_mapping: Explicit column mapping
        raise_on_error: If True, raises ValueError on issues; if False, logs warning

    Returns:
        True if validation passed, False otherwise

    Raises:
        ValueError: If raise_on_error=True and validation fails
    """
    # Load data if it's a path
    data = load_data(data_source) if isinstance(data_source, str) else data_source

    # Validate
    report = validate_column_mapping(data, column_mapping, sample_size=min(1000, len(data)))

    if report["valid"]:
        logger.info("SUCCESS: Dataset validation passed")
        for col, presence in report["column_presence"].items():
            logger.info(
                f"  • {col}: {presence*100:.1f}% present, "
                f"{report['column_stats'][col]['non_empty_ratio']*100:.1f}% non-empty"
            )
        return True

    # Handle invalid case
    error_msg = "ERROR: Dataset validation failed:\n"
    for issue in report["issues"]:
        error_msg += f"  • {issue}\n"

    error_msg += "\nColumn statistics:\n"
    for col, stats in report["column_stats"].items():
        error_msg += (
            f"  {col}: {stats['presence']}/{len(data)} records, "
            f"{stats['non_empty']} non-empty\n"
        )

    if raise_on_error:
        raise ValueError(error_msg)
    else:
        logger.warning(error_msg)
        return False


def print_validation_report(report: dict) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    status = (
        "SUCCESS: COLUMN VALIDATION REPORT"
        if report["valid"]
        else "ERROR: COLUMN VALIDATION REPORT"
    )
    print(status)
    print("=" * 70)

    if not report["valid"]:
        print("\nERROR: Validation issues detected:\n")
        for issue in report["issues"]:
            print(f"  • {issue}")
        print()
    else:
        print("\nSUCCESS: All columns validated successfully\n")

    print("Column Statistics:")
    for col, stats in report["column_stats"].items():
        print(f"\n  {col}:")
        presence_pct = stats["presence_ratio"] * 100
        print(f"    • Present in: {stats['presence']} records ({presence_pct:.1f}%)")
        non_empty_pct = stats["non_empty_ratio"] * 100
        print(f"    • Non-empty: {stats['non_empty']} ({non_empty_pct:.1f}%)")
        if stats["empty_indices"]:
            print(f"    • Empty records at indices: {stats['empty_indices']}")

    print("\n" + "=" * 70 + "\n")
