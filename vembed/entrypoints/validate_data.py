"""CLI entrypoint for data validation."""

import argparse
import logging

from vembed.data.column_validator import print_validation_report, validate_column_mapping
from vembed.data.data_cleaner import print_cleaning_report
from vembed.data.loading import load_data
from vembed.data.validation import print_validation_report as print_data_validation_report
from vembed.data.validation import validate_dataset

logger = logging.getLogger(__name__)


def validate_data_command(args):
    """Validate dataset quality and report issues."""
    data_path = args.data_path
    sample_size = args.sample
    image_root = args.image_root
    check_images = args.check_images

    print("\n" + "=" * 70)
    print("📊 VEMBED DATA VALIDATION")
    print("=" * 70)

    print(f"\n📂 Loading data from: {data_path}")
    try:
        data = load_data(data_path)
        print(f"✓ Loaded {len(data)} records")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return 1

    print(f"\n✓ Running dataset validation (sample: {sample_size})...")
    stats = validate_dataset(data_path, sample_size=sample_size)
    print_data_validation_report(stats)

    if args.column_mapping:
        print("\n📋 Validating column mapping...")
        column_mapping = {}
        for mapping in args.column_mapping:
            key, val = mapping.split("=")
            column_mapping[key] = val

        col_report = validate_column_mapping(data, column_mapping, sample_size=sample_size)
        print_validation_report(col_report)

        if not col_report["valid"]:
            return 1

    if check_images:
        print("\n🖼️  Checking image loading...")
        from vembed.data.data_cleaner import DataCleaningConfig, validate_and_clean_data

        config = DataCleaningConfig(skip_missing_images=True, skip_invalid_paths=True)
        cleaned, clean_report = validate_and_clean_data(
            data, config=config, image_root=image_root or ""
        )
        print_cleaning_report(clean_report)

        # Warn if missing_image ratio > 90% (likely misconfigured --image-root)
        if clean_report["total"] > 0:
            missing_ratio = clean_report["issues"].get("missing_image", 0) / clean_report["total"]
            if missing_ratio > 0.9:
                print("\n⚠️  WARNING: Large number of missing images detected")
                print(
                    f"   Missing: {clean_report['issues'].get('missing_image', 0)}/{clean_report['total']} records"
                )
                print("\n   Possible causes:")
                print("   1. --image-root parameter not provided or incorrect")
                print("   2. Relative paths in data do not match actual file structure")
                print("   3. Image files not yet downloaded or in wrong location")
                print("\n   Suggested fix:")
                print(
                    "   vembed validate-data data.jsonl --check-images --image-root /path/to/images/\n"
                )

    print("=" * 70 + "\n")
    return 0


def add_validate_data_parser(subparsers):
    """Add validate-data subcommand to argparse."""
    parser = subparsers.add_parser(
        "validate-data",
        help="Validate dataset quality and detect issues",
        description="Validate dataset before training. Checks format, columns, and data quality.",
    )

    parser.add_argument(
        "data_path",
        type=str,
        help="Path to data file (JSONL, CSV, Parquet) or HuggingFace dataset",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of records to sample for validation (default: 100)",
    )

    parser.add_argument(
        "--column-mapping",
        type=str,
        nargs="+",
        help="Column mapping (e.g., query=question positive=image)",
    )

    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Check if image files exist and can be loaded",
    )

    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Root directory for relative image paths",
    )

    parser.set_defaults(func=validate_data_command)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate vembed dataset")
    parser.add_argument("data_path", help="Path to data file")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--column-mapping", nargs="+")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--image-root", default="")

    args = parser.parse_args()
    exit(validate_data_command(args))
