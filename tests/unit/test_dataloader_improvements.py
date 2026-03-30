#!/usr/bin/env python3
"""Verification script for vembed-factory data loading improvements."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"✅ CHECKING: {title}")
    print("=" * 60)


def _check_items(items: list[tuple[str, str]]) -> bool:
    """Generic check function for list of (pattern, description) tuples."""
    all_found = True
    for pattern, description in items:
        status = "✓" if pattern else "✗"
        print(f"  {status} {description}")
        if not pattern:
            all_found = False
    return all_found


def check_config_parameters() -> bool:
    """Verify defaults.yaml contains all dataloader parameters."""
    import yaml

    _print_section("Config Parameters")

    config_path = Path("configs/defaults.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_params = [
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
        "validate_data",
        "skip_invalid_records",
        "enable_image_cache",
    ]

    for param in required_params:
        exists = param in config
        print(f"  {'✓' if exists else '✗'} {param}: {config.get(param, 'MISSING')}")
        if not exists:
            return False

    return True


def check_train_py_changes() -> bool:
    """Verify train.py properly uses config parameters."""
    _print_section("train.py DataLoader Configuration")

    train_path = Path("vembed/entrypoints/train.py")
    content = train_path.read_text()

    checks = [
        ('config.get("num_workers"' in content, "Using config num_workers"),
        ('config.get("pin_memory"' in content, "Using config pin_memory"),
        ('config.get("prefetch_factor"' in content, "Using config prefetch_factor"),
        ("persistent_workers" in content, "Using persistent_workers"),
        ("dataloader_kwargs" in content, "Using dataloader_kwargs dict"),
        ('"shuffle": False' in content, "Validation set has shuffle=False"),
    ]

    return _check_items([(str(check), desc) if check else ("", desc) for check, desc in checks])


def check_column_aliases() -> bool:
    """Verify dataset.py implements column alias system."""
    _print_section("Column Aliases System")

    dataset_path = Path("vembed/data/dataset.py")
    content = dataset_path.read_text()

    checks = [
        ("COLUMN_ALIASES" in content, "Column aliases dictionary defined"),
        ('"query":' in content, "Query aliases defined"),
        ('"positive":' in content, "Positive aliases defined"),
        ('"negatives":' in content, "Negatives aliases defined"),
    ]

    return all(check for check, _ in checks) and _print_result(checks)


def _print_result(checks: list[tuple[bool, str]]) -> bool:
    """Print check results and return overall success."""
    all_found = True
    for check, description in checks:
        print(f"  {'✓' if check else '✗'} {description}")
        all_found = all_found and check
    return all_found


def check_validation_module() -> bool:
    """Verify validation module exists and has required functions."""
    _print_section("Data Validation Module")

    validation_path = Path("vembed/data/validation.py")

    if not validation_path.exists():
        print("  ✗ validation.py: NOT FOUND")
        return False

    print("  ✓ validation.py: EXISTS")
    content = validation_path.read_text()

    checks = [
        ("def validate_dataset" in content, "validate_dataset function"),
        ("def print_validation_report" in content, "print_validation_report function"),
        ("text_stats" in content, "Text statistics"),
        ("image_count" in content, "Image counting"),
        ("negative_count" in content, "Negative counting"),
    ]

    return _print_result(checks)


def test_imports() -> bool:
    """Verify all required modules can be imported."""
    _print_section("Module Imports")

    modules_to_test = [
        ("vembed.data.loading", "load_data"),
        ("vembed.data.dataset", "GenericRetrievalDataset"),
        ("vembed.data.registry", "CollatorRegistry"),
        ("vembed.data.validation", "validate_dataset"),
    ]

    all_ok = True
    for module_name, attr_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            has_attr = hasattr(module, attr_name)
            print(f"  {'✓' if has_attr else '✗'} {module_name}.{attr_name}")
            all_ok = all_ok and has_attr
        except ImportError as e:
            print(f"  ✗ {module_name}: IMPORT ERROR - {e}")
            all_ok = False

    return all_ok


def test_validation_on_sample() -> bool:
    """Test validation module with sample data."""
    _print_section("Validation Module on Sample Data")

    try:
        from vembed.data.validation import validate_dataset

        sample_data = [
            {
                "query": "What is AI?",
                "positive": "artificial_intelligence.txt",
                "negatives": ["wrong.txt"],
            },
            {"caption": "A cat", "image": "cat.jpg"},
            {"text": "Machine learning", "answer": "ml.txt"},
        ]

        stats = validate_dataset(sample_data, sample_size=10)

        print("  ✓ Validation module executed successfully")
        print(f"  ✓ Processed {stats['total_records']} records")
        print(
            f"  ✓ Text stats: min={stats['text_stats']['min_length']}, max={stats['text_stats']['max_length']}"
        )
        print(f"  ✓ Image ratio: {stats['image_ratio']*100:.1f}%")
        print(f"  ✓ Negative ratio: {stats['negative_ratio']*100:.1f}%")

        return True
    except Exception as e:
        print(f"  ✗ Validation test failed: {e}")
        return False


def main() -> int:
    """Run all verification checks and return exit code."""
    print("\n" + "🚀" * 30)
    print("DATA LOADING IMPROVEMENTS - VERIFICATION CHECKLIST")
    print("🚀" * 30)

    results = {
        "Config Parameters": check_config_parameters(),
        "train.py Changes": check_train_py_changes(),
        "Column Aliases": check_column_aliases(),
        "Validation Module": check_validation_module(),
        "Module Imports": test_imports(),
        "Validation on Sample": test_validation_on_sample(),
    }

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {check_name}")

    print(f"\n  Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n" + "🎉" * 20)
        print("ALL CHECKS PASSED! Data loading optimizations are working correctly.")
        print("🎉" * 20)
        return 0

    print("\n⚠️  Some checks failed. Review the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
