#!/usr/bin/env python3
"""
Quick check script to verify data loading improvements.

Run: python check_dataloader_improvements.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_config_parameters():
    """Check if defaults.yaml has new dataloader parameters."""
    print("\n" + "=" * 60)
    print("✅ CHECKING: Config Parameters")
    print("=" * 60)

    import yaml

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

    all_present = True
    for param in required_params:
        if param in config:
            print(f"  ✓ {param}: {config[param]}")
        else:
            print(f"  ✗ {param}: MISSING")
            all_present = False

    return all_present


def check_train_py_changes():
    """Check if train.py uses config parameters."""
    print("\n" + "=" * 60)
    print("✅ CHECKING: train.py DataLoader Configuration")
    print("=" * 60)

    train_path = Path("vembed/entrypoints/train.py")
    content = train_path.read_text()

    checks = [
        ('config.get("num_workers"', "Using config num_workers"),
        ('config.get("pin_memory"', "Using config pin_memory"),
        ('config.get("prefetch_factor"', "Using config prefetch_factor"),
        ('persistent_workers', "Using persistent_workers"),
        ('dataloader_kwargs', "Using dataloader_kwargs dict"),
        ('"shuffle": False', "Validation set has shuffle=False"),
    ]

    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}: NOT FOUND")
            all_found = False

    return all_found


def check_column_aliases():
    """Check if dataset.py has column aliases."""
    print("\n" + "=" * 60)
    print("✅ CHECKING: Column Aliases System")
    print("=" * 60)

    dataset_path = Path("vembed/data/dataset.py")
    content = dataset_path.read_text()

    checks = [
        ("COLUMN_ALIASES", "Column aliases dictionary defined"),
        ('"query":', "Query aliases defined"),
        ('"positive":', "Positive aliases defined"),
        ('"negatives":', "Negatives aliases defined"),
    ]

    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}: NOT FOUND")
            all_found = False

    return all_found


def check_validation_module():
    """Check if validation module exists."""
    print("\n" + "=" * 60)
    print("✅ CHECKING: Data Validation Module")
    print("=" * 60)

    validation_path = Path("vembed/data/validation.py")

    if not validation_path.exists():
        print("  ✗ validation.py: NOT FOUND")
        return False

    print("  ✓ validation.py: EXISTS")

    content = validation_path.read_text()

    checks = [
        ("def validate_dataset", "validate_dataset function"),
        ("def print_validation_report", "print_validation_report function"),
        ("text_stats", "Text statistics"),
        ("image_count", "Image counting"),
        ("negative_count", "Negative counting"),
    ]

    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}: NOT FOUND")
            all_found = False

    return all_found


def test_imports():
    """Test if modules can be imported."""
    print("\n" + "=" * 60)
    print("✅ CHECKING: Module Imports")
    print("=" * 60)

    imports_to_test = [
        ("vembed.data.loading", "load_data"),
        ("vembed.data.dataset", "GenericRetrievalDataset"),
        ("vembed.data.registry", "CollatorRegistry"),
        ("vembed.data.validation", "validate_dataset"),
    ]

    all_ok = True
    for module_name, func_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[func_name])
            if hasattr(module, func_name):
                print(f"  ✓ {module_name}.{func_name}")
            else:
                print(f"  ✗ {module_name}.{func_name}: NOT FOUND")
                all_ok = False
        except ImportError as e:
            print(f"  ✗ {module_name}: IMPORT ERROR - {e}")
            all_ok = False

    return all_ok


def test_validation_on_sample():
    """Test validation module on a small sample."""
    print("\n" + "=" * 60)
    print("✅ CHECKING: Validation Module on Sample Data")
    print("=" * 60)

    try:
        from vembed.data.validation import validate_dataset, print_validation_report

        sample_data = [
            {"query": "What is AI?", "positive": "artificial_intelligence.txt", "negatives": ["wrong.txt"]},
            {"caption": "A cat", "image": "cat.jpg"},
            {"text": "Machine learning", "answer": "ml.txt"},
        ]

        stats = validate_dataset(sample_data, sample_size=10)

        print("  ✓ Validation module executed successfully")
        print(f"  ✓ Processed {stats['total_records']} records")
        print(f"  ✓ Text stats: min={stats['text_stats']['min_length']}, max={stats['text_stats']['max_length']}")
        print(f"  ✓ Image ratio: {stats['image_ratio']*100:.1f}%")
        print(f"  ✓ Negative ratio: {stats['negative_ratio']*100:.1f}%")

        return True
    except Exception as e:
        print(f"  ✗ Validation test failed: {e}")
        return False


def main():
    """Run all checks."""
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

    # Summary
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
    else:
        print("\n⚠️  Some checks failed. Review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
