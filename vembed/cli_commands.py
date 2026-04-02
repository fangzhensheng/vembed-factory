"""CLI subcommands for vembed-factory.

Provides additional commands like list-configs, show-dataset, etc.
"""

import sys
from pathlib import Path


def list_datasets_command():
    """List available datasets from dataset_info.json."""
    examples_dir = Path(__file__).parent.parent / "examples"

    try:
        # Import config_manager from examples
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config_manager", examples_dir / "config_manager.py"
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        manager = config_module.DatasetManager(str(examples_dir))

        print("\nAvailable Datasets:")
        print("-" * 60)
        for ds_name in manager.list_datasets():
            ds_info = manager.get_dataset(ds_name)
            mode = ds_info.get("retrieval_mode", "unknown")
            file_name = ds_info.get("file_name", "N/A")
            print(f"  * {ds_name:<30} | mode: {mode:<4} | {file_name}")
        print("-" * 60)
        return 0
    except Exception as e:
        print(f"Error listing datasets: {e}", file=sys.stderr)
        return 1


def list_configs_command():
    """List available training configurations from examples directory."""
    examples_dir = Path(__file__).parent.parent / "examples"

    # Scan for YAML files organized by category
    configs_by_category = {}

    for yaml_file in sorted(examples_dir.rglob("*.yaml")):
        # Skip root level YAML files (old style)
        rel_path = yaml_file.relative_to(examples_dir)
        if len(rel_path.parts) == 1:
            continue

        parts = rel_path.parts
        if len(parts) >= 2:
            category = parts[0]
            config_name = yaml_file.stem

            if category not in configs_by_category:
                configs_by_category[category] = []
            configs_by_category[category].append(config_name)

    print("\nAvailable Training Configurations:")
    print("-" * 60)

    for category in sorted(configs_by_category.keys()):
        configs = sorted(configs_by_category[category])
        print(f"\n  [{category}/]")
        for config in configs:
            print(f"    * {config}.yaml")

    print("\n" + "-" * 60)
    print(f"Total: {sum(len(v) for v in configs_by_category.values())} configurations")
    print("\nUsage: vembed train examples/{category}/{config_name}.yaml")
    return 0


def show_dataset_command(dataset_name: str):
    """Show detailed info for a specific dataset."""
    examples_dir = Path(__file__).parent.parent / "examples"

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config_manager", examples_dir / "config_manager.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load config_manager module")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        manager = config_module.DatasetManager(str(examples_dir))
        manager.print_dataset_info(dataset_name)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Demo
    list_configs_command()
    list_datasets_command()
