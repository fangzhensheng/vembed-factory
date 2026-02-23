#!/usr/bin/env python
import argparse
import logging
import os
import subprocess
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.bench_datasets.registry import discover_dataset_modules

# Setup simple logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_cmd(cmd):
    """Run a shell command."""
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    modules = discover_dataset_modules()
    dataset_choices = list(modules.keys())

    # Create a parser that only looks for the dataset name first
    # We use parse_known_args to handle the first positional argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("dataset", nargs="?", choices=dataset_choices, help="Dataset to run")

    args, _ = pre_parser.parse_known_args()

    if not args.dataset or args.dataset in ["-h", "--help"]:
        # Show help if no dataset provided or help requested
        print("Usage: python benchmark/run.py <dataset> [options]")
        print("\nAvailable datasets:")
        for name in dataset_choices:
            print(f"  - {name}")
        sys.exit(0 if args.dataset in ["-h", "--help"] else 1)

    dataset_name = args.dataset
    if dataset_name not in modules:
        logger.error(f"Dataset '{dataset_name}' not found.")
        sys.exit(1)

    module = modules[dataset_name]

    # Now create the full parser
    parser = argparse.ArgumentParser(
        description=f"Run benchmark for {dataset_name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=dataset_choices, help="Dataset to run")

    # Add dataset-specific arguments
    if hasattr(module, "add_run_arguments"):
        module.add_run_arguments(parser)

    # Parse all arguments
    args = parser.parse_args()

    # Run the benchmark
    if not hasattr(module, "run"):
        logger.error(f"Module {dataset_name} does not have a 'run' function.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    module.run(args, run_cmd=run_cmd, script_dir=script_dir)


if __name__ == "__main__":
    main()
