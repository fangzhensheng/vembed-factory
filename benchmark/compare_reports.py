#!/usr/bin/env python3
"""Compare two benchmark report JSON files side-by-side.

Usage:
    python benchmark/compare_reports.py <before.json> <after.json> [--metrics recall@1,recall@10,recall@100]
"""

import argparse
import json
import sys
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def load_report(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"{RED}Error:{RESET} report not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _sort_key(k: str) -> tuple:
    """Sort metrics: group by prefix, then by the @K number (if any)."""
    if "@" in k:
        prefix, num = k.rsplit("@", 1)
        try:
            return (prefix, int(num))
        except ValueError:
            pass
    return (k, 0)


def compare(before: dict, after: dict, metrics: list[str] | None = None):
    # Auto-detect: all overlapping numeric keys, excluding pure metadata
    _META = {"num_images", "num_captions", "num_valid_queries", "topk"}
    if metrics is None:
        metrics = [
            k
            for k in before
            if k in after and k not in _META and _is_numeric(before[k]) and _is_numeric(after[k])
        ]
        metrics.sort(key=_sort_key)

    if not metrics:
        print(f"{RED}No overlapping metrics found.{RESET}", file=sys.stderr)
        sys.exit(1)

    max_label = max(len(k) for k in metrics)
    col_l = max(max_label + 2, 16)
    col_w = 12
    header = (
        f"  {BOLD}{'Metric':<{col_l}}{'Before':>{col_w}}{'After':>{col_w}}"
        f"{'Delta':>{col_w}}{RESET}"
    )
    separator = f"  {'─' * (col_l + col_w * 3)}"

    print()
    print(header)
    print(separator)

    prev_prefix = None
    for k in metrics:
        # Insert a blank line between groups (e.g. t2i_* → i2t_*)
        prefix = (
            k.rsplit("_recall@", 1)[0]
            if "_recall@" in k
            else k.rsplit("_", 1)[0] if "_" in k else ""
        )
        if prev_prefix is not None and prefix != prev_prefix:
            print()
        prev_prefix = prefix

        b = float(before.get(k, 0))
        a = float(after.get(k, 0))
        delta = a - b
        delta_pp = delta * 100

        if delta > 0:
            sign, color = "+", GREEN
        elif delta < 0:
            sign, color = "", RED
        else:
            sign, color = " ", DIM

        print(
            f"  {k:<{col_l}}{b:>{col_w}.4f}{a:>{col_w}.4f}"
            f"{color}{sign}{delta_pp:>{col_w - 1}.2f}pp{RESET}"
        )

    print(separator)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark reports")
    parser.add_argument("before", help="Path to the BEFORE report JSON")
    parser.add_argument("after", help="Path to the AFTER report JSON")
    parser.add_argument(
        "--metrics",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated metric names (default: auto-detect recall@K)",
    )
    args = parser.parse_args()

    before = load_report(args.before)
    after = load_report(args.after)
    compare(before, after, args.metrics)


if __name__ == "__main__":
    main()
