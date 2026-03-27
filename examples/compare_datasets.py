"""Compare Flickr30k and COCO dataset statistics.

Shows:
- Data size and distribution
- Sample format
- Training recommendations
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def analyze_jsonl(filepath: str, name: str, max_samples: int = 3) -> dict:
    """Analyze JSONL dataset file.

    Args:
        filepath: Path to JSONL file
        name: Dataset name for display
        max_samples: Number of samples to show

    Returns:
        Statistics dictionary
    """
    if not os.path.exists(filepath):
        print(f"⚠ {name} file not found: {filepath}")
        return {}

    stats = {
        "total_pairs": 0,
        "unique_images": set(),
        "queries_per_image": defaultdict(int),
        "query_length": [],
        "samples": [],
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                item = json.loads(line)

                # Collect statistics
                stats["total_pairs"] += 1
                image_id = item.get("image_id", item.get("positive", "unknown"))
                stats["unique_images"].add(image_id)
                stats["queries_per_image"][image_id] += 1
                stats["query_length"].append(len(item.get("query", "")))

                # Sample data
                if i < max_samples:
                    stats["samples"].append(
                        {
                            "query": item.get("query", ""),
                            "image": item.get("positive", ""),
                        }
                    )

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    return stats


def print_comparison():
    """Print comparison of Flickr30k and COCO datasets."""

    print("\n" + "=" * 80)
    print("FLICKR30K vs COCO DATASET COMPARISON")
    print("=" * 80)

    datasets = [
        ("data/flickr30k/train.jsonl", "Flickr30k Train", "Flickr30k"),
        ("data/coco/train.jsonl", "COCO 2017 Train", "COCO 2017"),
        ("data/coco_dummy/train.jsonl", "COCO Dummy Train", "COCO Dummy"),
    ]

    results = {}

    for filepath, display_name, key in datasets:
        print(f"\n📊 Analyzing {display_name}...")
        stats = analyze_jsonl(filepath, display_name)

        if stats:
            results[key] = stats
            num_images = len(stats["unique_images"])
            num_pairs = stats["total_pairs"]
            avg_query_len = (
                sum(stats["query_length"]) / len(stats["query_length"])
                if stats["query_length"]
                else 0
            )
            captions_per_image = (
                num_pairs / num_images if num_images > 0 else 0
            )

            print(f"  ✓ Total pairs: {num_pairs:,}")
            print(f"  ✓ Unique images: {num_images:,}")
            print(f"  ✓ Captions per image: {captions_per_image:.2f}")
            print(f"  ✓ Avg query length: {avg_query_len:.0f} chars")

            print(f"\n  Sample pairs from {display_name}:")
            for i, sample in enumerate(stats["samples"]):
                print(f"    {i + 1}. Query: {sample['query'][:70]}...")
                print(f"       Image: {sample['image']}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(
        f"\n{'Dataset':<20} {'Pairs':>12} {'Images':>12} {'Caption/Img':>14} {'Avg Query':>12}"
    )
    print("-" * 70)

    for key, stats in results.items():
        num_pairs = stats["total_pairs"]
        num_images = len(stats["unique_images"])
        captions_per_image = (
            num_pairs / num_images if num_images > 0 else 0
        )
        avg_query_len = (
            sum(stats["query_length"]) / len(stats["query_length"])
            if stats["query_length"]
            else 0
        )
        print(
            f"{key:<20} {num_pairs:>12,} {num_images:>12,} {captions_per_image:>14.2f} {avg_query_len:>12.0f}"
        )

    # Print training recommendations
    print("\n" + "=" * 80)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 80)

    recommendations = {
        "Flickr30k": {
            "batch_size": "1024",
            "epochs": "1-2",
            "learning_rate": "2.0e-5",
            "gradient_cache": "Optional (24GB+ GPU)",
            "training_time": "~30 min (V100)",
        },
        "COCO 2017": {
            "batch_size": "512",
            "epochs": "3-5",
            "learning_rate": "1.0e-5",
            "gradient_cache": "Recommended (saves 4× memory)",
            "training_time": "~2-3 hours (V100)",
        },
        "COCO Dummy": {
            "batch_size": "32",
            "epochs": "1",
            "learning_rate": "5.0e-5",
            "gradient_cache": "Not needed",
            "training_time": "~2 min (any GPU)",
        },
    }

    for dataset_name, config in recommendations.items():
        if dataset_name in results:
            print(f"\n📋 {dataset_name}:")
            for key, value in config.items():
                print(f"  • {key:<18}: {value}")

    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)

    print("""
1. SIZE & SCALE:
   • Flickr30k: ~30K images (~150K image-text pairs)
   • COCO 2017: ~118K images (~590K image-text pairs) - 4× larger!

2. DATA FORMAT:
   • Both converted to same JSONL format for training
   • Column: {"query": "caption", "positive": "image_path", "image_id": int}

3. TRAINING TIME:
   • Flickr30k: ~30 min on single V100
   • COCO 2017: ~2-3 hours on single V100 (due to 4× more data)

4. MEMORY REQUIREMENTS:
   • Flickr30k: 12-16GB GPU sufficient
   • COCO 2017: 24-40GB GPU recommended (use gradient cache for 24GB)

5. CONVERGENCE:
   • Flickr30k: Faster (~1 epoch sufficient)
   • COCO 2017: Needs more epochs (3-5) for better convergence

6. DIVERSITY:
   • Flickr30k: Smaller but high-quality datasets
   • COCO 2017: Larger, more diverse, better for pretraining
    """)

    print("=" * 80)
    print("\nFor detailed guide, see: examples/COCO_GUIDE.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_comparison()
