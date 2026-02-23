import json
import os
import random
from collections import defaultdict

from tqdm import tqdm


def process_split(input_file, output_file, image_root):
    """
    Process SOP split file and generate JSONL dataset.

    Format of input_file (Ebay_train.txt/Ebay_test.txt):
    image_id class_id super_class_id path
    """
    print(f"Processing {input_file} -> {output_file} ...")

    # Read and group by class_id
    class_groups = defaultdict(list)

    with open(input_file) as f:
        # Skip header if present (first line usually "image_id class_id super_class_id path")
        lines = f.readlines()
        if "image_id" in lines[0]:
            lines = lines[1:]

        for line in tqdm(lines, desc="Reading file"):
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            # parts: [image_id, class_id, super_class_id, path]
            class_id = int(parts[1])
            rel_path = parts[3]

            class_groups[class_id].append(rel_path)

    # Generate pairs
    records = []

    for class_id, images in tqdm(class_groups.items(), desc="Generating pairs"):
        # For each image in the class, pick a positive
        for i, query_img in enumerate(images):
            # Strategy:
            # If there are other images in the same class, pick one randomly (different from query).
            # If it's the only image, use itself (self-supervision).

            if len(images) > 1:
                # Pick a random index distinct from i
                # choices = [idx for idx in range(len(images)) if idx != i]
                # pos_idx = random.choice(choices)

                # Optimization: just pick random until != i, efficient enough for small N
                pos_idx = i
                while pos_idx == i:
                    pos_idx = random.randint(0, len(images) - 1)

                pos_img = images[pos_idx]
            else:
                pos_img = query_img

            record = {"query_image": query_img, "positive": pos_img, "label": class_id}
            records.append(record)

    print(f"Generated {len(records)} pairs.")

    # Write JSONL
    with open(output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved to {output_file}")


def main():
    # Base directory for SOP dataset
    # Default: data/stanford_online_products relative to project root
    # Override with: python examples/prepare_sop_data.py --sop_root /path/to/sop
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SOP dataset for training")
    parser.add_argument(
        "--sop_root",
        type=str,
        default="data/stanford_online_products",
        help="Path to Stanford Online Products dataset root",
    )
    args = parser.parse_args()
    base_dir = args.sop_root

    if not os.path.exists(base_dir):
        print(f"Error: Dataset directory {base_dir} does not exist.")
        return

    # Train Split
    train_txt = os.path.join(base_dir, "Ebay_train.txt")
    train_jsonl = os.path.join(base_dir, "train.jsonl")

    if os.path.exists(train_txt):
        process_split(train_txt, train_jsonl, base_dir)
    else:
        print(f"Warning: {train_txt} not found.")

    # Test Split (Validation)
    test_txt = os.path.join(base_dir, "Ebay_test.txt")
    val_jsonl = os.path.join(base_dir, "val.jsonl")

    if os.path.exists(test_txt):
        process_split(test_txt, val_jsonl, base_dir)
    else:
        print(f"Warning: {test_txt} not found.")


if __name__ == "__main__":
    main()
