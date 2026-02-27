import json
import os
import random

import requests
from datasets import load_dataset
from tqdm import tqdm

# Karpathy Splits URLs
KARPATHY_URLS = {
    "train": "https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_train_karpathy.txt",
    "val": "https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_val_karpathy.txt",
    "test": "https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt",
}


def download_file(url, path):
    """Downloads file with optional proxy support."""
    print(f"Downloading {url}")
    proxies = None
    if os.environ.get("HTTP_PROXY"):
        proxies = {
            "http": os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("HTTPS_PROXY"),
        }
    try:
        response = requests.get(url, stream=True, timeout=30, proxies=proxies)
        response.raise_for_status()
    except Exception as e:
        print(f"Proxy download failed: {e}. Retrying without proxy...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        except Exception as e2:
            print(f"Download failed: {e2}")
            return

    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_karpathy_image_ids(split_file):
    """Parses Karpathy split file to get set of image filenames."""
    if not os.path.exists(split_file) or os.path.getsize(split_file) == 0:
        return set()

    image_ids = set()
    with open(split_file) as f:
        lines = f.readlines()
        # Skip header if present
        if lines and "image" in lines[0] and "caption" in lines[0]:
            lines = lines[1:]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Format: image_id.jpg,caption OR image_id.jpg#0 caption
            # We just need the filename part
            try:
                if ".jpg" in line:
                    filename = line.split(".jpg")[0] + ".jpg"
                    image_ids.add(filename)
            except (IndexError, ValueError):
                pass
    return image_ids


def prepare_flickr30k_dataset():
    """
    Downloads the full Flickr30k dataset and creates Train/Val/Test splits
    RESPECTING THE OFFICIAL KARPATHY SPLIT.
    """
    print("Downloading Flickr30k dataset...")

    output_dir = "data/flickr30k"
    images_dir = os.path.join(output_dir, "images")
    split_files_dir = os.path.join(output_dir, "karpathy_splits_txt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(split_files_dir, exist_ok=True)

    print("Fetching Karpathy split definitions...")
    split_image_sets = {}  # split_name -> set of filenames

    for split_name, url in KARPATHY_URLS.items():
        filename = f"flickr30k_{split_name}_karpathy.txt"
        path = os.path.join(split_files_dir, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            download_file(url, path)

        split_image_sets[split_name] = get_karpathy_image_ids(path)
        print(f"   - {split_name}: {len(split_image_sets[split_name])} images")

    dataset_name = "nlphuji/flickr30k"
    print(f"Loading {dataset_name} from HuggingFace...")

    try:
        # Load full dataset
        ds = load_dataset(dataset_name, split="test", streaming=False)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        print("Fallback: Creating dummy synthetic data for testing...")
        create_dummy_data()
        return

    print(f"Full dataset size: {len(ds)}")

    splits_data = {"train": [], "val": [], "test": []}

    processed_count = 0

    print("Partitioning data...")
    for item in tqdm(ds):
        if "filename" not in item:
            continue

        filename = item["filename"]
        image = item["image"]
        captions = item["caption"]

        target_split = None
        if filename in split_image_sets["test"]:
            target_split = "test"
        elif filename in split_image_sets["val"]:
            target_split = "val"
        elif filename in split_image_sets["train"]:
            target_split = "train"
        else:
            # Not in any official split? Skip to avoid contamination
            continue

        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            try:
                image.convert("RGB").save(image_path)
            except (OSError, ValueError):
                continue

        for cap in captions:
            entry = {
                "query": cap,
                "positive": os.path.join("images", filename),
                "image_id": filename,
            }
            splits_data[target_split].append(entry)

        processed_count += 1

    for split_name, entries in splits_data.items():
        jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
        print(f"Saving {split_name} split ({len(entries)} pairs) -> {jsonl_path}")

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    print("Dataset preparation complete.")
    print(f"   - Images processed: {processed_count}")


def create_dummy_data():
    """Creates synthetic data if download fails"""
    import numpy as np
    from PIL import Image

    output_dir = "examples/data/dummy"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "train.jsonl")

    print("Generating synthetic images...")
    with open(output_jsonl, "w") as f:
        for i in range(50):
            # Create random image
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            filename = f"dummy_{i}.jpg"
            img.save(os.path.join(images_dir, filename))

            entry = {
                "query": f"random noise image {i}",
                "positive": os.path.join("images", filename),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Dummy data created at {output_dir}")


def _process_sop_split(input_file: str, output_file: str) -> int:
    """Process SOP split file and generate JSONL dataset.

    Format of input_file (Ebay_train.txt/Ebay_test.txt):
    image_id class_id super_class_id path

    Returns: Number of pairs generated
    """
    from collections import defaultdict

    print(f"Processing {input_file} -> {output_file} ...")

    # Read and group by class_id
    class_groups: dict[int, list[str]] = defaultdict(list)

    with open(input_file, encoding="utf-8") as f:
        # Skip header if present
        lines = f.readlines()
        if lines and "image_id" in lines[0]:
            lines = lines[1:]

        for line in tqdm(lines, desc="Reading file"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
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
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved to {output_file}")
    return len(records)


def prepare_sop_i2i_dataset(
    dataset_root: str | None,
    output_dir: str,
) -> None:
    """Prepare Stanford Online Products dataset for training.

    Args:
        dataset_root: Path to SOP dataset root. If None, downloads via kagglehub.
        output_dir: Output directory for JSONL files.
    """
    if dataset_root is None:
        # Auto-download via kagglehub
        os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(os.path.dirname(__file__), "..", "data", "kagglehub"))
        import kagglehub
        dataset_root = kagglehub.dataset_download("liucong12601/stanford-online-products-dataset")
        print(f"Downloaded dataset to: {dataset_root}")

    # Find split files
    def find_file(root: str, filename: str) -> str:
        if os.path.isfile(os.path.join(root, filename)):
            return os.path.join(root, filename)
        for current, _dirs, files in os.walk(root):
            if filename in files:
                return os.path.join(current, filename)
        raise FileNotFoundError(f"Cannot find {filename} in {root}")

    train_txt = find_file(dataset_root, "Ebay_train.txt")
    test_txt = find_file(dataset_root, "Ebay_test.txt")

    os.makedirs(output_dir, exist_ok=True)

    # Process train split
    train_jsonl = os.path.join(output_dir, "train.jsonl")
    train_count = _process_sop_split(train_txt, train_jsonl)

    # Process test split (as validation)
    val_jsonl = os.path.join(output_dir, "val.jsonl")
    val_count = _process_sop_split(test_txt, val_jsonl)

    print("\n✓ SOP dataset preparation complete:")
    print(f"  - train: {train_jsonl} ({train_count} pairs)")
    print(f"  - val:   {val_jsonl} ({val_count} pairs)")


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) >= 2 else ""

    if dataset == "sop_i2i":
        prepare_sop_i2i_dataset(None, "data/stanford_online_products")
        sys.exit(0)

    if dataset == "flickr30k":
        prepare_flickr30k_dataset()
        sys.exit(0)

    sys.exit(0)
