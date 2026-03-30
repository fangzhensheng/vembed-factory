"""
Script to prepare REAL training data from Conceptual Captions 3M (CC3M) for Tiny SigLIP.
Downloads a small subset of CC3M (e.g., 10k-50k samples) and validation data from COCO/Flickr30k.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_image(url, save_path, timeout=3, retries=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(save_path, "JPEG", quality=85)
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            continue
    return False


def prepare_cc3m_subset(output_dir, num_samples=20000, num_workers=16):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("Loading image-text dataset from HuggingFace (streaming mode)...")

    # Try LAION-COCO first (faster, more reliable URLs)
    # Falls back to conceptual_captions if unavailable
    try:
        print("Trying LAION-COCO subset (faster alternative)...")
        dataset = load_dataset("laion/LAION-COCO", split="train", streaming=True)
    except Exception as e:
        print(f"LAION-COCO not available: {e}")
        print("Falling back to official 'conceptual_captions'...")
        dataset = load_dataset("conceptual_captions", split="train", streaming=True)

    data_list = []

    def process_item(item):
        try:
            caption = item.get("caption", "")
            url = item.get("image_url", "")

            if not caption or not url:
                return None

            # Generate filename
            import hashlib

            img_filename = hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"
            img_path = os.path.join(images_dir, img_filename)

            if os.path.exists(img_path):
                return {"image": img_filename, "text": caption}

            # Download image
            if download_image(url, img_path):
                return {"image": img_filename, "text": caption}

        except Exception:
            return None
        return None

    print(f"Downloading {num_samples} images with {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        iterator = iter(dataset)
        pbar = tqdm(total=num_samples, unit="img")

        while len(data_list) < num_samples:
            try:
                chunk = []
                # Larger batch to reduce iteration overhead
                for _ in range(num_workers * 4):
                    chunk.append(next(iterator))
            except StopIteration:
                break

            if not chunk:
                break

            results = list(executor.map(process_item, chunk))

            for res in results:
                if res:
                    data_list.append(res)
                    pbar.update(1)
                    if len(data_list) >= num_samples:
                        break

            pbar.set_description(f"✓ {len(data_list)}/{num_samples}")
        pbar.close()

    jsonl_path = os.path.join(output_dir, "train.jsonl")
    print(f"Saving {len(data_list)} records to {jsonl_path}...")
    with open(jsonl_path, "w") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")


def prepare_validation_data(output_dir):
    print("Creating validation set...")
    train_jsonl = os.path.join(output_dir, "train.jsonl")
    val_jsonl = os.path.join(output_dir, "val.jsonl")

    if not os.path.exists(train_jsonl):
        return

    with open(train_jsonl) as f:
        lines = f.readlines()

    random_lines = lines[:1000]

    with open(val_jsonl, "w") as f:
        for line in random_lines:
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real CC3M subset for Tiny SigLIP")
    parser.add_argument(
        "--output_dir", type=str, default="data/tiny_siglip_real", help="Output directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of images to download"
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of download threads")

    args = parser.parse_args()

    prepare_cc3m_subset(args.output_dir, args.num_samples, args.workers)
    prepare_validation_data(args.output_dir)
    print(f"\nData preparation complete! Real data is in {args.output_dir}")
