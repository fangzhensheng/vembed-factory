import json
import os
import random
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simple progress
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

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
        os.environ.setdefault(
            "KAGGLEHUB_CACHE", os.path.join(os.path.dirname(__file__), "..", "data", "kagglehub")
        )
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


def prepare_msmarco_t2t_dataset(output_dir: str = "data/msmarco") -> None:
    """Download and prepare MS MARCO passage ranking dataset for t2t training.

    Uses the sentence-transformers version of MS MARCO which provides
    query-positive passage pairs in the correct format.

    Args:
        output_dir: Output directory for JSONL files.
    """
    print("Downloading MS MARCO dataset for text-to-text training...")

    os.makedirs(output_dir, exist_ok=True)

    # Load MS MARCO from HuggingFace (sentence-transformers version)
    # This version provides ready-to-use (query, positive) pairs
    try:
        print("Loading msmarco dataset from HuggingFace...")
        ds = load_dataset("sentence-transformers/msmarco-cohere-embeddings", split="train")
        print(f"Dataset size: {len(ds)}")
    except Exception as e:
        print(f"Failed to load MS MARCO: {e}")
        print("Trying alternative: creating dummy t2t data...")
        create_dummy_t2t_data(output_dir)
        return

    # Process and convert to JSONL format
    # Format: {"query_text": "...", "pos_text": "..."}
    train_data = []

    print("Converting dataset to JSONL format...")
    for item in tqdm(ds, desc="Processing"):
        # Extract query and positive passage
        # The dataset format varies, so we need to handle different field names
        query_text = item.get("query") or item.get("query_text") or item.get("question", "")
        pos_text = item.get("positive") or item.get("pos_text") or item.get("passage", "")

        if not query_text or not pos_text:
            continue

        train_data.append(
            {
                "query_text": query_text,
                "pos_text": pos_text,
            }
        )

    # Write training data
    train_jsonl = os.path.join(output_dir, "train.jsonl")
    print(f"Saving training data ({len(train_data)} pairs) -> {train_jsonl}")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Create a small validation set from the end of training data
    val_size = min(1000, len(train_data) // 10)
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]

    # Re-write train data without validation samples
    print(f"Saving training data ({len(train_data)} pairs) -> {train_jsonl}")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write validation data
    val_jsonl = os.path.join(output_dir, "val.jsonl")
    print(f"Saving validation data ({len(val_data)} pairs) -> {val_jsonl}")
    with open(val_jsonl, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n✓ MS MARCO t2t dataset preparation complete:")
    print(f"  - train: {train_jsonl} ({len(train_data)} pairs)")
    print(f"  - val:   {val_jsonl} ({len(val_data)} pairs)")


def create_dummy_t2t_data(output_dir: str) -> None:
    """Creates dummy t2t data if download fails."""
    print("Creating dummy t2t data for testing...")

    os.makedirs(output_dir, exist_ok=True)

    # Sample query-positive pairs
    dummy_data = [
        {
            "query_text": "What is the capital of France?",
            "pos_text": "The capital of France is Paris.",
        },
        {
            "query_text": "Who wrote Romeo and Juliet?",
            "pos_text": "Romeo and Juliet was written by William Shakespeare.",
        },
        {
            "query_text": "What is machine learning?",
            "pos_text": "Machine learning is a subset of artificial intelligence.",
        },
        {
            "query_text": "How do you bake a cake?",
            "pos_text": "To bake a cake, mix flour, sugar, eggs, and butter.",
        },
        {
            "query_text": "What is the meaning of life?",
            "pos_text": "The meaning of life is a philosophical question.",
        },
        {
            "query_text": "How does photosynthesis work?",
            "pos_text": "Photosynthesis converts light energy into chemical energy.",
        },
        {
            "query_text": "What is Python programming?",
            "pos_text": "Python is a high-level programming language.",
        },
        {
            "query_text": "Who discovered America?",
            "pos_text": "Christopher Columbus reached the Americas in 1492.",
        },
    ]

    # Duplicate for more training data
    train_data = dummy_data * 10  # 80 pairs
    val_data = dummy_data[:2]  # 2 pairs for validation

    train_jsonl = os.path.join(output_dir, "train.jsonl")
    val_jsonl = os.path.join(output_dir, "val.jsonl")

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✓ Dummy t2t data created at {output_dir}")
    print(f"  - train: {train_jsonl} ({len(train_data)} pairs)")
    print(f"  - val:   {val_jsonl} ({len(val_data)} pairs)")


def prepare_coco_from_huggingface(
    split: str = "train,val",
    year: int = 2017,
    output_dir: str = "data/coco",
) -> None:
    """Prepare COCO dataset from HuggingFace.

    Args:
        split: "train", "val", or comma-separated multiple splits
        year: 2014 or 2017
        output_dir: Output directory for JSONL files
    """
    if load_dataset is None:
        print("Error: HuggingFace 'datasets' library not installed")
        print("Install with: pip install datasets")
        return

    splits = [s.strip() for s in split.split(",")]
    os.makedirs(output_dir, exist_ok=True)

    dataset_id = "detection-datasets/coco"
    print(f"Loading COCO {year} from HuggingFace: {dataset_id}")

    for split_name in splits:
        try:
            print(f"\nProcessing {split_name} split...")

            # Load dataset
            config = f"{year}" if year in [2014, 2017] else "2017"
            ds = load_dataset(
                dataset_id,
                name=config,
                split=split_name,
                streaming=False,
            )

            print(f"Dataset size: {len(ds)}")

            records = []
            image_cache = {}  # Cache to avoid duplicate image processing

            for item in tqdm(ds, desc=f"Processing {split_name}"):
                # HuggingFace COCO format varies, handle common fields
                image_id = item.get("image_id")
                if not image_id:
                    continue

                image = item.get("image")
                captions = item.get("captions")

                # Fallback for different formats
                if not captions:
                    if "caption" in item:
                        captions = [item["caption"]]
                    elif "text" in item:
                        captions = [item["text"]]
                    else:
                        continue

                if not isinstance(captions, list):
                    captions = [captions]

                captions = [c.strip() for c in captions if c and isinstance(c, str)]
                if not captions:
                    continue

                # Save image (only once per image_id)
                if image_id not in image_cache and image is not None:
                    try:
                        image_dir = os.path.join(output_dir, "images")
                        os.makedirs(image_dir, exist_ok=True)
                        image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")

                        if isinstance(image, str):
                            # URL case - skip for now
                            pass
                        else:
                            # PIL Image
                            image.convert("RGB").save(image_path)

                        image_cache[image_id] = image_path
                    except (OSError, ValueError) as e:
                        print(f"Error saving image {image_id}: {e}")
                        continue

                # Create records for each caption
                relative_image_path = os.path.join("images", f"{image_id:012d}.jpg")
                for caption in captions:
                    record = {
                        "query": caption,
                        "positive": relative_image_path,
                        "image_id": image_id,
                    }
                    records.append(record)

            # Save JSONL
            if records:
                output_jsonl = os.path.join(output_dir, f"{split_name}.jsonl")
                print(f"Saving {len(records)} pairs to {output_jsonl}")

                with open(output_jsonl, "w", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"✓ {split_name}: {output_jsonl} ({len(records)} pairs)")

        except Exception as e:
            print(f"Failed to process {split_name}: {e}")
            return


def prepare_coco_official(
    dataset_root: str,
    year: int = 2017,
    output_dir: str = "data/coco",
) -> None:
    """Prepare COCO dataset from official distribution.

    Expects directory structure:
    dataset_root/
    ├── train2017/
    ├── val2017/
    ├── annotations/
    │   ├── captions_train2017.json
    │   └── captions_val2017.json

    Args:
        dataset_root: Root directory of COCO dataset
        year: 2014 or 2017
        output_dir: Output directory for JSONL files
    """
    os.makedirs(output_dir, exist_ok=True)

    # For 2014, use "train2014" etc.
    split_names = ["train", "val"]

    for split in split_names:
        # Build paths
        if year == 2014:
            images_dir = os.path.join(dataset_root, f"{split}2014")
            anno_file = os.path.join(
                dataset_root, "annotations", f"captions_{split}2014.json"
            )
        else:  # 2017
            images_dir = os.path.join(dataset_root, f"{split}2017")
            anno_file = os.path.join(
                dataset_root, "annotations", f"captions_{split}2017.json"
            )

        if not os.path.exists(anno_file):
            print(f"Skip {split}: annotation file not found at {anno_file}")
            continue

        print(f"\nProcessing {split} split (year {year})...")

        # Load annotations
        with open(anno_file, encoding="utf-8") as f:
            coco_data = json.load(f)

        # Build image_id -> captions mapping
        image_captions: dict[int, list[str]] = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            image_id = ann.get("image_id")
            caption = ann.get("caption", "").strip()
            if image_id and caption:
                image_captions[image_id].append(caption)

        print(f"Found {len(image_captions)} images with captions")

        # Generate records
        records = []
        missing_images = 0

        for image_id, captions in tqdm(image_captions.items(), desc=f"Generating pairs for {split}"):
            # Find image file (search for common extensions)
            image_path = None
            for ext in [".jpg", ".png"]:
                candidate = os.path.join(images_dir, f"{image_id:012d}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if not image_path:
                missing_images += 1
                continue

            # Create record for each caption
            relative_path = os.path.join("images", os.path.basename(image_path))
            for caption in captions:
                record = {
                    "query": caption,
                    "positive": relative_path,
                    "image_id": image_id,
                }
                records.append(record)

        # Save JSONL
        if records:
            output_jsonl = os.path.join(output_dir, f"{split}.jsonl")
            print(f"Saving {len(records)} pairs to {output_jsonl}")

            with open(output_jsonl, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"✓ {split}: {output_jsonl}")
            print(f"  - Pairs: {len(records)}")
            print(f"  - Missing images: {missing_images}")
        else:
            print(f"⚠ No records generated for {split}")

    print("\n✓ COCO dataset preparation complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python prepare_data.py flickr30k")
        print("  python prepare_data.py sop_i2i")
        print("  python prepare_data.py msmarco_t2t")
        print("  python prepare_data.py coco dummy")
        print("  python prepare_data.py coco hf [--split train,val] [--year 2017] [--output data/coco]")
        print("  python prepare_data.py coco official /path/to/coco [--year 2017] [--output data/coco]")
        sys.exit(1)

    dataset = sys.argv[1]

    if dataset == "flickr30k":
        prepare_flickr30k_dataset()
        sys.exit(0)

    if dataset == "sop_i2i":
        prepare_sop_i2i_dataset(None, "data/stanford_online_products")
        sys.exit(0)

    if dataset == "msmarco_t2t":
        prepare_msmarco_t2t_dataset()
        sys.exit(0)

    if dataset == "coco":
        if len(sys.argv) < 3:
            print("COCO usage:")
            print("  python prepare_data.py coco hf [--split train,val] [--year 2017] [--output data/coco]")
            print("  python prepare_data.py coco official /path/to/coco [--year 2017] [--output data/coco]")
            sys.exit(1)

        mode = sys.argv[2]

        if mode == "hf":
            split = "train,val"
            year = 2017
            output_dir = "data/coco"

            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--split" and i + 1 < len(sys.argv):
                    split = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--year" and i + 1 < len(sys.argv):
                    year = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                    output_dir = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1

            prepare_coco_from_huggingface(split, year, output_dir)
            sys.exit(0)

        elif mode == "official":
            if len(sys.argv) < 4:
                print("Error: official mode requires dataset root path")
                print("Usage: python prepare_data.py coco official /path/to/coco [--year 2017] [--output data/coco]")
                sys.exit(1)

            dataset_root = sys.argv[3]
            year = 2017
            output_dir = "data/coco"

            i = 4
            while i < len(sys.argv):
                if sys.argv[i] == "--year" and i + 1 < len(sys.argv):
                    year = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                    output_dir = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1

            prepare_coco_official(dataset_root, year, output_dir)
            sys.exit(0)

        else:
            print(f"Unknown COCO mode: {mode}")
            print("Available modes: hf, official")
            sys.exit(1)

    print(f"Unknown dataset: {dataset}")
    print("Available datasets: flickr30k, sop_i2i, msmarco_t2t, coco")
    sys.exit(1)
