import json
import os

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


def _find_first_file(root_dir: str, filename: str) -> str | None:
    if os.path.isfile(os.path.join(root_dir, filename)):
        return os.path.join(root_dir, filename)
    for current, _dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(current, filename)
    return None


def _load_sop_index(
    txt_path: str,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, list[str]]]:
    class_to_paths: dict[str, list[str]] = {}
    class_to_super: dict[str, str] = {}
    super_to_classes: dict[str, list[str]] = {}
    with open(txt_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.lower().startswith("image_id"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            _image_id, class_id, super_class_id, rel_path = parts[0], parts[1], parts[2], parts[3]
            class_to_paths.setdefault(class_id, []).append(rel_path)
            class_to_super[class_id] = super_class_id
            super_to_classes.setdefault(super_class_id, []).append(class_id)
    for super_id, class_ids in super_to_classes.items():
        super_to_classes[super_id] = sorted(set(class_ids))
    return class_to_paths, class_to_super, super_to_classes


def _sample_i2i_pairs(
    class_to_paths: dict[str, list[str]],
    class_to_super: dict[str, str],
    super_to_classes: dict[str, list[str]],
    *,
    max_pairs: int,
    seed: int,
    num_hard_negatives: int,
) -> list[dict]:
    import random

    rng = random.Random(seed)
    classes = [cid for cid, paths in class_to_paths.items() if len(paths) >= 2]
    if not classes or max_pairs <= 0:
        return []

    pairs: list[dict] = []
    for _ in range(max_pairs):
        cid = rng.choice(classes)
        paths = class_to_paths[cid]
        query, positive = rng.sample(paths, 2)

        super_id = class_to_super.get(cid)
        candidate_classes = []
        if super_id is not None:
            candidate_classes = [c for c in super_to_classes.get(super_id, []) if c != cid]

        negs: list[str] = []
        if candidate_classes:
            for _n in range(num_hard_negatives):
                neg_cid = rng.choice(candidate_classes)
                neg_paths = class_to_paths.get(neg_cid, [])
                if not neg_paths:
                    continue
                negs.append(rng.choice(neg_paths))

        pairs.append(
            {
                "query_image": query,
                "positive": positive,
                "negatives": negs,
                "class_id": cid,
                "super_class_id": super_id,
            }
        )
    return pairs


def prepare_sop_i2i_dataset(
    dataset_root: str | None,
    output_dir: str,
    *,
    max_train_pairs: int = 200_000,
    max_val_pairs: int = 10_000,
    seed: int = 42,
):
    if dataset_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(data_dir, "kagglehub"))

        import kagglehub

        dataset_root = kagglehub.dataset_download("liucong12601/stanford-online-products-dataset")

    train_txt = _find_first_file(dataset_root, "Ebay_train.txt")
    test_txt = _find_first_file(dataset_root, "Ebay_test.txt")
    if train_txt is None or test_txt is None:
        raise FileNotFoundError(
            "Cannot find Ebay_train.txt / Ebay_test.txt under dataset_root. "
            f"dataset_root={dataset_root}"
        )

    base_dir = os.path.dirname(train_txt)
    local_root = os.path.join("data", "stanford_online_products")
    local_base_dir = local_root

    os.makedirs("data", exist_ok=True)
    if os.path.abspath(base_dir) != os.path.abspath(local_base_dir):
        if os.path.islink(local_base_dir):
            os.unlink(local_base_dir)

        if not os.path.exists(local_base_dir):
            try:
                os.symlink(base_dir, local_base_dir, target_is_directory=True)
                print(f"Linked dataset: {local_base_dir} -> {base_dir}")
            except OSError:
                raise RuntimeError(
                    "Failed to create symlink under data/. "
                    f"Please run: ln -s '{base_dir}' '{local_base_dir}'"
                ) from None
        else:
            print(f"Using existing local dataset path: {local_base_dir}")

    num_hard_negatives = 8

    train_index, train_class_to_super, train_super_to_classes = _load_sop_index(train_txt)
    test_index, test_class_to_super, test_super_to_classes = _load_sop_index(test_txt)

    train_pairs = _sample_i2i_pairs(
        train_index,
        train_class_to_super,
        train_super_to_classes,
        max_pairs=max_train_pairs,
        seed=seed,
        num_hard_negatives=num_hard_negatives,
    )
    val_pairs = _sample_i2i_pairs(
        test_index,
        test_class_to_super,
        test_super_to_classes,
        max_pairs=max_val_pairs,
        seed=seed + 1,
        num_hard_negatives=num_hard_negatives,
    )

    for p in train_pairs:
        p["query_image"] = os.path.join(local_base_dir, p["query_image"])
        p["positive"] = os.path.join(local_base_dir, p["positive"])
        p["negatives"] = [os.path.join(local_base_dir, n) for n in p.get("negatives", [])]

    for p in val_pairs:
        p["query_image"] = os.path.join(local_base_dir, p["query_image"])
        p["positive"] = os.path.join(local_base_dir, p["positive"])
        p["negatives"] = [os.path.join(local_base_dir, n) for n in p.get("negatives", [])]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("SOP i2i data saved:")
    print(f"   - dataset_root: {dataset_root}")
    print(f"   - image_root  : {local_base_dir} (already written as absolute paths in jsonl)")
    print(f"   - train: {train_path} ({len(train_pairs)})")
    print(f"   - val  : {val_path} ({len(val_pairs)})")
    print(
        f"   - hard_negatives_per_query: {num_hard_negatives} (same super_class, different class_id)"
    )


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) >= 2 else ""

    if dataset == "sop_i2i":
        prepare_sop_i2i_dataset(None, "data/sop_i2i")
        sys.exit(0)

    if dataset == "flickr30k":
        prepare_flickr30k_dataset()
        sys.exit(0)

    sys.exit(0)
