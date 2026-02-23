import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# 设置代理
os.environ["https_proxy"] = "http://10.198.7.60:7890"
os.environ["http_proxy"] = "http://10.198.7.60:7890"


def load_flickr30k_data(jsonl_path, image_root):
    """
    加载 Flickr30k 测试数据
    返回:
    - texts: 5000 个 caption
    - image_paths: 5000 个 对应的图片路径 (会有重复)
    - image_ids: 5000 个 image_id 用于去重
    """
    texts = []
    image_paths = []
    image_ids = []

    print(f"Loading data from {jsonl_path}...")
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["query"])
            # jsonl 中的 positive 是 "images/xxx.jpg"，需要拼上 root
            # 假设 image_root 是 /.../vembed-factory/data/flickr30k
            # 那么全路径是 /.../vembed-factory/data/flickr30k/images/xxx.jpg
            # 但这里 item['positive'] 已经是 "images/xxx.jpg"
            # 如果 image_root 指向 data/flickr30k，直接 join 即可
            # 注意: 有些 dataset 实现可能是 os.path.basename 处理
            # 这里我们用最稳妥的方式：
            rel_path = item["positive"]
            full_path = os.path.join(image_root, rel_path)
            image_paths.append(full_path)

            # 获取 image_id 用于去重 (filename)
            img_id = item.get("image_id", os.path.basename(rel_path))
            image_ids.append(img_id)

    return texts, image_paths, image_ids


def get_unique_images(image_paths, image_ids):
    """
    从 5000 个 pair 中提取 1000 张唯一图片
    返回:
    - unique_paths: 1000 个路径
    - caption_idx_to_image_idx: [5000] 的映射，表示第 i 个 caption 对应第几个 unique image
    """
    unique_ids = []
    unique_paths = []
    id_to_idx = {}

    caption_idx_to_image_idx = []

    for path, img_id in zip(image_paths, image_ids, strict=False):
        if img_id not in id_to_idx:
            id_to_idx[img_id] = len(unique_ids)
            unique_ids.append(img_id)
            unique_paths.append(path)

        caption_idx_to_image_idx.append(id_to_idx[img_id])

    return unique_paths, np.array(caption_idx_to_image_idx)


def _pool_hidden(hidden, attention_mask=None):
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _extract_embedding(output, attention_mask=None):
    if isinstance(output, torch.Tensor):
        return output
    for name in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return _pool_hidden(hidden, attention_mask)
    raise TypeError(f"Unsupported output type: {type(output)}")


def main():
    # 配置
    MODEL_ID = "google/siglip-base-patch16-224"
    # Use relative path or env var for data root
    DATA_ROOT = os.environ.get("FLICKR_ROOT", "./data/flickr30k")
    JSONL_PATH = os.path.join(DATA_ROOT, "test.jsonl")
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")

    # 1. 加载模型
    print("Loading model and processor...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 2. 准备数据
    texts, image_paths, image_ids = load_flickr30k_data(JSONL_PATH, DATA_ROOT)
    unique_image_paths, gt_indices = get_unique_images(image_paths, image_ids)

    print(f"Captions: {len(texts)}")
    print(f"Unique Images: {len(unique_image_paths)}")

    # 3. 提取文本特征
    print("Encoding texts...")
    text_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = (
                model.get_text_features(**inputs)
                if hasattr(model, "get_text_features")
                else model(**inputs)
            )
            emb = _extract_embedding(outputs, inputs.get("attention_mask"))
            emb = F.normalize(emb, p=2, dim=-1)
            text_embs.append(emb.cpu())
    text_embs = torch.cat(text_embs, dim=0)  # [5000, D]

    # 4. 提取图片特征
    print("Encoding images...")
    image_embs = []
    # 这里用 PIL 加载，确保 convert RGB
    for i in tqdm(range(0, len(unique_image_paths), BATCH_SIZE)):
        batch_paths = unique_image_paths[i : i + BATCH_SIZE]
        batch_images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                batch_images.append(Image.new("RGB", (224, 224)))

        inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = (
                model.get_image_features(**inputs)
                if hasattr(model, "get_image_features")
                else model(**inputs)
            )
            emb = _extract_embedding(outputs)
            emb = F.normalize(emb, p=2, dim=-1)
            image_embs.append(emb.cpu())
    image_embs = torch.cat(image_embs, dim=0)  # [1000, D]

    # 5. 计算相似度 & 指标
    print("Computing metrics...")
    # Similarity: [5000, 1000]
    sim_matrix = torch.matmul(text_embs, image_embs.T)

    # T2I Recall
    # 对于每个 text，看 ground truth image index 是否在 topk 中
    # gt_indices: [5000]

    ks = [1, 5, 10]
    t2i_res = {}

    # top-k indices: [5000, 10]
    _, topk_indices = torch.topk(sim_matrix, k=10, dim=1)

    gt_tensor = torch.tensor(gt_indices).unsqueeze(1)  # [5000, 1]

    for k in ks:
        # Check if gt is in top-k columns
        hits = (topk_indices[:, :k] == gt_tensor).any(dim=1).float().mean().item()
        t2i_res[f"R@{k}"] = hits

    print("\nText-to-Image Retrieval (Zero-Shot):")
    for k, v in t2i_res.items():
        print(f"  {k}: {v:.4f}")

    # I2T Recall
    # 对于每个 image，找对应的 5 个 captions
    # Sim matrix 转置: [1000, 5000]
    sim_i2t = sim_matrix.T

    # 每个 image 对应的正确 caption indices
    # gt_indices 是 [5000]，内容是 0~999
    # 需要反向映射：image_idx -> [cap_idx1, cap_idx2, ...]
    img_to_caps = {}
    for cap_idx, img_idx in enumerate(gt_indices):
        if img_idx not in img_to_caps:
            img_to_caps[img_idx] = []
        img_to_caps[img_idx].append(cap_idx)

    i2t_res = {}
    _, topk_indices_i2t = torch.topk(sim_i2t, k=10, dim=1)  # [1000, 10]

    for k in ks:
        hits = 0
        for img_idx in range(len(unique_image_paths)):
            correct_caps = set(img_to_caps[img_idx])  # {c1, c2, c3, c4, c5}
            retrieved = topk_indices_i2t[img_idx, :k].tolist()
            # 只要有一个对的就算对 (R@K 定义通常如此)
            if any(c in correct_caps for c in retrieved):
                hits += 1
        i2t_res[f"R@{k}"] = hits / len(unique_image_paths)

    print("\nImage-to-Text Retrieval (Zero-Shot):")
    for k, v in i2t_res.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
