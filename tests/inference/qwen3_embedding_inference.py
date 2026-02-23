import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add model directory to sys.path to import scripts
MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
if os.path.isdir(MODEL_ID):
    sys.path.append(MODEL_ID)
elif os.path.isdir(os.path.join(os.getcwd(), MODEL_ID)):
    sys.path.append(os.path.join(os.getcwd(), MODEL_ID))

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
except ImportError:
    # Fallback: try importing directly if scripts is not a package
    try:
        from qwen3_vl_embedding import Qwen3VLEmbedder
    except ImportError:
        print(f"Error: Could not import Qwen3VLEmbedder from {MODEL_ID}/scripts")
        print("Please ensure the model directory contains scripts/qwen3_vl_embedding.py")
        sys.exit(1)

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

            # 处理图片路径
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


def main():
    # 配置
    MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
    # Use relative path or env var for data root
    DATA_ROOT = os.environ.get("FLICKR_ROOT", "./data/flickr30k")
    IMAGE_ROOT = DATA_ROOT  # Image root is the same as data root for Flickr30k structure
    JSONL_PATH = os.path.join(DATA_ROOT, "test.jsonl")
    BATCH_SIZE = 128

    print(f"Model: {MODEL_ID}")

    # 初始化 Qwen3VLEmbedder
    # 推荐开启 flash_attention_2 以获得加速
    model = Qwen3VLEmbedder(
        model_name_or_path=MODEL_ID,
        # torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2"
    )

    # 1. 加载数据
    if not os.path.exists(JSONL_PATH):
        print(f"Error: {JSONL_PATH} not found.")
        return

    texts, image_paths, image_ids = load_flickr30k_data(JSONL_PATH, IMAGE_ROOT)
    unique_image_paths, caption_idx_to_image_idx = get_unique_images(image_paths, image_ids)

    print(f"Loaded {len(texts)} captions, {len(unique_image_paths)} unique images.")

    # 2. 提取 Text Embeddings
    text_embeddings = []
    # 构造 Qwen3VLEmbedder 需要的 input 格式: [{"text": "..."}]
    text_inputs = [{"text": t} for t in texts]

    print("Encoding texts...")
    for i in tqdm(range(0, len(text_inputs), BATCH_SIZE)):
        batch_inputs = text_inputs[i : i + BATCH_SIZE]
        # model.process 返回的是 tensor
        with torch.no_grad():
            batch_embs = model.process(batch_inputs)
            text_embeddings.append(batch_embs.cpu())

    text_embeddings = torch.cat(text_embeddings, dim=0)  # [5000, D]

    # 3. 提取 Image Embeddings
    image_embeddings = []
    # 构造 Qwen3VLEmbedder 需要的 input 格式: [{"image": "path/to/img"}]
    image_inputs = [{"image": p} for p in unique_image_paths]

    print("Encoding images...")
    # 图像处理可能显存占用较大，batch size 减半
    IMG_BATCH_SIZE = max(1, BATCH_SIZE // 2)

    for i in tqdm(range(0, len(image_inputs), IMG_BATCH_SIZE)):
        batch_inputs = image_inputs[i : i + IMG_BATCH_SIZE]
        with torch.no_grad():
            batch_embs = model.process(batch_inputs)
            image_embeddings.append(batch_embs.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)  # [1000, D]

    # 4. 计算指标 (Recall@K)
    # text_embeddings: [5000, D]
    # image_embeddings: [1000, D]

    # 计算相似度矩阵: [5000, 1000]
    print("Computing similarity matrix...")
    # 确保归一化 (虽然 Qwen3VLEmbedder 内部可能已经归一化，但再做一次无妨)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    sim_matrix = torch.matmul(text_embeddings.float(), image_embeddings.float().T).numpy()

    # 计算 Recall@K
    # caption_idx_to_image_idx[i] 是第 i 个 query 的正确答案在 unique_images 中的索引

    ks = [1, 5, 10]
    recalls = {k: 0 for k in ks}
    num_queries = len(texts)

    print("Calculating metrics...")
    for i in range(num_queries):
        # 第 i 个 query 的相似度分数
        scores = sim_matrix[i]
        target_idx = caption_idx_to_image_idx[i]

        # 获取排序后的索引 (从大到小)
        # argsort 返回从小到大，[::-1] 反转
        sorted_indices = np.argsort(scores)[::-1]

        # 检查 target_idx 是否在前 k 个中
        for k in ks:
            if target_idx in sorted_indices[:k]:
                recalls[k] += 1

    for k in ks:
        recalls[k] = (recalls[k] / num_queries) * 100
        print(f"R@{k}: {recalls[k]:.2f}")


if __name__ == "__main__":
    main()
