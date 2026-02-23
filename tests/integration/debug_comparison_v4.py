import sys

import torch
import torch.nn.functional as F
from PIL import Image

# Add Qwen scripts to path
sys.path.append("/mnt/pfs/zitao_team/fangzhensheng/vembed-factory/Qwen/Qwen3-VL-Embedding-2B")
sys.path.append("/mnt/pfs/zitao_team/fangzhensheng/vembed-factory")

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
except ImportError:
    try:
        sys.path.append(
            "/mnt/pfs/zitao_team/fangzhensheng/vembed-factory/Qwen/Qwen3-VL-Embedding-2B/scripts"
        )
        from qwen3_vl_embedding import Qwen3VLEmbedder
    except ImportError:
        print("Could not import Qwen3VLEmbedder")
        sys.exit(1)

from transformers import AutoProcessor

from vembed.data.collators.qwen import QwenVisualRetrievalCollator
from vembed.model.backbones.qwen3 import Qwen3EmbeddingModel


def compare_implementations():
    model_path = "Qwen/Qwen3-VL-Embedding-2B"
    print(f"Comparing implementations for: {model_path}")

    # Use CPU to avoid OOM when loading two models
    device = "cpu"
    print(f"Using device: {device}")

    # Force single thread for CPU to avoid massive slowdown
    torch.set_num_threads(1)

    # 1. Reference Implementation
    print("\n--- Reference Implementation ---")
    ref_embedder = Qwen3VLEmbedder(model_name_or_path=model_path)
    ref_embedder.model.eval()
    ref_embedder.model.to(device)

    # 2. Factory Implementation
    print("\n--- Factory Implementation ---")
    # Simulate config dict
    config = {
        "model_name_or_path": model_path,
        "attn_implementation": "eager",
        "torch_dtype": "float32",  # Use float32 to avoid precision issues during comparison
    }
    factory_model = Qwen3EmbeddingModel(config)
    factory_model.eval()
    factory_model.backbone.to(device)  # Backbone is the AutoModel

    # IMPORTANT: Share weights to isolate data processing differences
    # Qwen3VLEmbedder wraps Qwen3VLForEmbedding which wraps Qwen3VLModel
    # Qwen3EmbeddingModel wraps AutoModel (Qwen3VLModel)

    print("Sharing weights from Reference to Factory...")
    # ref_embedder.model is Qwen3VLForEmbedding
    # ref_embedder.model.model is Qwen3VLModel
    factory_model.backbone = ref_embedder.model.model

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    collator = QwenVisualRetrievalCollator(processor, mode="eval")

    # Test Inputs
    text = "A dog running."
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))

    # --- Compare Text Encoding ---
    print("\n=== TEXT COMPARISON ===")

    # Reference
    ref_input = [{"text": text}]
    # We want to inspect intermediate values, so let's call internal methods if possible,
    # but Qwen3VLEmbedder.process does it all.
    with torch.no_grad():
        ref_emb = ref_embedder.process(ref_input)  # [1, D]

    # Factory
    factory_item = {"query_text": text, "pos_image": None}
    batch = collator([factory_item])

    # Check Input IDs
    # To check ref input ids, we need to hack/intercept.
    # But we can reconstruct ref logic:
    ref_convs = [ref_embedder.format_model_input(text=text)]
    ref_processed = ref_embedder._preprocess_inputs(ref_convs)
    ref_input_ids = ref_processed["input_ids"].to(device)

    fac_input_ids = batch["query_input_ids"].to(device)

    print(f"Ref Input IDs Shape: {ref_input_ids.shape}")
    print(f"Fac Input IDs Shape: {fac_input_ids.shape}")

    if torch.equal(ref_input_ids, fac_input_ids):
        print("✅ Input IDs MATCH")
    else:
        print("❌ Input IDs MISMATCH")
        print(f"Ref: {ref_input_ids[0].tolist()}")
        print(f"Fac: {fac_input_ids[0].tolist()}")

    # Factory Forward
    inputs = {
        "input_ids": batch["query_input_ids"].to(device),
        "attention_mask": batch["query_attention_mask"].to(device),
    }
    with torch.no_grad():
        fac_emb = factory_model(**inputs)

    # Compare Embeddings
    sim = F.cosine_similarity(ref_emb.to(device), fac_emb)
    print(f"Text Embedding Cosine Similarity: {sim.item():.6f}")
    if sim.item() > 0.9999:
        print("✅ Text Embeddings MATCH")
    else:
        print("❌ Text Embeddings MISMATCH")

    # --- Compare Image Encoding ---
    print("\n=== IMAGE COMPARISON ===")

    # Reference
    ref_input = [{"image": image}]
    with torch.no_grad():
        ref_emb = ref_embedder.process(ref_input)

    # Factory
    # Note: generate_embeddings.py uses 'query_image' for i2t query? Or 'pos_image'?
    # It uses _split_query_pos. If retrieval_mode='i2t', q_batch comes from query_pixel_values etc.
    # In collator, we put image in query_image if it's a query.
    factory_item = {"query_text": None, "query_image": image, "pos_image": None}
    batch = collator([factory_item])

    # Check Grid
    ref_convs = [ref_embedder.format_model_input(image=image)]
    ref_processed = ref_embedder._preprocess_inputs(ref_convs)
    ref_grid = ref_processed["image_grid_thw"].to(device)

    fac_grid = batch["query_image_grid_thw"].to(device)

    print(f"Ref Grid Shape: {ref_grid.shape}")
    print(f"Fac Grid Shape: {fac_grid.shape}")
    print(f"Ref Grid: {ref_grid.tolist()}")
    print(f"Fac Grid: {fac_grid.tolist()}")

    if torch.equal(ref_grid, fac_grid):
        print("✅ Grid MATCHES")
    else:
        print("❌ Grid MISMATCH")

    # Factory Forward
    inputs = {
        "input_ids": batch["query_input_ids"].to(device),
        "attention_mask": batch["query_attention_mask"].to(device),
        "pixel_values": batch["query_pixel_values"].to(device),
        "image_grid_thw": batch["query_image_grid_thw"].to(device),
    }

    with torch.no_grad():
        fac_emb = factory_model(**inputs)

    sim = F.cosine_similarity(ref_emb.to(device), fac_emb)
    print(f"Image Embedding Cosine Similarity: {sim.item():.6f}")
    # Test Inputs (Batch Size = 2)
    print("\n=== BATCH COMPARISON ===")
    text1 = "A dog running."
    text2 = "A very long sentence to force padding."
    texts = [text1, text2]

    # Reference
    print("Reference Encoding...")
    ref_input = [{"text": t} for t in texts]
    with torch.no_grad():
        ref_embs = ref_embedder.process(ref_input)  # [2, D]

    # Factory
    print("Factory Encoding...")
    fac_items = [{"query_text": t, "pos_image": None} for t in texts]
    batch = collator(fac_items)

    # Check Input IDs and Attention Mask
    fac_input_ids = batch["query_input_ids"].to(device)
    fac_attention_mask = batch["query_attention_mask"].to(device)

    print(f"Fac Input IDs Shape: {fac_input_ids.shape}")
    print(f"Fac Input IDs:\n{fac_input_ids}")
    print(f"Fac Attention Mask:\n{fac_attention_mask}")

    # Check if padding is correct (right padding)
    # The second sequence is longer, so the first one should be padded at the end
    # len1 = len(ref_embedder.processor.tokenizer(text1)["input_ids"])
    # len2 = len(ref_embedder.processor.tokenizer(text2)["input_ids"])  # Should be longer

    # Note: Processor adds system prompt, so length is longer than raw text tokenization
    # But relative length should hold.

    # Factory Forward
    inputs = {
        "input_ids": batch["query_input_ids"].to(device),
        "attention_mask": batch["query_attention_mask"].to(device),
    }
    with torch.no_grad():
        fac_embs = factory_model(**inputs)

    # Compare
    sim = F.cosine_similarity(ref_embs.to(device), fac_embs)
    print(f"Batch Embedding Cosine Similarity: {sim.tolist()}")

    if all(s > 0.9999 for s in sim.tolist()):
        print("✅ Batch Embeddings MATCH")
    else:
        print("❌ Batch Embeddings MISMATCH")


if __name__ == "__main__":
    compare_implementations()
