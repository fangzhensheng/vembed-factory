import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from vembed.data.collators.qwen import QwenVisualRetrievalCollator

# Add Qwen scripts to path
# Use absolute paths to be safe
BASE_DIR = "/mnt/pfs/zitao_team/fangzhensheng/vembed-factory"
sys.path.append(os.path.join(BASE_DIR, "Qwen/Qwen3-VL-Embedding-2B"))
sys.path.append(BASE_DIR)

try:
    # Try direct import first if in pythonpath
    sys.path.append(os.path.join(BASE_DIR, "Qwen/Qwen3-VL-Embedding-2B"))
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
except ImportError:
    try:
        # Try appending scripts dir
        sys.path.append(os.path.join(BASE_DIR, "Qwen/Qwen3-VL-Embedding-2B/scripts"))
        from qwen3_vl_embedding import Qwen3VLEmbedder
    except ImportError as e:
        print(f"Could not import Qwen3VLEmbedder: {e}")
        sys.exit(1)


def compare_dataset_embeddings():
    # 1. Load Reference Data
    DATA_ROOT = "data/flickr30k"
    JSONL_PATH = os.path.join(DATA_ROOT, "test.jsonl")

    print(f"Loading data from {JSONL_PATH}...")
    texts = []
    image_paths = []

    with open(JSONL_PATH) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break  # Only compare first 10 items for debugging
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["query"])

            rel_path = item["positive"]
            full_path = os.path.join(DATA_ROOT, rel_path)
            image_paths.append(full_path)

    print(f"Loaded {len(texts)} items for comparison.")

    # 2. Compute Reference Embeddings
    print("Computing Reference Embeddings...")
    ref_embedder = Qwen3VLEmbedder(
        model_name_or_path="/mnt/pfs/zitao_team/fangzhensheng/vembed-factory/Qwen/Qwen3-VL-Embedding-2B",
    )
    ref_embedder.model.to("cpu")

    ref_text_inputs = [{"text": t} for t in texts]
    ref_image_inputs = [{"image": p} for p in image_paths]

    # Inspect Reference Processing
    print("\n--- Inspecting Reference Processing ---")

    # Text
    print("Inspecting Text processing...")
    # Hack to call internal methods
    ref_convs = [ref_embedder.format_model_input(text=t) for t in texts]
    ref_processed = ref_embedder._preprocess_inputs(ref_convs)

    print("Inspecting Factory Processing...")
    fac_collator = QwenVisualRetrievalCollator(processor=ref_embedder.processor)
    fac_batch_items = []
    for t, p in zip(texts, image_paths, strict=False):
        fac_batch_items.append(
            {
                "query_text": t,
                "pos_image": p,
            }
        )
    fac_processed = fac_collator(fac_batch_items)

    # Compare Inputs
    print("\n--- Comparing Inputs ---")
    if "query_input_ids" in fac_processed:
        ref_ids = ref_processed["input_ids"]
        fac_ids = fac_processed["query_input_ids"]

        if torch.equal(ref_ids, fac_ids):
            print("✅ Input IDs match!")
        else:
            print("❌ Input IDs mismatch!")
            print(f"Ref shape: {ref_ids.shape}, Fac shape: {fac_ids.shape}")
            # print diff
            diff = ref_ids != fac_ids
            print(f"Diff count: {diff.sum().item()}")
            print(f"Ref[0]: {ref_ids[0]}")
            print(f"Fac[0]: {fac_ids[0]}")

        ref_mask = ref_processed["attention_mask"]
        fac_mask = fac_processed["query_attention_mask"]
        if torch.equal(ref_mask, fac_mask):
            print("✅ Attention Masks match!")
        else:
            print("❌ Attention Masks mismatch!")
            print(f"Ref mask[0]: {ref_mask[0]}")
            print(f"Fac mask[0]: {fac_mask[0]}")
    else:
        print("Fac processed output missing query_input_ids")

    # Image
    print("Inspecting Image processing...")
    ref_img_convs = [ref_embedder.format_model_input(image=p) for p in image_paths]
    ref_img_processed = ref_embedder._preprocess_inputs(ref_img_convs)

    if "image_grid_thw" in ref_img_processed and "pos_image_grid_thw" in fac_processed:
        ref_grid = ref_img_processed["image_grid_thw"]
        fac_grid = fac_processed["pos_image_grid_thw"]

        if torch.equal(ref_grid, fac_grid):
            print("✅ Image Grid matches!")
        else:
            print("❌ Image Grid mismatch!")
            print(f"Ref Grid Shape: {ref_grid.shape}")
            print(f"Fac Grid Shape: {fac_grid.shape}")
            print(f"Ref Grid[0]: {ref_grid[0]}")
            print(f"Fac Grid[0]: {fac_grid[0]}")
    else:
        print("Missing image_grid_thw in inputs")

    # Pixel Values
    if "pixel_values" in ref_img_processed and "pos_pixel_values" in fac_processed:
        ref_pix = ref_img_processed["pixel_values"]
        fac_pix = fac_processed["pos_pixel_values"]

        # Pixel values might be float, so use allclose
        if ref_pix.shape == fac_pix.shape and torch.allclose(ref_pix, fac_pix, atol=1e-5):
            print("✅ Pixel Values match!")
        else:
            print("❌ Pixel Values mismatch!")
            print(f"Ref Pix Shape: {ref_pix.shape}")
            print(f"Fac Pix Shape: {fac_pix.shape}")
            if ref_pix.shape == fac_pix.shape:
                diff = (ref_pix - fac_pix).abs().max().item()
                print(f"Max diff: {diff}")

    # Run Model Forward Pass (using Reference Model) to see if embeddings match with same inputs
    print("\n--- Running Forward Pass on Reference Model ---")
    # We will use ref_embedder.model directly with Fac inputs to see if outputs match

    with torch.no_grad():
        # Reference Forward
        ref_out = ref_embedder.model(**ref_processed)
        ref_last_hidden = ref_out.last_hidden_state

        # Factory Inputs (need to rename keys)
        fac_inputs_renamed = {
            "input_ids": fac_processed["query_input_ids"],
            "attention_mask": fac_processed["query_attention_mask"],
            "pixel_values": fac_processed.get("query_pixel_values"),
            "image_grid_thw": fac_processed.get("query_image_grid_thw"),
        }
        # Filter None
        fac_inputs_renamed = {k: v for k, v in fac_inputs_renamed.items() if v is not None}

        # Run with bfloat16 to match original model behavior if needed
        # But we are on CPU? Qwen3-VL usually requires BF16 on CUDA.
        # Let's check model dtype.
        dtype = ref_embedder.model.dtype
        print(f"Model dtype: {dtype}")

        # Ensure inputs are correct dtype if they are floats
        if "pixel_values" in ref_processed:
            ref_processed["pixel_values"] = ref_processed["pixel_values"].to(dtype)
        if "pixel_values" in fac_inputs_renamed:
            fac_inputs_renamed["pixel_values"] = fac_inputs_renamed["pixel_values"].to(dtype)

        fac_out = ref_embedder.model(**fac_inputs_renamed)
        fac_last_hidden = fac_out.last_hidden_state

        if torch.allclose(ref_last_hidden, fac_last_hidden, atol=1e-3):  # Relax tolerance for BF16
            print("✅ Model Forward Pass matches (Hidden States)!")
        else:
            print("❌ Model Forward Pass mismatch (Hidden States)!")
            diff = (ref_last_hidden - fac_last_hidden).abs().max().item()
            print(f"Max diff: {diff}")

        # Check Pooling
        # Qwen3VLEmbedder Pooling Logic:
        # last_hidden_state = outputs.last_hidden_state
        # embeddings = last_hidden_state[:, -1] # Last token pooling?
        # Let's check Qwen3VLEmbedder.process code

        # Ref Process
        ref_emb_final = ref_embedder.process(ref_text_inputs)
        ref_q_emb = ref_emb_final
        ref_p_emb = ref_embedder.process(ref_image_inputs)

        # Fac Process (mimic pooling using correct logic)
        # The reference 'process' method does:
        # embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

        # Helper for pooling
        def pool_last_token(hidden, mask):
            flipped = mask.flip(dims=[1])
            last_one_pos = flipped.argmax(dim=1)
            col = mask.shape[1] - last_one_pos - 1
            row = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[row, col]

        fac_emb_mimic = pool_last_token(fac_last_hidden, fac_processed["query_attention_mask"])
        # Normalize
        fac_emb_mimic = F.normalize(fac_emb_mimic, p=2, dim=-1)

        if torch.allclose(ref_emb_final.cpu(), fac_emb_mimic.cpu(), atol=1e-3):
            print("✅ Pooling Logic matches!")
        else:
            print("❌ Pooling Logic mismatch!")
            diff = (ref_emb_final.cpu() - fac_emb_mimic.cpu()).abs().max().item()
            print(f"Max diff: {diff}")

    # 3. Load Factory Embeddings (Generated by run.py)
    print("Loading Factory Embeddings...")
    factory_dir = "benchmark_output_flickr30k_compare/after"
    fac_q_path = os.path.join(factory_dir, "test_query_embeddings.npy")
    fac_p_path = os.path.join(factory_dir, "test_doc_embeddings.npy")

    if not os.path.exists(fac_q_path):
        print(f"Factory embeddings not found at {fac_q_path}")
        return

    fac_q_all = np.load(fac_q_path)
    fac_p_all = np.load(fac_p_path)

    # Take first N items
    fac_q_emb = torch.from_numpy(fac_q_all[: len(texts)])
    fac_p_emb = torch.from_numpy(fac_p_all[: len(texts)])

    # 4. Compare
    print("\n=== COMPARISON ===")

    # Text
    sim_q = F.cosine_similarity(ref_q_emb, fac_q_emb)
    print(f"Text Cosine Similarities (First {len(texts)}):")
    print(sim_q)
    print(f"Mean Text Sim: {sim_q.mean().item():.6f}")

    # Image
    sim_p = F.cosine_similarity(ref_p_emb, fac_p_emb)
    print(f"Image Cosine Similarities (First {len(texts)}):")
    print(sim_p)
    print(f"Mean Image Sim: {sim_p.mean().item():.6f}")

    if sim_q.mean().item() > 0.99 and sim_p.mean().item() > 0.99:
        print("\n✅ Embeddings Match!")
    else:
        print("\n❌ Embeddings Mismatch!")


if __name__ == "__main__":
    compare_dataset_embeddings()
