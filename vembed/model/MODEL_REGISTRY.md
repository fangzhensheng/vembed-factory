# Model Support Matrix

This document describes which models are supported, which backend and processor to use for each, and how LoRA fine-tuning works.

## Model-Backend-Processor Mapping

| Model Pattern | Backend | Processor | Example | LoRA | Description |
|---|---|---|---|---|---|
| `qwen3-vl*` | `qwen3_vl` | `qwen3_vl` | `Qwen/Qwen3-VL-7B-Instruct` | ✓ | Multimodal VLM with vision input |
| `qwen3-embedding*` | `qwen3_embedding` | `qwen3_embedding` | `Qwen/Qwen3-Embedding` | ✓ | Text-only embedding model |
| `siglip*` | `auto` | `siglip` | `google/siglip-base-patch16-256` | ✓ | Dual-encoder (text + image) |
| `clip*` | `composed` | `default` | `openai/clip-vit-base-patch32` | ✓ | Dual-encoder requiring separate loading |
| `gemma*` | `vlm_generic` | `default` | `google/gemma-2-27b` | ✓ | Generic VLM (Gemma-VL) |
| `internvl*` | `vlm_generic` | `default` | `OpenGVLab/InternVL2` | ✓ | Generic VLM (InternVL) |
| Other models | `auto` | `default` | Any HuggingFace model | ✓ | Automatic inference via AutoModel |

## Backend Descriptions

### `auto` (Default)
- Uses `AutoModel.from_pretrained()` for generic HuggingFace model loading
- Supports any standard transformer architecture
- Best for models not explicitly optimized in vembed-factory
- Can infer pooling and embedding extraction automatically

### `qwen3_vl`
- Specialized backend for Qwen3-VL multimodal models
- Handles special `image_grid_thw` format (image patch metadata)
- Implements proper token pooling for vision-language tasks
- Required for models using Qwen3-VL architecture

### `qwen3_embedding`
- Specialized backend for Qwen3-Embedding text-only models
- Optimized text encoder pooling
- Supports Multi-Vector Retrieval (MRL) for better accuracy

### `vlm_generic`
- Wrapper for general causal VLMs (Gemma-VL, InternVL, Qwen-VL v1, etc.)
- Uses last non-padding token as the sentence embedding (standard for decoder-only models)
- Supports both text and vision inputs
- Includes projection head support for dimension reduction

### `composed`
- For dual-encoder models with separate text and image towers
- Requires separate `text_model_name` and `image_model_name` in config
- Each tower can use different model families (e.g., BERT + SigLIP)
- Both towers support LoRA fine-tuning

## Processor Descriptions

### `qwen3_vl`
- Handles image patch metadata (`image_grid_thw`)
- Sets `padding_side='right'` for proper sequence alignment
- Handles variable-length image patches from dynamic image resolutions

### `qwen3_embedding`
- Standard text processor for Qwen3-Embedding
- Optimizes tokenization for text-only tasks

### `siglip`
- Forces `padding='max_length'` with `max_length=64` for text
- Critical for consistent embeddings in SigLIP models
- Handles image-only inference when text is absent

### `default`
- Automatically loads via `AutoProcessor.from_pretrained()`
- Falls back processor for models without specialized loaders
- Includes `trust_remote_code=True` for models with custom code

## Auto-Detection Rules

If you don't specify `encoder_mode` or processor mode, vembed-factory auto-detects based on model name:

```
Model name contains:        → Backend    → Processor
─────────────────────────────────────────────────
"qwen3-vl" or "qwen3_vl"   → qwen3_vl   → qwen3_vl
"qwen3-embedding"          → qwen3_embedding → qwen3_embedding
"siglip"                   → auto       → siglip
Other                      → auto       → default
```

For composed models, specify both `text_model_name` and `image_model_name`:
```yaml
encoder_mode: composed
text_model_name: bert-base-uncased
image_model_name: google/siglip-base-patch16-256
```

## Configuration Examples

### Text-to-Image Retrieval (CLIP-style, Dual-Encoder)
```yaml
# For CLIP or SigLIP models
model_name: openai/clip-vit-base-patch32
encoder_mode: auto  # or omit for auto-detect
batch_size: 128
```

### Qwen3-VL Multimodal (Vision-Language Model)
```yaml
model_name: Qwen/Qwen3-VL-7B-Instruct
encoder_mode: qwen3_vl  # Required for Qwen3-VL
batch_size: 16
```

### Text-Only Embedding (Qwen3-Embedding)
```yaml
model_name: Qwen/Qwen3-Embedding
# encoder_mode omitted, auto-detected as qwen3_embedding
batch_size: 256
```

### Composed Dual-Encoder (Custom Text + Image Models)
```yaml
encoder_mode: composed
text_model_name: sentence-transformers/all-minilm-l6-v2
image_model_name: google/siglip-base-patch16-256
batch_size: 64
```

## LoRA Fine-Tuning Support

All backends support LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:

```yaml
use_lora: true
lora_r: 8                    # Rank (default: 16)
lora_alpha: 16               # Alpha (default: 32, usually 2*r)
lora_dropout: 0.1            # Dropout (default: 0.05)
lora_target_modules:         # Target modules (default shown)
  - q_proj
  - v_proj
  - dense
```

### LoRA Loading Behavior

**Training time**: LoRA adapters are applied in-memory during training via `apply_lora()`, allowing full model training with reduced parameters.

**Inference time**: If a saved model directory contains `adapter_config.json` and `adapter_model.bin`, LoRA weights are automatically:
1. Loaded via `PeftModel.from_pretrained()`
2. Merged into the backbone (`merge_and_unload()`)
3. Used for inference without wrapping overhead

### Composed Model LoRA

For composed models, LoRA is applied to both towers independently:

```yaml
encoder_mode: composed
text_model_name: bert-base-uncased
image_model_name: google/siglip-base-patch16-256
use_lora: true
lora_r: 8
# Result: Both text and image encoders will have LoRA adapters
```

## Choosing Your Setup

### Quick Decision Tree

1. **Do you have separate text and image models?**
   - Yes → Use `composed` backend
   - No → Continue

2. **Is it a Qwen3 model?**
   - Qwen3-VL (multimodal) → Use `qwen3_vl` backend
   - Qwen3-Embedding (text) → Use `qwen3_embedding` backend
   - No → Continue

3. **Is it SigLIP, CLIP, or another dual-encoder?**
   - SigLIP → Use `auto` backend, siglip processor will auto-detect
   - CLIP → Can use `auto` backend, or `composed` if you want separate loading
   - No → Continue

4. **Otherwise**
   - Use `auto` backend (default)
   - Processor will auto-detect or use default

## Processor Auto-Discovery

The processor resolution system tries loaders in this order:

1. **If `encoder_mode` specified** → Try that loader first
2. **Smart name matching** → Try `qwen3_vl`, `qwen3_embedding`, `siglip` in order based on model name
3. **Fallback** → Use `AutoProcessor.from_pretrained()`

To see which processor is being used, check the training logs for messages like:
```
Loading processor: qwen3_vl
Loading processor: siglip (from default loader)
Processing with: AutoProcessor
```

## Adding Custom Models

To add support for a new model:

1. **If standard AutoModel works**: Just use `encoder_mode: auto`, no changes needed

2. **If needs custom backbone**: Create a new backend file in `vembed/model/backbones/my_model.py`:
   ```python
   from ..base import BaseEmbeddingModel
   from ..registry import ModelRegistry

   @ModelRegistry.register("my_model")
   class MyModelEmbedding(BaseEmbeddingModel):
       def __init__(self, config):
           super().__init__(config)
           # Custom initialization
           self.backbone = ...

       def forward(self, **kwargs):
           # Custom forward
           pass
   ```

3. **If needs custom processor**: Create file in `vembed/model/processors/my_model.py`:
   ```python
   from .registry import ProcessorRegistry

   @ProcessorRegistry.register("my_model")
   class MyModelProcessor:
       @staticmethod
       def match(model_name: str) -> bool:
           return "my-model" in model_name.lower()

       @staticmethod
       def load(model_name: str, encoder_mode=None):
           # Custom processor loading
           pass
   ```

The system will auto-discover both backend and processor via dynamic imports.

## Troubleshooting

### "ProcessorRegistry.resolve failed" warning
- Processor couldn't be loaded, falling back to AutoProcessor
- Check model name matches expected patterns or specify explicitly
- Some models may have custom preprocessing requirements

### "Failed to load LoRA adapter" warning
- `adapter_config.json` found but loading failed
- Check `peft` is installed: `pip install peft`
- Verify `adapter_model.bin` exists in the same directory

### Model loads but embeddings are wrong shape
- Wrong pooling method for this model
- Try adjusting `pooling_method` config: `"cls"`, `"mean"`, `"last_token"`, `"none"`

### CUDA OOM with specific model
- Models like Qwen3-VL-32B need distributed training
- Use FSDP: set `use_fsdp: true` and `num_gpus: 8`
- Or reduce batch size and enable gradient accumulation
