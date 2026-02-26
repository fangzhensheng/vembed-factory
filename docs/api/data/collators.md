# Data Collators

Batch processing and tokenization for different model architectures.

## Overview

Collators handle batching, tokenization, and format conversion for different model types (dual-encoders, VLMs, etc.). Each collator optimizes batch construction for its specific model architecture.

### Key Classes

| Class | Purpose |
|-------|---------|
| `DefaultCollator` | For standard CLIP-like models |
| `QwenCollator` | For Qwen-VL multimodal models |

## Quick Start

```python
from vembed.data.collators import DefaultCollator
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
collator = DefaultCollator(processor=processor)

# Process batch
batch = [sample1, sample2, sample3]
processed = collator(batch)
# Returns: {"input_ids": Tensor, "pixel_values": Tensor, "attention_mask": Tensor}
```

## Supported Models

### DefaultCollator
```python
# For CLIP, SigLIP, EVA-CLIP
from vembed.data.collators import DefaultCollator

collator = DefaultCollator(processor=clip_processor)
```

### QwenCollator
```python
# For Qwen-VL models
from vembed.data.collators.qwen import QwenCollator

collator = QwenCollator(processor=qwen_processor)
```

## Custom Collators

Create a custom collator for new model types:

```python
from vembed.data.collators import BaseCollator

class CustomCollator(BaseCollator):
    def __call__(self, batch):
        # Custom batching logic
        return processed_batch
```

---

::: vembed.data.collators.default.DefaultCollator
::: vembed.data.collators.qwen.QwenCollator
