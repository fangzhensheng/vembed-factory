# Visual Retrieval Model

Main facade for training and inference.

## Overview

`VisualRetrievalModel` combines text and image encoders into a single unified model for training and inference.

### Architecture

```
Input (Text + Image)
    ↓
Text Encoder  +  Image Encoder
    ↓
Embeddings
    ↓
Similarity / Loss
```

---

::: vembed.model.modeling.VisualRetrievalModel
