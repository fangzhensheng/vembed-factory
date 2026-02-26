# Evaluation Reports

Automated report generation for benchmark results.

## Overview

The report module generates comprehensive evaluation reports in HTML/Markdown format with performance tables, visualizations, and comparisons.

## Quick Start

```python
from vembed.evaluation.report import generate_report

report = generate_report(
    query_embs=query_embeddings,
    gallery_embs=gallery_embeddings,
    model_name="CLIP ViT-B/32",
    dataset_name="Flickr30K"
)

# Save report
report.save("evaluation_report.html")
```

---

::: vembed.evaluation.report
