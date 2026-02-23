import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def generate_report(
    metrics: dict[str, float],
    output_dir: str,
    query_embs: np.ndarray | None = None,
    positive_embs: np.ndarray | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.md")

    with open(report_path, "w") as f:
        f.write("# Visual Retrieval Evaluation Report\n\n")
        f.write("## Metrics\n\n| Metric | Value |\n| --- | --- |\n")
        for name, value in metrics.items():
            f.write(f"| {name} | {value:.4f} |\n")
        f.write("\n## Visualization\n\n![TSNE Plot](tsne_plot.png)\n")

    if query_embs is not None and positive_embs is not None:
        plot_tsne(query_embs, positive_embs, os.path.join(output_dir, "tsne_plot.png"))


def plot_tsne(
    query_embs: np.ndarray, positive_embs: np.ndarray, save_path: str, max_samples: int = 500
):
    if len(query_embs) > max_samples:
        idx = np.random.choice(len(query_embs), max_samples, replace=False)
        query_embs, positive_embs = query_embs[idx], positive_embs[idx]

    combined = np.concatenate([query_embs, positive_embs])
    is_query = np.array([True] * len(query_embs) + [False] * len(positive_embs))

    reduced = TSNE(n_components=2, random_state=42).fit_transform(combined)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[is_query, 0], reduced[is_query, 1], c="blue", label="Query", alpha=0.5)
    plt.scatter(reduced[~is_query, 0], reduced[~is_query, 1], c="red", label="Positive", alpha=0.5)
    plt.legend()
    plt.title("T-SNE of Query and Positive Embeddings")
    plt.savefig(save_path)
    plt.close()
