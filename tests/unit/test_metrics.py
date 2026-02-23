import numpy as np

from vembed.evaluation.metrics import compute_metrics


def test_compute_metrics_basic():
    np.random.seed(0)
    q = np.random.randn(10, 8)
    p = q + 0.1 * np.random.randn(10, 8)
    m = compute_metrics(q, p, top_k=[1, 5])
    assert "Recall@1" in m and "MRR" in m
    assert 0 <= m["Recall@1"] <= 1
