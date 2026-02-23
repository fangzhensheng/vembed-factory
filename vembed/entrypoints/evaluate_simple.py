import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vembed.data.dataset import VisualRetrievalCollator, VisualRetrievalDataset
from vembed.evaluation.metrics import compute_metrics
from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="t2i",
        choices=["t2i", "i2i", "t2t"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = ProcessorRegistry.resolve(args.model_path)
    model = VisualRetrievalModel(args.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    dataset = VisualRetrievalDataset(
        data_source=args.data_path,
        processor=processor,
        image_root="",
        mode="eval",
        column_mapping=None,
    )
    collator = VisualRetrievalCollator(processor, mode="eval")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    query_embs_all, positive_embs_all = [], []

    with torch.no_grad():
        for batch in loader:
            device = next(model.parameters()).device

            if args.retrieval_mode.startswith("i"):
                q_batch = {"pixel_values": batch["query_pixel_values"].to(device)}
            else:
                q_batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }

            if args.retrieval_mode.endswith("t"):
                p_batch = {
                    "input_ids": batch["pos_input_ids"].to(device),
                    "attention_mask": batch["pos_attention_mask"].to(device),
                }
            else:
                p_batch = {"pixel_values": batch["pixel_values"].to(device)}

            query_embs_all.append(model(**q_batch).detach().cpu().numpy())
            positive_embs_all.append(model(**p_batch).detach().cpu().numpy())

    query_embs = np.concatenate(query_embs_all)
    positive_embs = np.concatenate(positive_embs_all)

    metrics = compute_metrics(query_embs, positive_embs, labels=None, top_k=[1, 5, 10])

    report_lines = ["# Evaluation Report", ""]
    report_lines += [f"- {k}: {v:.4f}" for k, v in metrics.items()]
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
