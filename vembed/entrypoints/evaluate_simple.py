import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.evaluation.metrics import compute_metrics
from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry
from vembed.training.data_utils import unpack_positive_batch, unpack_query_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoder_mode", type=str, default="auto")
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
    model = VisualRetrievalModel(args.model_path, encoder_mode=args.encoder_mode)
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

    collator_cls = CollatorRegistry.get(args.encoder_mode)
    if collator_cls is None:
        print(f"encoder_mode={args.encoder_mode} not registered, falling back to 'default'")
        collator_cls = CollatorRegistry.get("default")
        if collator_cls is None:
            raise ValueError(
                f"No collator registered for encoder_mode={args.encoder_mode} or 'default'. "
                f"Available: {CollatorRegistry.list_collators()}"
            )
    collator = collator_cls(processor=processor, mode="eval", retrieval_mode=args.retrieval_mode)
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

            q_inputs = unpack_query_batch(batch, args.retrieval_mode)
            p_inputs = unpack_positive_batch(batch, args.retrieval_mode)

            q_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in q_inputs.items()}
            p_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in p_inputs.items()}

            query_embs_all.append(model(**q_inputs).detach().cpu().numpy())
            positive_embs_all.append(model(**p_inputs).detach().cpu().numpy())

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
