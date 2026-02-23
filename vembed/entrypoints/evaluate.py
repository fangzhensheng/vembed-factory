import argparse
import os
import sys

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vembed.data.dataset import VisualRetrievalCollator, VisualRetrievalDataset
from vembed.evaluation.metrics import compute_metrics
from vembed.evaluation.report import generate_report
from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    # Load vembed specific config if exists to ensure consistent evaluation
    config_path = os.path.join(args.model_path, "vembed_config.json")
    model_kwargs = {}
    if os.path.exists(config_path):
        import json

        with open(config_path) as f:
            vembed_config = json.load(f)
            # Keys to forward to VisualRetrievalModel
            for key in ["pooling_method", "projection_dim", "topk_tokens", "use_mrl", "mrl_dims"]:
                if key in vembed_config:
                    model_kwargs[key] = vembed_config[key]
        accelerator.print(f"Loaded vembed config: {model_kwargs}")

    model = VisualRetrievalModel(args.model_path, **model_kwargs)

    try:
        processor = ProcessorRegistry.resolve(args.model_path)
    except Exception:
        processor = None

    dataset = VisualRetrievalDataset(
        data_source=args.data_path,
        processor=processor,
        image_root=args.image_root,
        mode="eval",
    )
    collator = VisualRetrievalCollator(processor, mode="eval")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    all_query_embs, all_positive_embs = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            q_emb = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            p_emb = model(pixel_values=batch["pixel_values"])

            all_query_embs.append(accelerator.gather_for_metrics(q_emb).cpu().numpy())
            all_positive_embs.append(accelerator.gather_for_metrics(p_emb).cpu().numpy())

    if accelerator.is_local_main_process:
        query_embs = np.concatenate(all_query_embs)
        positive_embs = np.concatenate(all_positive_embs)

        metrics = compute_metrics(query_embs, positive_embs)
        print("Metrics:", metrics)

        generate_report(metrics, args.output_dir, query_embs, positive_embs)
        print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
