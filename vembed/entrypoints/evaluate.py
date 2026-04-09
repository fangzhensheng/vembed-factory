import argparse
import json
import logging
import os

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from vembed.data.dataset import VisualRetrievalDataset
from vembed.data.registry import CollatorRegistry
from vembed.evaluation.metrics import compute_metrics
from vembed.evaluation.report import generate_report
from vembed.model.modeling import VisualRetrievalModel
from vembed.model.processors import ProcessorRegistry
from vembed.training.data_utils import unpack_positive_batch, unpack_query_batch

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--encoder_mode", type=str, default="auto")
    parser.add_argument("--retrieval_mode", type=str, default="t2i")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    config_path = os.path.join(args.model_path, "vembed_config.json")
    model_kwargs: dict = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            vembed_config = json.load(f)
        for key in ["pooling_method", "projection_dim", "topk_tokens", "use_mrl", "mrl_dims"]:
            if key in vembed_config:
                model_kwargs[key] = vembed_config[key]
        if "encoder_mode" in vembed_config and args.encoder_mode == "auto":
            args.encoder_mode = vembed_config["encoder_mode"]
        if "retrieval_mode" in vembed_config and args.retrieval_mode == "t2i":
            args.retrieval_mode = vembed_config["retrieval_mode"]
        accelerator.print(f"Loaded vembed config: {model_kwargs}")

    model = VisualRetrievalModel(args.model_path, encoder_mode=args.encoder_mode, **model_kwargs)

    try:
        processor = ProcessorRegistry.resolve(args.model_path)
    except (ValueError, KeyError, ImportError):
        processor = None

    dataset = VisualRetrievalDataset(
        data_source=args.data_path,
        processor=processor,
        image_root=args.image_root,
        mode="eval",
    )

    collator_cls = CollatorRegistry.get(args.encoder_mode)
    if collator_cls is None:
        accelerator.print(
            f"encoder_mode={args.encoder_mode} not registered, falling back to 'default'"
        )
        collator_cls = CollatorRegistry.get("default")
        if collator_cls is None:
            raise ValueError(
                f"No collator registered for encoder_mode={args.encoder_mode} or 'default'. "
                f"Available: {CollatorRegistry.list_collators()}"
            )
    collator = collator_cls(processor=processor, mode="eval", retrieval_mode=args.retrieval_mode)
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
            q_inputs = unpack_query_batch(batch, args.retrieval_mode)
            p_inputs = unpack_positive_batch(batch, args.retrieval_mode)
            q_emb = model(**q_inputs)
            p_emb = model(**p_inputs)

            all_query_embs.append(accelerator.gather_for_metrics(q_emb).cpu().numpy())
            all_positive_embs.append(accelerator.gather_for_metrics(p_emb).cpu().numpy())

    if accelerator.is_local_main_process:
        query_embs = np.concatenate(all_query_embs)
        positive_embs = np.concatenate(all_positive_embs)

        metrics = compute_metrics(query_embs, positive_embs)
        print(f"Metrics: {metrics}")

        generate_report(metrics, args.output_dir, query_embs, positive_embs)
        print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
