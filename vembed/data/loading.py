import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)


def load_data(
    path: str | Path,
    data_format: str | None = None,
    split: str = "train",
    **kwargs: Any,
) -> list[dict[str, Any]] | Dataset:
    """Load data from local files (JSONL, CSV, Parquet) or HuggingFace Hub.

    Args:
        path: File path or HuggingFace dataset name.
        data_format: Explicit format ('jsonl', 'csv', 'parquet', 'huggingface').
                     If None, inferred from file extension.
        split: Dataset split to load (for HuggingFace datasets).
        **kwargs: Additional arguments passed to the loader.

    Returns:
        A list of records (dicts) or a HuggingFace Dataset object.
    """
    path_obj = Path(path)

    if data_format is None:
        if path_obj.suffix == ".jsonl":
            data_format = "jsonl"
        elif path_obj.suffix == ".csv":
            data_format = "csv"
        elif path_obj.suffix == ".tsv":
            data_format = "tsv"
        elif path_obj.suffix == ".parquet":
            data_format = "parquet"
        elif path_obj.is_dir() or not path_obj.exists():
            # Assume it's a local or remote HF dataset
            data_format = "huggingface"
        else:
            raise ValueError(f"Cannot infer format for {path}. Please specify `data_format`.")

    logger.info("Loading data from %s with format %s", path, data_format)

    if data_format == "jsonl":
        return _load_jsonl(path_obj)

    if data_format in ("csv", "tsv"):
        sep = "\t" if data_format == "tsv" else ","
        return _load_csv(path_obj, sep)

    if data_format == "parquet":
        return _load_parquet(path_obj)

    if data_format == "huggingface":
        return _load_huggingface(str(path), split, **kwargs)

    raise ValueError(f"Unsupported data format: {data_format}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file with error handling."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d: %s", path, line_num, exc)
                continue
    return records


def _load_csv(path: Path, sep: str) -> list[dict[str, Any]]:
    """Load CSV/TSV file."""
    dataframe = pd.read_csv(path, sep=sep)
    return dataframe.to_dict("records")


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    """Load Parquet file."""
    dataframe = pd.read_parquet(path)
    return dataframe.to_dict("records")


def _load_huggingface(path: str, split: str, **kwargs: Any) -> Dataset:
    """Load dataset from HuggingFace Hub."""
    try:
        # Cast to Dataset because load_dataset can return DatasetDict or other types
        # depending on arguments, but we generally expect a specific split here.
        dataset = hf_load_dataset(path, split=split, **kwargs)
        return dataset  # type: ignore
    except Exception as exc:
        logger.error("Failed to load HF dataset %s: %s", path, exc)
        raise
