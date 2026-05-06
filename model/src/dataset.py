"""
dataset.py
----------
Data loading and tf.data pipeline for the Senti-DLF sentiment classifier.

Public API
----------
  DatasetConfig   : Dataclass holding all paths and pipeline hyperparameters.
  SentimentDataset: Loads raw CSV/Parquet files, tokenizes text with
                    DistilBertTokenizerFast, and returns ready-to-train
                    tf.data.Dataset objects.

Typical usage
-------------
    from model.src.dataset import DatasetConfig, SentimentDataset

    cfg = DatasetConfig()
    ds  = SentimentDataset(cfg)

    train_ds = ds.get_split("train")   # tf.data.Dataset
    val_ds   = ds.get_split("val")
    test_ds  = ds.get_split("test")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizerFast

from model.src.utils import LABEL_MAP, get_logger, set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Central configuration for paths and dataset pipeline settings.

    All path attributes default to the project-standard layout defined in
    the architecture document.  Override any field at instantiation time to
    point to custom locations.

    Attributes
    ----------
    raw_dir       : Folder holding the original, unmodified CSV / JSONL files.
    processed_dir : Folder holding cleaned Parquet splits (train/val/test).

    train_file    : Filename inside `processed_dir` for the training split.
    val_file      : Filename inside `processed_dir` for the validation split.
    test_file     : Filename inside `processed_dir` for the test split.

    text_col      : Name of the column that contains raw text.
    label_col     : Name of the column that contains string sentiment labels.

    tokenizer_name: HuggingFace model ID used for tokenization.
    max_length    : Maximum token sequence length (longer sequences are truncated).
    batch_size    : Number of samples per training batch.
    shuffle_buffer: Buffer size for tf.data shuffle (training split only).
    prefetch_size : Number of batches to prefetch (AUTOTUNE = -1).
    seed          : Random seed forwarded to set_seed() for reproducibility.
    num_labels    : Number of sentiment classes (negative / neutral / positive).
    """

    # -- Paths ----------------------------------------------------------------
    raw_dir: str = str(Path(__file__).resolve().parents[2] / "data" / "raw")
    processed_dir: str = str(Path(__file__).resolve().parents[2] / "data" / "processed")

    train_file: str = "train.parquet"
    val_file: str   = "val.parquet"
    test_file: str  = "test.parquet"

    # -- Column names ---------------------------------------------------------
    text_col: str  = "text"
    label_col: str = "label"

    # -- Tokenizer & model identity -------------------------------------------
    tokenizer_name: str = "distilbert-base-uncased"

    # -- Pipeline hyperparameters ---------------------------------------------
    max_length: int     = 128
    batch_size: int     = 32
    shuffle_buffer: int = 10_000
    prefetch_size: int  = tf.data.AUTOTUNE   # -1 lets TF decide at runtime
    seed: int           = 42
    num_labels: int     = len(LABEL_MAP)     # 3 by default

    def processed_path(self, filename: str) -> Path:
        """Return the full absolute path for a processed split file."""
        return Path(self.processed_dir) / filename

    def raw_path(self, filename: str) -> Path:
        """Return the full absolute path for a raw data file."""
        return Path(self.raw_dir) / filename


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class SentimentDataset:
    """
    Handles all data I/O and tf.data pipeline construction for Senti-DLF.

    Workflow
    --------
    1. Load a CSV or Parquet split from disk into a Pandas DataFrame.
    2. Validate columns and drop rows with missing values.
    3. Map string labels → integer IDs using LABEL_MAP from utils.
    4. Tokenize the text column with DistilBertTokenizerFast in batch mode.
    5. Wrap the result in a tf.data.Dataset with shuffle / batch / prefetch.

    Parameters
    ----------
    config : DatasetConfig
        Configuration object.  If omitted, default DatasetConfig() is used.

    Example
    -------
        cfg = DatasetConfig(batch_size=64, max_length=64)
        ds  = SentimentDataset(cfg)
        train_ds = ds.get_split("train")  # -> tf.data.Dataset
    """

    _SPLIT_MAP: Dict[str, str] = {
        "train": "train_file",
        "val":   "val_file",
        "test":  "test_file",
    }

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()
        set_seed(self.config.seed)

        log.info("Initialising tokenizer: %s", self.config.tokenizer_name)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.config.tokenizer_name
        )
        log.info(
            "DatasetConfig → max_length=%d | batch_size=%d | labels=%d",
            self.config.max_length,
            self.config.batch_size,
            self.config.num_labels,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, split: str) -> tf.data.Dataset:
        """
        Build and return a batched, prefetched tf.data.Dataset for the
        requested split.

        Args:
            split: One of "train", "val", or "test".

        Returns:
            tf.data.Dataset yielding ({
                "input_ids":      tf.Tensor [batch, max_length],
                "attention_mask": tf.Tensor [batch, max_length],
            }, labels tf.Tensor [batch])

        Raises:
            ValueError : If `split` is not one of the three expected values.
            FileNotFoundError: If the split file does not exist on disk.
        """
        if split not in self._SPLIT_MAP:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {list(self._SPLIT_MAP)}."
            )

        filename_attr = self._SPLIT_MAP[split]
        filename      = getattr(self.config, filename_attr)
        file_path     = self.config.processed_path(filename)

        log.info("Loading split '%s' from: %s", split, file_path)
        df = self._load_file(file_path)
        df = self._validate_and_clean(df)

        texts  = df[self.config.text_col].tolist()
        labels = df[self.config.label_col].map(LABEL_MAP).tolist()

        log.info(
            "Split '%s' → %d samples | label distribution: %s",
            split,
            len(texts),
            df[self.config.label_col].value_counts().to_dict(),
        )

        tf_dataset = self._build_tf_dataset(texts, labels, is_training=(split == "train"))
        return tf_dataset

    def get_all_splits(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Convenience method — returns (train_ds, val_ds, test_ds) in one call.

        Returns:
            Tuple of three tf.data.Dataset objects.

        Example:
            train_ds, val_ds, test_ds = ds.get_all_splits()
        """
        return (
            self.get_split("train"),
            self.get_split("val"),
            self.get_split("test"),
        )

    def num_classes(self) -> int:
        """Return the number of output classes (sourced from DatasetConfig)."""
        return self.config.num_labels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> pd.DataFrame:
        """
        Load a Parquet or CSV file from disk.

        Parquet is the preferred format because it preserves dtypes and is
        significantly faster than CSV for large datasets.  CSV is supported
        as a fallback for convenience during early experimentation.

        Args:
            path: Absolute Path to the data file.

        Returns:
            Pandas DataFrame with raw contents of the file.

        Raises:
            FileNotFoundError : File does not exist.
            ValueError        : Unsupported file extension.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run the preprocessing script first or check DatasetConfig paths."
            )

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df  = pd.read_csv(path, sep=sep)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Expected .parquet, .csv, or .tsv."
            )

        log.debug("Loaded %d rows from %s", len(df), path.name)
        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that required columns exist and remove unusable rows.

        Checks performed:
          - Both `text_col` and `label_col` exist in the DataFrame.
          - All label values are present in LABEL_MAP (unknown labels are dropped
            with a warning rather than raising, to handle noisy real-world data).
          - Rows where text or label is NaN/empty are dropped.

        Args:
            df: Raw DataFrame as returned by _load_file.

        Returns:
            Cleaned DataFrame ready for tokenization.

        Raises:
            KeyError: If required columns are missing entirely.
        """
        required_cols = {self.config.text_col, self.config.label_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(
                f"Required columns {missing} not found in dataset.\n"
                f"Available columns: {list(df.columns)}"
            )

        original_len = len(df)

        # Drop rows with null text or labels
        df = df.dropna(subset=[self.config.text_col, self.config.label_col])

        # Drop rows with empty strings in text
        df = df[df[self.config.text_col].str.strip().astype(bool)]

        # Drop rows whose label is not in LABEL_MAP
        unknown_labels = set(df[self.config.label_col].unique()) - set(LABEL_MAP.keys())
        if unknown_labels:
            log.warning(
                "Dropping %d rows with unrecognised labels: %s",
                df[self.config.label_col].isin(unknown_labels).sum(),
                unknown_labels,
            )
            df = df[~df[self.config.label_col].isin(unknown_labels)]

        dropped = original_len - len(df)
        if dropped:
            log.warning("Dropped %d invalid rows (%.1f%%)", dropped, 100 * dropped / original_len)

        return df.reset_index(drop=True)

    def _tokenize(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Batch-tokenize a list of strings using DistilBertTokenizerFast.

        Truncation and padding are both applied to `max_length` so that all
        sequences in a batch have an identical shape — required by tf.data.

        Args:
            texts: List of raw text strings.

        Returns:
            Dict with keys "input_ids" and "attention_mask", each a NumPy
            array of shape (len(texts), max_length) and dtype int32.
        """
        encoding = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",      # pad shorter sequences to max_length
            truncation=True,           # truncate longer sequences
            return_attention_mask=True,
            return_token_type_ids=False,  # DistilBERT has no token type IDs
            return_tensors="np",       # return NumPy arrays for tf.data compat
        )
        return {
            "input_ids":      encoding["input_ids"].astype(np.int32),
            "attention_mask": encoding["attention_mask"].astype(np.int32),
        }

    def _build_tf_dataset(
        self,
        texts: List[str],
        labels: List[int],
        is_training: bool,
    ) -> tf.data.Dataset:
        """
        Assemble the final tf.data.Dataset from tokenized inputs.

        Pipeline order:
          tokenize → from_tensor_slices → (shuffle if training) → batch → prefetch

        Shuffling is applied BEFORE batching so that every batch contains a
        random mix of samples rather than entire shuffled batches.

        Args:
            texts       : List of raw text strings.
            labels      : Corresponding integer label IDs.
            is_training : If True, shuffle the dataset each epoch.

        Returns:
            tf.data.Dataset of (features_dict, labels_tensor) tuples, batched
            and prefetched.
        """
        log.info("Tokenizing %d texts (max_length=%d)…", len(texts), self.config.max_length)
        encoded = self._tokenize(texts)

        labels_array = np.array(labels, dtype=np.int32)

        # Build dataset from in-memory tensors
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids":      encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            labels_array,
        ))

        if is_training:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer,
                seed=self.config.seed,
                reshuffle_each_iteration=True,  # fresh shuffle every epoch
            )

        dataset = (
            dataset
            .batch(self.config.batch_size, drop_remainder=False)
            .prefetch(self.config.prefetch_size)
        )

        log.info(
            "tf.data.Dataset ready | batches=%d | shuffle=%s",
            len(dataset),  # approximate — exact only for finite datasets
            is_training,
        )
        return dataset
