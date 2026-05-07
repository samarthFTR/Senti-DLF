"""
utils.py
--------
Shared utility helpers for the Senti-DLF project.

Provides:
  - get_logger      : Standardised console + file logger
  - set_seed        : Reproducibility seeding for Python, NumPy, TensorFlow
  - LABEL_MAP       : Canonical label → integer mapping
  - load_label_map  : Inverse integer → label lookup
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Label mapping — single source of truth across dataset, trainer, and metrics
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[str, int] = {
    "negative": 0,
    "neutral":  1,
    "positive": 2,
    "mixed":    3,
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


def load_label_map(inverse: bool = False) -> Dict:
    """
    Return the label mapping used across the project.

    Args:
        inverse: If True, return {int → str}. Otherwise return {str → int}.

    Returns:
        Dict containing the requested mapping.
    """
    return ID_TO_LABEL if inverse else LABEL_MAP


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Build and return a logger that writes to stdout (always) and optionally
    to a log file.

    Args:
        name     : Logger name — typically __name__ of the calling module.
        log_file : Optional absolute path to a .log file.
        level    : Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.

    Example:
        >>> log = get_logger(__name__, log_file="logs/train.log")
        >>> log.info("Dataset loaded successfully.")
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when the logger is retrieved more than once
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 so Unicode symbols (→, ·) don't crash
    # on Windows terminals that default to cp1252.
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure
    reproducible training runs.

    Args:
        seed: Integer seed value (default 42).

    Example:
        >>> set_seed(2026)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger = get_logger(__name__)
    logger.info("Global seed set to %d", seed)
