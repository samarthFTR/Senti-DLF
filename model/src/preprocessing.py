"""
preprocessing.py
----------------
Text cleaning and dataset preparation pipeline for Senti-DLF.

Reads raw Twitter CSV files, applies a multi-step NLP cleaning pipeline,
filters labels to the three canonical classes (negative / neutral / positive),
merges the provided validation split into the training pool, and writes
stratified train / val / test Parquet files to `data/processed/`.

Public API
----------
  PreprocessingConfig : Dataclass — all paths, split ratios, and cleaning flags.
  TextPreprocessor    : Stateless cleaner; call .clean(text) on any string.
  DataPipeline        : Orchestrates the full raw → processed workflow.

Run as a script
---------------
    # from the project root (d:/Senti-DLF):
    python -m model.src.preprocessing
"""

from __future__ import annotations

import html
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Bootstrap sys.path so this module is importable both as a package and as a
# directly executed script (python -m model.src.preprocessing OR
# python model/src/preprocessing.py from the project root).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.src.utils import LABEL_MAP, get_logger, set_seed  # noqa: E402

log = get_logger(
    __name__,
    log_file=str(_PROJECT_ROOT / "logs" / "preprocessing.log"),
)


# ---------------------------------------------------------------------------
# Label normalisation map
# ---------------------------------------------------------------------------
# The raw Twitter dataset uses Title-Case labels.  "Irrelevant" has no
# equivalent in our 3-class schema so it is mapped to None and dropped.

_RAW_LABEL_MAP: Dict[str, Optional[str]] = {
    "Positive":   "positive",
    "Negative":   "negative",
    "Neutral":    "neutral",
    "Irrelevant": None,          # will be filtered out
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """
    Central configuration for the preprocessing pipeline.

    Attributes
    ----------
    raw_dir           : Folder holding the raw CSV files.
    processed_dir     : Destination folder for Parquet outputs.

    train_raw_file    : Filename of the raw training CSV inside `raw_dir`.
    val_raw_file      : Filename of the raw validation CSV inside `raw_dir`.

    train_out_file    : Output filename for the training Parquet.
    val_out_file      : Output filename for the validation Parquet.
    test_out_file     : Output filename for the test Parquet.

    text_col          : Column name to use for text in output DataFrames.
    label_col         : Column name to use for labels in output DataFrames.

    val_ratio         : Fraction of the full dataset reserved for validation.
    test_ratio        : Fraction of the full dataset reserved for test.
    seed              : Random seed for stratified splitting.

    min_text_len      : Drop rows where cleaned text is shorter than this
                        (in characters).  Catches empty / near-empty tweets.
    max_text_len      : Drop rows where cleaned text is longer than this.
                        Keeps sequences within DistilBERT's 512-token limit.

    normalise_urls    : Replace URLs with the `[URL]` placeholder token.
    normalise_mentions: Replace @mentions with the `[USER]` placeholder.
    normalise_hashtags: Strip the `#` from hashtags (keep the word).
    remove_html       : Unescape HTML entities and strip HTML tags.
    strip_extra_space : Collapse runs of whitespace into a single space.
    """

    # -- Paths ----------------------------------------------------------------
    raw_dir: str       = str(_PROJECT_ROOT / "model" / "data" / "raw")
    processed_dir: str = str(_PROJECT_ROOT / "model" / "data" / "processed")

    train_raw_file: str = "twitter_training.csv"
    val_raw_file: str   = "twitter_validation.csv"

    train_out_file: str = "train.parquet"
    val_out_file: str   = "val.parquet"
    test_out_file: str  = "test.parquet"

    # -- Column names ---------------------------------------------------------
    text_col: str  = "text"
    label_col: str = "label"

    # -- Split ratios ---------------------------------------------------------
    val_ratio: float  = 0.10   # 10 % of total for validation
    test_ratio: float = 0.10   # 10 % of total for test
    seed: int         = 42

    # -- Cleaning flags -------------------------------------------------------
    min_text_len: int       = 3
    max_text_len: int       = 512
    normalise_urls: bool    = True
    normalise_mentions: bool= True
    normalise_hashtags: bool= True
    remove_html: bool       = True
    strip_extra_space: bool = True

    def raw_path(self, filename: str) -> Path:
        """Return the absolute path for a file inside `raw_dir`."""
        return Path(self.raw_dir) / filename

    def processed_path(self, filename: str) -> Path:
        """Return the absolute path for a file inside `processed_dir`."""
        return Path(self.processed_dir) / filename


# ---------------------------------------------------------------------------
# Text Preprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Stateless text cleaner for Twitter / social-media sentiment data.

    Each cleaning step is implemented as a private method and applied in a
    fixed, deterministic order inside `clean()`.  The order matters:

        1. HTML unescape + strip tags   (before regex, so `&amp;` → `&`)
        2. URL normalisation            (before mention, so `https://t.co/…` goes first)
        3. Mention normalisation        (`@user`)
        4. Hashtag normalisation        (`#topic` → `topic`)
        5. Unicode normalisation        (decompose + strip non-ASCII control chars)
        6. Whitespace collapse

    Parameters
    ----------
    config : PreprocessingConfig
        Config object controlling which steps are active.

    Example
    -------
        cleaner = TextPreprocessor(config)
        cleaned = cleaner.clean("Check out https://t.co/xyz @elonmusk #AI!")
        # → "Check out [URL] [USER] AI !"

        cleaned_batch = cleaner.clean_batch(list_of_texts)
    """

    # Compiled regex patterns — compiled once at class level for performance
    _RE_URL      = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE,
    )
    _RE_MENTION  = re.compile(r"@\w+")
    _RE_HASHTAG  = re.compile(r"#(\w+)")
    _RE_HTML_TAG = re.compile(r"<[^>]+>")
    _RE_CTRL     = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # control chars
    _RE_SPACES   = re.compile(r"\s+")
    _RE_REPEATED = re.compile(r"(.)\1{3,}")                           # looool → lool

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to a single string.

        Args:
            text: Raw input string. Non-string values are cast to str.

        Returns:
            Cleaned string.  Returns an empty string if `text` is NaN /
            None after casting.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None and text == text else ""

        if self.config.remove_html:
            text = self._strip_html(text)

        if self.config.normalise_urls:
            text = self._RE_URL.sub("[URL]", text)

        if self.config.normalise_mentions:
            text = self._RE_MENTION.sub("[USER]", text)

        if self.config.normalise_hashtags:
            # Keep the word, remove the `#`
            text = self._RE_HASHTAG.sub(r"\1", text)

        # Unicode: NFKC normalisation folds ligatures, strips combining chars
        text = unicodedata.normalize("NFKC", text)

        # Remove ASCII control characters (keep newlines as space)
        text = self._RE_CTRL.sub("", text)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

        # Collapse repeated characters: "looooool" → "lool"
        text = self._RE_REPEATED.sub(r"\1\1\1", text)

        if self.config.strip_extra_space:
            text = self._RE_SPACES.sub(" ", text).strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a list of strings.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned strings, same length as input.
        """
        return [self.clean(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_html(self, text: str) -> str:
        """Unescape HTML entities then remove any remaining HTML tags."""
        text = html.unescape(text)          # &amp; → &, &lt; → <, etc.
        text = self._RE_HTML_TAG.sub(" ", text)
        return text


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Orchestrates the full raw → processed data workflow.

    Steps
    -----
    1. Load raw training and validation CSVs.
    2. Merge them into one pool (validation CSV is small; we re-split properly).
    3. Normalise labels (Title-Case → lower-case, drop "Irrelevant").
    4. Drop rows with null / short / overlength text.
    5. Apply TextPreprocessor to the text column.
    6. Log a cleaning report (rows dropped per reason).
    7. Stratified split → train / val / test.
    8. Save each split as Parquet.

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object.

    Example
    -------
        pipeline = DataPipeline(PreprocessingConfig())
        pipeline.run()
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config    = config or PreprocessingConfig()
        self.cleaner   = TextPreprocessor(self.config)
        set_seed(self.config.seed)
        Path(self.config.processed_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.

        Returns:
            (train_df, val_df, test_df) — the three processed DataFrames
            that were also saved to disk as Parquet files.
        """
        log.info("=" * 60)
        log.info("Senti-DLF  ·  Preprocessing Pipeline  ·  START")
        log.info("=" * 60)

        # Step 1 — Load
        df = self._load_raw()

        # Step 2 — Normalise labels
        df = self._normalise_labels(df)

        # Step 3 — Clean text
        df = self._clean_text(df)

        # Step 4 — Filter by length
        df = self._filter_by_length(df)

        # Step 5 — Split
        train_df, val_df, test_df = self._split(df)

        # Step 6 — Save
        self._save(train_df, self.config.train_out_file)
        self._save(val_df,   self.config.val_out_file)
        self._save(test_df,  self.config.test_out_file)

        log.info("=" * 60)
        log.info("Pipeline complete.")
        log.info("  Train  : %d rows → %s", len(train_df), self.config.train_out_file)
        log.info("  Val    : %d rows → %s", len(val_df),   self.config.val_out_file)
        log.info("  Test   : %d rows → %s", len(test_df),  self.config.test_out_file)
        log.info("=" * 60)

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _load_raw(self) -> pd.DataFrame:
        """
        Load the raw training and validation CSVs and merge them.

        The Twitter dataset ships with no header row.  Column positions are:
            0 → tweet id
            1 → topic / game entity
            2 → sentiment label  (Title-Case: Positive / Negative / Neutral / Irrelevant)
            3 → tweet text

        Returns:
            Combined DataFrame with columns [id, topic, label, text].
        """
        col_names = ["id", "topic", self.config.label_col, self.config.text_col]

        train_path = self.config.raw_path(self.config.train_raw_file)
        val_path   = self.config.raw_path(self.config.val_raw_file)

        log.info("Loading training CSV  : %s", train_path)
        train_df = pd.read_csv(train_path, header=None, names=col_names,
                               encoding="utf-8", on_bad_lines="skip")
        log.info("  → %d rows loaded", len(train_df))

        log.info("Loading validation CSV: %s", val_path)
        val_df = pd.read_csv(val_path, header=None, names=col_names,
                             encoding="utf-8", on_bad_lines="skip")
        log.info("  → %d rows loaded", len(val_df))

        combined = pd.concat([train_df, val_df], ignore_index=True)
        log.info("Combined pool         : %d rows total", len(combined))
        return combined

    def _normalise_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Title-Case labels to lower-case canonical labels and drop
        rows whose label is not in the three-class schema.

        Stats logged:
          - Number of 'Irrelevant' rows dropped.
          - Final label distribution.

        Args:
            df: Raw combined DataFrame.

        Returns:
            DataFrame with normalised labels, 'Irrelevant' rows removed.
        """
        log.info("--- Step: Label normalisation ---")

        before = len(df)
        df[self.config.label_col] = df[self.config.label_col].map(_RAW_LABEL_MAP)

        irrelevant_count = df[self.config.label_col].isna().sum()
        df = df.dropna(subset=[self.config.label_col]).reset_index(drop=True)

        log.info(
            "Dropped %d 'Irrelevant' rows  (%.1f%% of pool)",
            irrelevant_count,
            100 * irrelevant_count / before,
        )
        log.info("Remaining after label filter: %d rows", len(df))
        log.info("Label distribution:")
        for label, count in df[self.config.label_col].value_counts().items():
            log.info("  %-10s : %d  (%.1f%%)", label, count, 100 * count / len(df))

        return df

    def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop null texts then apply TextPreprocessor to every row.

        Stats logged:
          - Number of null rows dropped before cleaning.
          - Progress every 10 000 rows.

        Args:
            df: DataFrame with normalised labels.

        Returns:
            DataFrame with a cleaned text column.
        """
        log.info("--- Step: Text cleaning ---")

        # Drop null texts first
        null_count = df[self.config.text_col].isna().sum()
        if null_count:
            log.warning("Dropping %d rows with null text", null_count)
            df = df.dropna(subset=[self.config.text_col]).reset_index(drop=True)

        log.info("Cleaning %d texts…", len(df))
        cleaned_texts: List[str] = []

        for i, text in enumerate(df[self.config.text_col]):
            cleaned_texts.append(self.cleaner.clean(text))
            if (i + 1) % 10_000 == 0:
                log.info("  Cleaned %d / %d rows…", i + 1, len(df))

        df[self.config.text_col] = cleaned_texts
        log.info("Text cleaning complete.")
        return df

    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where the cleaned text falls outside the acceptable
        character-length range defined in PreprocessingConfig.

        Stats logged:
          - Count and percentage of too-short and too-long rows dropped.
          - Text length statistics (min / median / max / mean) after filtering.

        Args:
            df: DataFrame with cleaned text.

        Returns:
            Filtered DataFrame.
        """
        log.info("--- Step: Length filtering (min=%d, max=%d chars) ---",
                 self.config.min_text_len, self.config.max_text_len)

        lengths = df[self.config.text_col].str.len()
        before  = len(df)

        too_short = (lengths < self.config.min_text_len).sum()
        too_long  = (lengths > self.config.max_text_len).sum()

        df = df[
            (lengths >= self.config.min_text_len) &
            (lengths <= self.config.max_text_len)
        ].reset_index(drop=True)

        log.info("Dropped %d too-short rows (< %d chars)", too_short, self.config.min_text_len)
        log.info("Dropped %d too-long  rows (> %d chars)", too_long,  self.config.max_text_len)
        log.info("Remaining: %d rows  (dropped %.1f%% total)",
                 len(df), 100 * (before - len(df)) / before)

        lengths_after = df[self.config.text_col].str.len()
        log.info(
            "Text length stats → min=%d | median=%.0f | mean=%.0f | max=%d",
            lengths_after.min(),
            lengths_after.median(),
            lengths_after.mean(),
            lengths_after.max(),
        )
        return df

    def _split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified train / val / test split.

        The split is stratified on the label column so every class is
        proportionally represented in all three sets.

        Split sizes (with default ratios 80 / 10 / 10):
            train = 80 %  of `df`
            val   = 10 %  of `df`
            test  = 10 %  of `df`

        Args:
            df: Clean, filtered DataFrame.

        Returns:
            (train_df, val_df, test_df)
        """
        log.info("--- Step: Stratified split (val=%.0f%%, test=%.0f%%) ---",
                 self.config.val_ratio * 100,
                 self.config.test_ratio * 100)

        # First cut off the test set
        remaining, test_df = train_test_split(
            df,
            test_size=self.config.test_ratio,
            stratify=df[self.config.label_col],
            random_state=self.config.seed,
        )

        # From the remainder, split out validation
        adjusted_val = self.config.val_ratio / (1.0 - self.config.test_ratio)
        train_df, val_df = train_test_split(
            remaining,
            test_size=adjusted_val,
            stratify=remaining[self.config.label_col],
            random_state=self.config.seed,
        )

        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            dist = part[self.config.label_col].value_counts().to_dict()
            log.info("  %-6s : %5d rows  |  %s", name, len(part), dist)

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save a DataFrame to Parquet, keeping only the text and label columns.

        Args:
            df       : DataFrame to save.
            filename : Output filename (will be written inside `processed_dir`).
        """
        out_path = self.config.processed_path(filename)
        # Keep only the two columns the training pipeline needs
        df[[self.config.text_col, self.config.label_col]].to_parquet(
            out_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        log.info("Saved → %s  (%d rows)", out_path, len(df))


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Running preprocessing pipeline as __main__")
    pipeline = DataPipeline(PreprocessingConfig())
    pipeline.run()
