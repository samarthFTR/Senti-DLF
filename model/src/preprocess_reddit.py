import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from model.src.utils import get_logger, set_seed

log = get_logger("model.src.preprocess_reddit")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def bin_sentiment(score: float) -> str:
    if score < -0.05:
        return "negative"
    elif score > 0.05:
        return "positive"
    else:
        return "neutral"

def process_reddit():
    set_seed(42)
    raw_path = _PROJECT_ROOT / "model" / "data" / "raw" / "the-reddit-dataset-dataset-comments.csv"
    proc_dir = _PROJECT_ROOT / "model" / "data" / "processed"
    
    log.info("Loading Reddit comments...")
    df = pd.read_csv(raw_path)
    
    # We only need the text (body) and the continuous sentiment score
    df = df[["body", "sentiment"]].dropna()
    
    # Rename for pipeline compatibility
    df = df.rename(columns={"body": "text"})
    
    # Bin continuous sentiment into categories
    log.info("Binning continuous sentiment scores into categorical classes...")
    df["label"] = df["sentiment"].apply(bin_sentiment)
    
    # Drop original sentiment float column
    df = df.drop(columns=["sentiment"])
    
    log.info(f"Reddit Class Distribution:\n{df['label'].value_counts()}")
    
    # Stratified split: 80% train, 10% val, 10% test
    log.info("Splitting dataset...")
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    
    # Save to parquet
    log.info("Saving to Parquet format...")
    train_df.to_parquet(proc_dir / "reddit_train.parquet", index=False)
    val_df.to_parquet(proc_dir / "reddit_val.parquet", index=False)
    test_df.to_parquet(proc_dir / "reddit_test.parquet", index=False)
    
    log.info("Reddit Preprocessing Complete! Files saved as reddit_train/val/test.parquet")

if __name__ == "__main__":
    process_reddit()