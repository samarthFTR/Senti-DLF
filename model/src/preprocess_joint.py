import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from model.src.utils import get_logger, set_seed

log = get_logger("model.src.preprocess_joint")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def bin_reddit(score: float) -> str:
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"

def process_joint():
    set_seed(42)
    raw_dir = _PROJECT_ROOT / "model" / "data" / "raw"
    proc_dir = _PROJECT_ROOT / "model" / "data" / "processed"
    
    # ---------------------------------------------------------
    # 1. Process Twitter Data
    # ---------------------------------------------------------
    log.info("Loading Twitter datasets...")
    tw_train = pd.read_csv(raw_dir / "twitter_training.csv", header=None, names=["id", "entity", "sentiment", "text"])
    tw_val = pd.read_csv(raw_dir / "twitter_validation.csv", header=None, names=["id", "entity", "sentiment", "text"])
    tw_df = pd.concat([tw_train, tw_val], ignore_index=True)
    tw_df = tw_df[["text", "sentiment"]].dropna()
    
    # Map Twitter labels (dropping Irrelevant to maintain pure signal for Sigmoid)
    tw_map = {
        "Positive": "positive",
        "Negative": "negative",
        "Neutral": "neutral"
    }
    tw_df["label"] = tw_df["sentiment"].map(tw_map)
    tw_df = tw_df.drop(columns=["sentiment"]).dropna()
    
    # ---------------------------------------------------------
    # 2. Process Reddit Data
    # ---------------------------------------------------------
    log.info("Loading Reddit datasets...")
    rd_df = pd.read_csv(raw_dir / "the-reddit-dataset-dataset-comments.csv")
    rd_df = rd_df[["body", "sentiment"]].dropna()
    rd_df = rd_df.rename(columns={"body": "text"})
    
    rd_df["label"] = rd_df["sentiment"].apply(bin_reddit)
    rd_df = rd_df.drop(columns=["sentiment"]).dropna()
    
    # ---------------------------------------------------------
    # 3. Merge and Split
    # ---------------------------------------------------------
    log.info("Merging datasets for Joint Training...")
    joint_df = pd.concat([tw_df, rd_df], ignore_index=True)
    
    # Clean texts
    joint_df = joint_df[joint_df["text"].str.strip().astype(bool)]
    
    log.info(f"Joint Class Distribution:\n{joint_df['label'].value_counts()}")
    
    log.info("Splitting dataset...")
    train_df, temp_df = train_test_split(joint_df, test_size=0.2, stratify=joint_df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    
    # Save to parquet
    log.info("Saving to Parquet format...")
    train_df.to_parquet(proc_dir / "joint_train.parquet", index=False)
    val_df.to_parquet(proc_dir / "joint_val.parquet", index=False)
    test_df.to_parquet(proc_dir / "joint_test.parquet", index=False)
    
    log.info("Joint Preprocessing Complete! Files saved as joint_train/val/test.parquet")

if __name__ == "__main__":
    process_joint()
