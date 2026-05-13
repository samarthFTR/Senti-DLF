"""
trainer.py
----------
Orchestrates the training loop for the Senti-DLF model.

Brings together `dataset.py` (for the tf.data.Dataset splits) and `model.py`
(for the compiled Keras model), and executes the training loop with
standard callbacks: EarlyStopping, ModelCheckpoint, and TensorBoard.

Public API
----------
  TrainerConfig : Dataclass for training hyperparameters (epochs, paths).
  ModelTrainer  : Handles the model.fit() lifecycle and artifact saving.

Run as a script
---------------
    python -m model.src.trainer
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tensorflow as tf

# Bootstrap sys.path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.src.dataset import DatasetConfig, SentimentDataset
from model.src.model import ModelConfig, SentimentModel
from model.src.utils import get_logger, set_seed

log = get_logger(__name__, log_file=str(_PROJECT_ROOT / "logs" / "training.log"))


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    Configuration for the training loop and artifact saving.

    Attributes
    ----------
    epochs          : Maximum number of passes over the training dataset.
    patience        : How many epochs to wait for validation loss improvement
                      before stopping early.
    checkpoint_dir  : Where to save the best model weights during training.
    log_dir         : Where to save TensorBoard logs.
    final_model_dir : Where to save the final compiled model (SavedModel format)
                      for backend serving.
    """
    epochs: int   = 5
    patience: int = 2
    
    checkpoint_dir: str  = str(_PROJECT_ROOT / "model" / "saved_models" / "checkpoints")
    log_dir: str         = str(_PROJECT_ROOT / "model" / "saved_models" / "logs")
    final_model_dir: str = str(_PROJECT_ROOT / "model" / "saved_models" / "bert_v1")


# ---------------------------------------------------------------------------
# Main Trainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Executes the training and evaluation loop.

    Example
    -------
        trainer = ModelTrainer()
        history = trainer.train()
    """

    def __init__(
        self,
        trainer_config: Optional[TrainerConfig] = None,
        dataset_config: Optional[DatasetConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.config = trainer_config or TrainerConfig()
        
        # Initialise dependencies
        self.dataset = SentimentDataset(dataset_config)
        self.model_wrapper = SentimentModel(model_config)
        self.model: Optional[tf.keras.Model] = None

        # Ensure output directories exist
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.final_model_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> tf.keras.callbacks.History:
        """
        Builds the model, loads data splits, and executes training.

        Returns:
            Keras History object containing train/val metrics per epoch.
        """
        log.info("=" * 60)
        log.info("Senti-DLF  ·  Training Loop  ·  START")
        log.info("=" * 60)

        # Load datasets (using the new Joint parquets for combined training)
        log.info("Loading Joint datasets for fresh training...")
        train_ds = self.dataset.get_split("train")
        val_ds   = self.dataset.get_split("val")
        test_ds  = self.dataset.get_split("test")

        # 2. Build Model
        self.model = self.model_wrapper.build()
        
        # 3. Setup Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, f"{'deberta' if 'deberta' in self.config.final_model_dir else 'bert'}_best_weights.ckpt"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.log_dir,
                histogram_freq=1,
            ),
        ]

        # 4. Train
        # Weighted loss: strongly upweight negative to counter class imbalance.
        # negative(0) gets 3x, neutral(1) gets 1x, positive(2) gets 1x.
        # Previous run had neutral double-weighted which starved the negative class.
        if "deberta" in self.config.final_model_dir:
            class_weight = {0: 3.0, 1: 1.0, 2: 1.0}
        else:
            class_weight = {0: 1.0, 1: 2.0, 2: 1.0}  # BERT: keep neutral weighted

        log.info("Starting model.fit() for %d epochs...", self.config.epochs)
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

        # 5. Evaluate on Test Set
        log.info("Evaluating on held-out test set...")
        test_metrics = self.model.evaluate(test_ds, return_dict=True)
        log.info("Test Metrics: %s", test_metrics)

        # 6. Save final model
        log.info("Saving final model to: %s", self.config.final_model_dir)
        self.model.save(self.config.final_model_dir)

        log.info("=" * 60)
        log.info("Training Loop  ·  COMPLETE")
        log.info("=" * 60)

        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Senti-DLF Model")
    parser.add_argument("--model", type=str, choices=["bert", "deberta"], default="bert",
                        help="Choose base model: 'bert' (default) or 'deberta'")
    args = parser.parse_args()

    # Standard hyperparams for RTX 3050 (4GB VRAM)
    # Batch size 8 for BERT with real LoRA (all 12 layers have active gradients)
    d_cfg = DatasetConfig(batch_size=8)
    d_cfg.train_file = "joint_train.parquet"
    d_cfg.val_file = "joint_val.parquet"
    d_cfg.test_file = "joint_test.parquet"
    
    if args.model == "deberta":
        d_cfg.tokenizer_name = "microsoft/deberta-v3-base"
        d_cfg.batch_size = 4  # DeBERTa needs small batches on 4GB VRAM
        m_cfg = ModelConfig(
            model_name="microsoft/deberta-v3-base",
            learning_rate=2e-5,
            label_smoothing=0.05,     # reduce smoothing so negative signal is sharper
            use_lora=False,           # partial fine-tuning (top 6 layers)
            num_unfreeze_layers=6,    # was 3 — more capacity for 3-class discrimination
        )
        t_cfg = TrainerConfig(
            epochs=4,                 # one extra epoch to converge with more unfrozen layers
            final_model_dir=str(Path(d_cfg.processed_dir).parent / "saved_models" / "deberta_v2")
        )
        log.info("Configured for DeBERTa-v3-base retraining (partial fine-tuning, top 6 layers).")
    else:
        d_cfg.tokenizer_name = "bert-base-uncased"
        d_cfg.batch_size = 8  # Real LoRA (all 12 layers) needs smaller batches on 4GB VRAM
        m_cfg = ModelConfig(
            model_name="bert-base-uncased",
            learning_rate=3e-5,
            label_smoothing=0.1
        )
        t_cfg = TrainerConfig(
            epochs=3,
            final_model_dir=str(Path(d_cfg.processed_dir).parent / "saved_models" / "bert_v2")
        )
        log.info("Configured for BERT-base-uncased training.")

    trainer = ModelTrainer(t_cfg, d_cfg, m_cfg)
    trainer.train()
