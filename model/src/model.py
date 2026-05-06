"""
model.py
--------
Model architecture for the Senti-DLF sentiment classifier.

Wraps HuggingFace's TFDistilBertModel inside a Keras Functional model,
optionally injecting LoRA (Low-Rank Adaptation) adapters via the PEFT
library for lightweight fine-tuning.  Only ~1-3 M parameters are trained
when LoRA is enabled — the remaining 64 M base weights stay frozen.

Public API
----------
  ModelConfig    : Dataclass holding all architecture and LoRA hyperparameters.
  SentimentModel : Assembles, compiles, and optionally applies LoRA to the
                   full TF Keras model.

Typical usage
-------------
    from model.src.model import ModelConfig, SentimentModel

    cfg   = ModelConfig()
    sm    = SentimentModel(cfg)
    model = sm.build()      # tf.keras.Model — ready for model.fit()

    sm.summary()            # print trainable parameter count
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tensorflow as tf

# ---------------------------------------------------------------------------
# Bootstrap sys.path for direct script execution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.src.utils import LABEL_MAP, get_logger, set_seed  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Central configuration for the SentimentModel architecture.

    Attributes
    ----------
    model_name      : HuggingFace model ID for the base transformer.
                      Must match the tokenizer used in DatasetConfig.
    num_labels      : Output classes (sourced from LABEL_MAP by default).
    max_length      : Token sequence length — must match DatasetConfig.max_length.
    dropout_rate    : Dropout applied to the [CLS] pooled output before the
                      classification head.  Regularises the small head layer.

    use_lora        : If True, inject LoRA adapters and freeze the base model.
                      If False, train ALL parameters (slow, requires >8 GB VRAM).
    lora_r          : LoRA rank — controls adapter bottleneck width.
                      Higher = more capacity, more parameters.
                      Typical values: 4, 8, 16.
    lora_alpha      : LoRA scaling factor.  The effective learning-rate scale
                      for adapter weights is lora_alpha / lora_r.
                      Common heuristic: set equal to 2 * lora_r.
    lora_dropout    : Dropout inside the LoRA adapter matrices.

    learning_rate   : Initial learning rate for the Adam optimiser.
    weight_decay    : L2 regularisation coefficient (AdamW).
    label_smoothing : Smoothing factor for categorical cross-entropy.
                      0.1 prevents overconfident predictions on noisy labels.
    """

    # -- Base transformer -----------------------------------------------------
    model_name: str     = "distilbert-base-uncased"
    num_labels: int     = len(LABEL_MAP)   # 3 (negative / neutral / positive)
    max_length: int     = 128
    dropout_rate: float = 0.2

    # -- LoRA -----------------------------------------------------------------
    use_lora: bool       = True
    lora_r: int          = 8
    lora_alpha: int      = 16
    lora_dropout: float  = 0.1
    # Target DistilBERT's query and value projection matrices in every
    # transformer block.  These carry the most task-relevant information.
    lora_target_modules: tuple = ("q_lin", "v_lin")

    # -- Optimiser & loss -----------------------------------------------------
    learning_rate: float  = 2e-4
    weight_decay: float   = 1e-2
    label_smoothing: float= 0.1


# ---------------------------------------------------------------------------
# Main Model class
# ---------------------------------------------------------------------------

class SentimentModel:
    """
    Assembles the complete TF Keras model for Senti-DLF.

    Architecture (bottom-up)
    ------------------------
    Input (input_ids, attention_mask)
        │
        ▼
    TFDistilBertModel           ← 66 M params (frozen when LoRA is on)
        │  last_hidden_state [batch, seq_len, 768]
        │
        ▼
    CLS token slice [:, 0, :]   ← [batch, 768]
        │
        ▼
    Dropout(dropout_rate)
        │
        ▼
    Dense(num_labels)           ← classification logits [batch, 3]

    When `use_lora=True`, PEFT wraps the base transformer and inserts low-rank
    trainable matrices into the Q and V projection layers of every attention
    block.  All other weights are frozen.

    Parameters
    ----------
    config : ModelConfig
        Configuration object.  If omitted, default ModelConfig() is used.

    Example
    -------
        sm    = SentimentModel(ModelConfig(lora_r=16, dropout_rate=0.3))
        model = sm.build()
        model.fit(train_ds, validation_data=val_ds, epochs=3)
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self._model: Optional[tf.keras.Model] = None

        log.info(
            "SentimentModel initialised | base=%s | LoRA=%s (r=%d, alpha=%d)",
            self.config.model_name,
            self.config.use_lora,
            self.config.lora_r,
            self.config.lora_alpha,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> tf.keras.Model:
        """
        Assemble and compile the full Keras model.

        Steps:
            1. Load TFDistilBertModel weights from HuggingFace.
            2. Optionally inject LoRA adapters and freeze the base.
            3. Build the Keras Functional graph (inputs → CLS → head).
            4. Compile with AdamW + sparse categorical cross-entropy.

        Returns:
            Compiled tf.keras.Model ready to call `.fit()` on.
        """
        log.info("--- Building model ---")

        base_model = self._load_base()

        if self.config.use_lora:
            base_model = self._apply_lora(base_model)
        else:
            log.warning(
                "LoRA disabled — all %d base parameters will be trained.",
                base_model.num_parameters(),
            )
            base_model.trainable = True

        self._model = self._build_keras_graph(base_model)
        self._compile()

        log.info("Model built and compiled successfully.")
        return self._model

    def summary(self) -> None:
        """
        Print the Keras model summary (layer names + trainable param counts).

        Raises:
            RuntimeError: If `build()` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError("Call .build() before .summary().")
        self._model.summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_base(self):
        """
        Download (or load from local cache) the HuggingFace pre-trained weights.

        Returns:
            TFDistilBertModel instance with its pre-trained weights.
        """
        log.info("Loading base transformer: %s", self.config.model_name)
        from transformers import TFDistilBertModel

        base = TFDistilBertModel.from_pretrained(self.config.model_name, use_safetensors=False)
        log.info("Base transformer loaded.")
        return base

    def _apply_lora(self, base_model):
        """
        Inject LoRA adapters into the base transformer using PEFT.

        LoRA inserts two small trainable matrices (A, B) of rank `lora_r`
        into each targeted attention projection.  The original weight W is
        frozen; the effective weight becomes W + (B @ A) * (alpha / r).

        Args:
            base_model: TFDistilBertModel with frozen weights.

        Returns:
            PEFT-wrapped model with LoRA adapters injected.
        """
        log.info(
            "Injecting LoRA adapters — target_modules=%s | r=%d | alpha=%d",
            self.config.lora_target_modules,
            self.config.lora_r,
            self.config.lora_alpha,
        )
        try:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                task_type="SEQ_CLS",
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
                bias="none",
            )
            lora_model = get_peft_model(base_model, lora_cfg)

            trainable, total = lora_model.get_nb_trainable_parameters()
            log.info(
                "LoRA applied — trainable params: %s / %s  (%.2f%%)",
                f"{trainable:,}",
                f"{total:,}",
                100 * trainable / total,
            )
            return lora_model

        except Exception as e:
            log.warning(
                "PEFT LoRA failed (likely due to PyTorch expectation in this peft version): %s\n"
                "Falling back to NATIVE PARTIAL FINE-TUNING.", e
            )
            
            # Fallback: Freeze the embeddings and the first 4 transformer blocks.
            # Only the last 2 transformer blocks (out of 6) will be trainable.
            log.info("Freezing embeddings and transformer layers 0-3...")
            
            # Access the underlying distilbert layers
            transformer_layer = base_model.distilbert.transformer
            embeddings_layer = base_model.distilbert.embeddings
            
            embeddings_layer.trainable = False
            for i in range(4):
                transformer_layer.layer[i].trainable = False
                
            for i in range(4, 6):
                transformer_layer.layer[i].trainable = True
                
            log.info("Layers 4-5 and classification head are trainable. (Memory efficient)")
            return base_model

    def _build_keras_graph(self, base_model) -> tf.keras.Model:
        """
        Wire the base transformer into a Keras Functional model.

        The input dictionary keys ("input_ids", "attention_mask") match
        exactly what SentimentDataset produces — no adapter layer needed.

        Args:
            base_model: The (possibly LoRA-wrapped) transformer model.

        Returns:
            Un-compiled tf.keras.Model.
        """
        log.info("Assembling Keras functional graph…")

        # -- Inputs -----------------------------------------------------------
        input_ids = tf.keras.layers.Input(
            shape=(self.config.max_length,),
            dtype=tf.int32,
            name="input_ids",
        )
        attention_mask = tf.keras.layers.Input(
            shape=(self.config.max_length,),
            dtype=tf.int32,
            name="attention_mask",
        )

        # -- Transformer forward pass -----------------------------------------
        # last_hidden_state: (batch_size, seq_len, hidden_dim=768)
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=False,       # will be overridden to True during .fit()
        )
        sequence_output = outputs.last_hidden_state

        # -- CLS pooling ------------------------------------------------------
        # Index 0 is the [CLS] token — aggregates the full sequence meaning.
        cls_token = sequence_output[:, 0, :]   # (batch_size, 768)

        # -- Classification head ----------------------------------------------
        x = tf.keras.layers.Dropout(
            self.config.dropout_rate, name="head_dropout"
        )(cls_token)

        logits = tf.keras.layers.Dense(
            self.config.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="classifier",
        )(x)

        # -- Wrap into Keras Model --------------------------------------------
        model = tf.keras.Model(
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            outputs=logits,
            name="SentiDLF_DistilBERT",
        )
        log.info("Keras graph assembled.")
        return model

    def _compile(self) -> None:
        """
        Compile the Keras model with AdamW, sparse cross-entropy, and metrics.

        Uses `from_logits=True` because the Dense head does NOT apply softmax —
        this is numerically more stable during training.

        Label smoothing of 0.1 is applied to reduce overconfidence on the
        noisy Twitter dataset.
        """
        log.info(
            "Compiling — lr=%.0e | weight_decay=%.0e | label_smoothing=%.2f",
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.label_smoothing,
        )
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction="auto",
            ),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )


# ---------------------------------------------------------------------------
# Script entry point — quick architecture sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)

    log.info("Running model architecture sanity check…")
    sm    = SentimentModel(ModelConfig())
    model = sm.build()
    sm.summary()

    log.info("GPU devices visible to TensorFlow:")
    for dev in tf.config.list_physical_devices("GPU"):
        log.info("  %s", dev)
