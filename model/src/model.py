"""
model.py
--------
Model architecture for the Senti-DLF sentiment classifier.

Wraps HuggingFace's TFBertModel inside a Keras Functional model,
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
# Native TF Keras LoRA Implementation
# ---------------------------------------------------------------------------

class TF_LoRADense(tf.keras.layers.Layer):
    def __init__(self, original_dense, r=8, alpha=16, dropout=0.1, **kwargs):
        super().__init__(name=original_dense.name + "_lora", **kwargs)
        # Rename the inner layer to avoid H5 group name collision with the
        # original BERT layer registry entry (e.g. "query" -> "query_base")
        original_dense._name = original_dense.name + "_base"
        self.original_dense = original_dense
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        if not self.original_dense.built:
            self.original_dense.build(input_shape)
        self.original_dense.trainable = False
        self.lora_A = self.add_weight(
            name='lora_A', shape=(input_shape[-1], self.r),
            initializer=tf.keras.initializers.RandomNormal(stddev=1.0 / self.r), trainable=True
        )
        self.lora_B = self.add_weight(
            name='lora_B', shape=(self.r, self.original_dense.units),
            initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        orig_out = self.original_dense(inputs, *args, **kwargs)
        lora_in = self.lora_dropout(inputs)
        lora_out = tf.matmul(lora_in, self.lora_A)
        lora_out = tf.matmul(lora_out, self.lora_B) * self.scaling
        return orig_out + lora_out

    def get_config(self):
        config = super().get_config()
        config.update({"r": self.r, "alpha": self.alpha,
                        "dropout": self.lora_dropout.rate})
        return config

class LoRAModel(tf.keras.Model):
    def __init__(self, inputs, outputs, lora_layers, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self._lora_layers = lora_layers

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
    model_name: str     = "bert-base-uncased"
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
    TFBertModel                 ← ~110 M params (frozen when LoRA is on)
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
            1. Load TFBertModel weights from HuggingFace.
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
            # Partial fine-tuning: freeze embeddings + bottom 9 transformer
            # layers, train only the top 3 layers + classification head.
            # This keeps VRAM usage manageable on 4GB GPUs while still
            # adapting the model effectively to the task.
            log.info("LoRA disabled — applying partial fine-tuning (top 3 layers).")
            base_model.trainable = False   # freeze everything first

            # Detect encoder attribute (bert vs deberta)
            if hasattr(base_model, 'bert'):
                encoder_layers = base_model.bert.encoder.layer
            elif hasattr(base_model, 'deberta'):
                encoder_layers = base_model.deberta.encoder.layer
            else:
                encoder_layers = []
                log.warning("Unknown model structure — base remains fully frozen.")

            total = len(encoder_layers)
            freeze_up_to = max(0, total - 3)   # unfreeze last 3 layers
            for i, layer in enumerate(encoder_layers):
                layer.trainable = i >= freeze_up_to

            trainable_count = sum(
                1 for l in encoder_layers if l.trainable
            )
            log.info(
                "Partial fine-tuning: %d/%d transformer layers unfrozen.",
                trainable_count, total
            )

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
            TFBertModel instance with its pre-trained weights.
        """
        log.info("Loading base transformer: %s", self.config.model_name)
        from transformers import TFAutoModel

        base = TFAutoModel.from_pretrained(self.config.model_name, use_safetensors=False)
        log.info("Base transformer loaded.")
        return base

    def _apply_lora(self, base_model):
        """
        Inject Custom TF LoRA adapters into the base transformer.
        """
        log.info(
            "Injecting TF-Native LoRA adapters — target_modules=%s | r=%d | alpha=%d",
            self.config.lora_target_modules,
            self.config.lora_r,
            self.config.lora_alpha,
        )
        base_model.trainable = False
        lora_layers = []

        if hasattr(base_model, 'bert'):
            encoder = base_model.bert.encoder
        elif hasattr(base_model, 'deberta'):
            encoder = base_model.deberta.encoder
        else:
            log.warning("Unrecognized base model. Falling back to freezing.")
            return base_model

        for layer in encoder.layer:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'self_attention'):
                self_attn = layer.attention.self_attention
                
                if hasattr(self_attn, 'query') and hasattr(self_attn, 'value'):
                    self_attn.query = TF_LoRADense(
                        self_attn.query, r=self.config.lora_r, 
                        alpha=self.config.lora_alpha, dropout=self.config.lora_dropout
                    )
                    self_attn.value = TF_LoRADense(
                        self_attn.value, r=self.config.lora_r, 
                        alpha=self.config.lora_alpha, dropout=self.config.lora_dropout
                    )
                    lora_layers.extend([self_attn.query, self_attn.value])
                    
                elif hasattr(self_attn, 'query_proj') and hasattr(self_attn, 'value_proj'):
                    self_attn.query_proj = TF_LoRADense(
                        self_attn.query_proj, r=self.config.lora_r, 
                        alpha=self.config.lora_alpha, dropout=self.config.lora_dropout
                    )
                    self_attn.value_proj = TF_LoRADense(
                        self_attn.value_proj, r=self.config.lora_r, 
                        alpha=self.config.lora_alpha, dropout=self.config.lora_dropout
                    )
                    lora_layers.extend([self_attn.query_proj, self_attn.value_proj])
        
        base_model._lora_layers = lora_layers
        log.info("TF-Native LoRA applied successfully.")
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
        token_type_ids = tf.keras.layers.Input(
            shape=(self.config.max_length,),
            dtype=tf.int32,
            name="token_type_ids",
        )

        # -- Transformer forward pass -----------------------------------------
        # last_hidden_state: (batch_size, seq_len, hidden_dim=768)
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
        lora_layers = getattr(base_model, '_lora_layers', [])
        if lora_layers:
            model = LoRAModel(
                inputs={
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                },
                outputs=logits,
                lora_layers=lora_layers,
                name="SentiDLF_BERT_LoRA",
            )
        else:
            model = tf.keras.Model(
                inputs={
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                },
                outputs=logits,
                name="SentiDLF_BERT",
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
            loss=tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                label_smoothing=self.config.label_smoothing,
            ),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
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
