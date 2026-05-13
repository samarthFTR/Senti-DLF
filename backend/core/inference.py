import logging
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from backend.core.config import settings

log = logging.getLogger("uvicorn.error")

# Ensure the backend uses the exact same label mapping as the training code
ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

class SentimentInferenceEngine:
    """
    Singleton class that loads the saved Keras model into memory once
    at startup and handles efficient tokenization and inference.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        log.info("Initialising Inference Engine...")
        self.models = {}
        self.tokenizers = {}
        
        # Load the model specified in environment/settings
        self._load_model(settings.model_type)
        self._initialized = True
        
    def _load_model(self, model_type: str):
        if model_type in self.models:
            return
            
        tokenizer_name = "microsoft/deberta-v3-base" if model_type == "deberta" else "bert-base-uncased"
        use_fast = model_type != "deberta"
        
        log.info(f"Loading {model_type} tokenizer...")
        self.tokenizers[model_type] = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=use_fast
        )
        
        log.info(f"Loading {model_type} architecture and weights...")
        try:
            from model.src.model import SentimentModel, ModelConfig
            from backend.core.config import _PROJECT_ROOT

            # DeBERTa is too large to share GPU VRAM with BERT on a 4GB GPU.
            # Force it to CPU — system RAM can hold 735MB comfortably.
            device = "/CPU:0" if model_type == "deberta" else "/GPU:0"
            log.info(f"Pinning {model_type} to {device}")

            with tf.device(device):
                if model_type == "bert":
                    # Use the original partial fine-tuning architecture (no LoRA)
                    # to match the bert_v1 SavedModel which had PEFT fallback.
                    # This restores the good-performing original model.
                    m_cfg = ModelConfig(model_name=tokenizer_name, use_lora=False)
                    model_wrapper = SentimentModel(m_cfg)
                    model = model_wrapper.build()

                    saved_model_vars = _PROJECT_ROOT / "model" / "saved_models" / "bert_v1" / "variables" / "variables"
                    ckpt_path       = _PROJECT_ROOT / "model" / "saved_models" / "checkpoints" / "bert_best_weights.ckpt"

                    if saved_model_vars.with_suffix(".index").exists():
                        log.info("Loading BERT weights from bert_v1 SavedModel variables (original partial fine-tuning).")
                        model.load_weights(str(saved_model_vars))
                    elif ckpt_path.with_suffix(".ckpt.index").exists():
                        log.info("Loading BERT weights from checkpoint.")
                        model.load_weights(str(ckpt_path))
                    else:
                        log.warning("No BERT weights found.")
                else:
                    # DeBERTa: partial fine-tuning architecture (top 3 layers),
                    # matching exactly what trainer.py produces with use_lora=False.
                    m_cfg = ModelConfig(model_name=tokenizer_name, use_lora=False)
                    model_wrapper = SentimentModel(m_cfg)
                    model = model_wrapper.build()

                    specific_ckpt = _PROJECT_ROOT / "model" / "saved_models" / "checkpoints" / f"{model_type}_best_weights.ckpt"
                    specific_h5   = _PROJECT_ROOT / "model" / "saved_models" / "checkpoints" / f"{model_type}_best_weights.h5"

                    weights_path = None
                    # TF checkpoints don't create a bare .ckpt file — check for
                    # the .ckpt.index sidecar to confirm the checkpoint exists.
                    ckpt_index = _PROJECT_ROOT / "model" / "saved_models" / "checkpoints" / f"{model_type}_best_weights.ckpt.index"
                    h5_path    = _PROJECT_ROOT / "model" / "saved_models" / "checkpoints" / f"{model_type}_best_weights.h5"

                    if ckpt_index.exists():
                        weights_path = ckpt_index.parent / ckpt_index.stem  # strips .index → .ckpt
                    elif h5_path.exists():
                        weights_path = h5_path

                    if weights_path:
                        log.info(f"Loading DeBERTa fine-tuned weights from {weights_path.name}.")
                        model.load_weights(str(weights_path))
                    else:
                        log.warning(
                            "No fine-tuned DeBERTa weights found — using pretrained base model. "
                            "Run: python -m model.src.trainer --model deberta"
                        )

            self.models[model_type] = model
            # Track which device this model is pinned to for inference
            self.model_devices = getattr(self, "model_devices", {})
            self.model_devices[model_type] = device
            # Track whether this model has fine-tuned weights loaded
            self.model_ready = getattr(self, "model_ready", {})
            self.model_ready[model_type] = bool(weights_path) if model_type != "bert" else True
            log.info(f"{model_type} loaded successfully on {device}.")
        except Exception as e:
            log.error(f"Failed to load {model_type}: {e}")
            raise e

    def predict(self, texts: list[str], model_type: str = None) -> list[dict]:
        """
        Run inference on a batch of texts using the specified model.
        """
        model_type = model_type or settings.model_type
        self._load_model(model_type)

        # Block inference if the model was loaded without fine-tuned weights
        model_ready = getattr(self, "model_ready", {})
        if not model_ready.get(model_type, True):
            raise RuntimeError(
                f"'{model_type}' model has no fine-tuned weights yet. "
                f"Train it first: python -m model.src.trainer --model {model_type}"
            )

        model = self.models[model_type]
        tokenizer = self.tokenizers[model_type]
        
        # 1. Tokenize (exactly as dataset.py did during training)
        encoded = tokenizer(
            texts,
            max_length=settings.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="np"
        )

        input_dict = {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32),
            "token_type_ids": encoded["token_type_ids"].astype(np.int32)
        }

        # 2. Forward Pass — run on whichever device the model was pinned to
        device = getattr(self, "model_devices", {}).get(model_type, "/GPU:0")
        with tf.device(device):
            logits = model.predict(input_dict, batch_size=4, verbose=0)
        # Use softmax for a proper single-label probability distribution
        softmax_probs = tf.nn.softmax(logits).numpy()
        # Also keep sigmoid scores for raw confidence display
        sigmoid_probs = tf.nn.sigmoid(logits).numpy()

        # 3. Decode results
        results = []
        for sm_probs, sig_probs in zip(softmax_probs, sigmoid_probs):
            prob_dict = {
                "negative": float(sm_probs[0]),
                "neutral":  float(sm_probs[1]),
                "positive": float(sm_probs[2]),
                "mixed":    0.0
            }
            
            # Mixed: only when softmax gives meaningfully high scores to BOTH
            # positive and negative (margin < 0.15 and both > 0.35)
            pos, neg = float(sm_probs[2]), float(sm_probs[0])
            if pos > 0.35 and neg > 0.35 and abs(pos - neg) < 0.15:
                label = "mixed"
                confidence = (pos + neg) / 2.0
                prob_dict["mixed"] = confidence
            else:
                # Standard argmax over softmax distribution
                pred_id = int(np.argmax(sm_probs))
                label = ID_TO_LABEL[pred_id]
                confidence = float(sm_probs[pred_id])
            
            results.append({
                "label": label,
                "confidence": confidence,
                "probabilities": prob_dict
            })

        return results

# Expose a global instance getter
def get_engine() -> SentimentInferenceEngine:
    return SentimentInferenceEngine()
