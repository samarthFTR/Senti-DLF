import logging
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast
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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(settings.tokenizer_name)
        
        log.info("Loading model architecture and weights...")
        try:
            from model.src.model import SentimentModel, ModelConfig
            import os
            from backend.core.config import _PROJECT_ROOT
            
            # Reconstruct the exact architecture used during training
            model_wrapper = SentimentModel(ModelConfig())
            self.model = model_wrapper.build()
            
            # Load the optimal weights from the checkpoint
            weights_path = str(_PROJECT_ROOT / "model" / "saved_models" / "checkpoints" / "best_weights.h5")
            self.model.load_weights(weights_path)
            
            log.info("Model architecture and weights loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise e
            
        self._initialized = True

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Run inference on a batch of texts.
        """
        # 1. Tokenize (exactly as dataset.py did during training)
        encoded = self.tokenizer(
            texts,
            max_length=settings.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np"
        )

        input_dict = {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32)
        }

        # 2. Forward Pass
        # The model outputs logits, so we apply softmax to get probabilities
        logits = self.model.predict(input_dict, verbose=0)
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()

        # 3. Decode results
        results = []
        for probs in probabilities:
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            
            prob_dict = {
                ID_TO_LABEL[i]: float(probs[i]) for i in range(len(ID_TO_LABEL))
            }
            
            results.append({
                "label": ID_TO_LABEL[pred_id],
                "confidence": confidence,
                "probabilities": prob_dict
            })

        return results

# Expose a global instance getter
def get_engine() -> SentimentInferenceEngine:
    return SentimentInferenceEngine()
