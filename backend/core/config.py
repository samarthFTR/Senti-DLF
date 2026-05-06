from pydantic_settings import BaseSettings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """
    FastAPI backend configuration.
    Reads from environment variables or defaults to these values.
    """
    api_title: str = "Senti-DLF Inference API"
    api_version: str = "1.0.0"
    
    # Model configuration
    model_dir: str = str(_PROJECT_ROOT / "model" / "saved_models" / "distilbert_v1")
    tokenizer_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 128
    
    # Confidence threshold for returning a valid prediction
    confidence_threshold: float = 0.50

    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()
