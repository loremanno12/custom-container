"""Configurazione centrale per il sistema AI Router (env + default)."""
import os
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_tuple(value: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if not value:
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    parsed: list[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError:
            return default
    return tuple(parsed) if parsed else default


@dataclass
class Config:
    """Configurazione centrale per il sistema AI Router."""

    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))
    CLASSIFIER_PATH: Path = None
    ENCODER_PATH: Path = None

    TRAINING_DATA_PATH: Path = Path(os.getenv("TRAINING_DATA_PATH", "training_data.json"))

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE: int = _parse_int(os.getenv("EMBEDDING_BATCH_SIZE"), 16)
    NORMALIZE_EMBEDDINGS: bool = _parse_bool(
        os.getenv("NORMALIZE_EMBEDDINGS"), True
    )
    MLP_HIDDEN_LAYERS: Tuple[int, ...] = _parse_int_tuple(
        os.getenv("MLP_HIDDEN_LAYERS"), (100, 50)
    )
    MLP_MAX_ITER: int = _parse_int(os.getenv("MLP_MAX_ITER"), 500)
    MLP_RANDOM_STATE: int = _parse_int(os.getenv("MLP_RANDOM_STATE"), 42)

    CONFIDENCE_THRESHOLD: float = _parse_float(
        os.getenv("CONFIDENCE_THRESHOLD"), 0.5
    )
    TOP_N_PREDICTIONS: int = _parse_int(os.getenv("TOP_N_PREDICTIONS"), 3)

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:270m")
    OLLAMA_TIMEOUT: int = _parse_int(os.getenv("OLLAMA_TIMEOUT"), 60)
    OLLAMA_TEMPERATURE: float = _parse_float(
        os.getenv("OLLAMA_TEMPERATURE"), 0.3
    )
    OLLAMA_TOP_P: float = _parse_float(os.getenv("OLLAMA_TOP_P"), 0.9)
    OLLAMA_NUM_PREDICT: int = _parse_int(os.getenv("OLLAMA_NUM_PREDICT"), 450)

    GRADIO_SERVER_NAME: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT: int = _parse_int(os.getenv("GRADIO_SERVER_PORT"), 7860)
    GRADIO_SHARE: bool = _parse_bool(os.getenv("GRADIO_SHARE"), False)

    CPU_THREADS: int = _parse_int(os.getenv("CPU_THREADS"), 2)
    RETRAIN_ON_DATA_CHANGE: bool = _parse_bool(
        os.getenv("RETRAIN_ON_DATA_CHANGE"), False
    )

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        """Crea le directory necessarie e inizializza i percorsi dei modelli."""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if self.CLASSIFIER_PATH is None:
            self.CLASSIFIER_PATH = self.MODEL_DIR / "mlp_classifier.pkl"
        if self.ENCODER_PATH is None:
            self.ENCODER_PATH = self.MODEL_DIR / "label_encoder.pkl"
        # Validazione dei parametri
        self.CONFIDENCE_THRESHOLD = min(max(self.CONFIDENCE_THRESHOLD, 0.0), 1.0)
        self.TOP_N_PREDICTIONS = max(1, self.TOP_N_PREDICTIONS)
        self.OLLAMA_TIMEOUT = max(1, self.OLLAMA_TIMEOUT)
        self.OLLAMA_TEMPERATURE = min(max(self.OLLAMA_TEMPERATURE, 0.0), 1.0)
        self.OLLAMA_TOP_P = min(max(self.OLLAMA_TOP_P, 0.0), 1.0)
        self.OLLAMA_NUM_PREDICT = max(64, self.OLLAMA_NUM_PREDICT)
        self.CPU_THREADS = max(1, self.CPU_THREADS)
        self.EMBEDDING_BATCH_SIZE = max(1, self.EMBEDDING_BATCH_SIZE)
