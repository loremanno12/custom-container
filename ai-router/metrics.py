"""Raccolta metriche e timer per il sistema AI Router."""
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metriche aggregate per le predizioni."""

    total_predictions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_inference_time: float = 0.0
    total_inference_time: float = 0.0
    errors: int = 0
    low_confidence_predictions: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Percentuale di cache hit."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converte le metriche in dizionario serializzabile."""
        return {
            "total_predictions": self.total_predictions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{self.cache_hit_rate:.1f}%",
            "avg_inference_time_ms": f"{self.avg_inference_time * 1000:.2f}",
            "errors": self.errors,
            "low_confidence_predictions": self.low_confidence_predictions,
            "timestamp": datetime.now().isoformat(),
        }


class MetricsCollector:
    """Raccoglie e aggiorna metriche del sistema."""

    def __init__(self) -> None:
        self.predictions = PredictionMetrics()

    def record_prediction(
        self,
        inference_time: float,
        is_cache_hit: bool = False,
        had_error: bool = False,
        confidence: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        """Registra una predizione."""
        self.predictions.total_predictions += 1

        if is_cache_hit:
            self.predictions.cache_hits += 1
        else:
            self.predictions.cache_misses += 1
            self.predictions.total_inference_time += inference_time

        if had_error:
            self.predictions.errors += 1

        if 0 < confidence < threshold:
            self.predictions.low_confidence_predictions += 1

        if self.predictions.cache_misses > 0:
            self.predictions.avg_inference_time = (
                self.predictions.total_inference_time / self.predictions.cache_misses
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Ritorna tutte le metriche come dizionario."""
        return self.predictions.to_dict()

    def log_metrics(self) -> None:
        """Scrive le metriche nei log."""
        metrics = self.get_metrics()
        logger.info("Metriche di sistema: %s", metrics)

    def reset(self) -> None:
        """Resetta tutte le metriche."""
        self.predictions = PredictionMetrics()


class Timer:
    """Context manager per misurare il tempo di esecuzione."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.start_time is None:
            return
        self.elapsed = time.time() - self.start_time
        if self.name:
            logger.debug("%s completato in %.2fms", self.name, self.elapsed * 1000)

