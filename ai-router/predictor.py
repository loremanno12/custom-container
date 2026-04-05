"""Predizione del modello AI per un dato prompt."""
import logging
from typing import Any, Dict

from cache import ModelCache
from config import Config
from ollama_service import validate_prompt
from metrics import Timer

logger = logging.getLogger(__name__)

# Iniettato da api.py al bootstrap dell'app FastAPI
metrics_collector = None


def predict_model(
    prompt: str, config: Config, model_cache: ModelCache
) -> Dict[str, Any]:
    """Predice quale modello utilizzare per un dato prompt, con cache e metriche."""
    try:
        is_valid, error_msg = validate_prompt(prompt)
        if not is_valid:
            logger.warning("Prompt non valido: %s", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "predicted_model": None,
                "confidence": None,
            }

        # Cache delle predizioni
        cached_result = model_cache.prediction_cache.get(prompt)
        if cached_result:
            logger.info("Risultato da cache per il prompt: %s...", prompt[:50])
            if metrics_collector:
                metrics_collector.record_prediction(
                    0.0,
                    is_cache_hit=True,
                    confidence=cached_result.get("confidence", 0.0),
                )
            return cached_result

        embedding_model = model_cache.get_embedding_model(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )
        classifier = model_cache.get_classifier(config.CLASSIFIER_PATH)
        label_encoder = model_cache.get_label_encoder(config.ENCODER_PATH)

        if classifier is None or label_encoder is None:
            error_msg = "Modelli non trovati. Addestrare prima il modello."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "predicted_model": None,
                "confidence": None,
            }

        with Timer("Predizione modello") as timer:
            logger.info("Predizione del modello per il prompt: %s...", prompt[:50])
            prompt_embedding = embedding_model.encode(
                [prompt],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            )

            prediction = classifier.predict(prompt_embedding)
            probabilities = classifier.predict_proba(prompt_embedding)[0]

        predicted_model = label_encoder.inverse_transform(prediction)[0]
        confidence = float(probabilities.max())

        # Mappa tutte le probabilità
        all_probabilities = {
            str(cls): float(prob)
            for cls, prob in zip(label_encoder.classes_, probabilities)
        }

        result = {
            "success": True,
            "error": None,
            "predicted_model": predicted_model,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
        }

        if metrics_collector:
            metrics_collector.record_prediction(
                timer.elapsed,
                is_cache_hit=False,
                confidence=confidence,
                threshold=config.CONFIDENCE_THRESHOLD,
            )

        model_cache.prediction_cache.set(prompt, result)
        return result

    except Exception as e:
        error_msg = f"Errore durante la predizione: {str(e)}"
        logger.exception(error_msg)
        if metrics_collector:
            metrics_collector.record_prediction(0.0, had_error=True)
        return {
            "success": False,
            "error": error_msg,
            "predicted_model": None,
            "confidence": None,
        }


def format_prediction_output(result: Dict[str, Any], config: Config) -> str:
    """Formatta il risultato della predizione per la visualizzazione."""
    if not result["success"]:
        return f"❌ **Errore:** {result['error']}"

    output = f"### 🎯 Modello Consigliato\n\n"
    output += f"**{result['predicted_model']}**\n\n"
    output += f"📊 Confidenza: **{result['confidence']:.1%}**\n\n"

    if result.get("all_probabilities"):
        output += "---\n\n"
        output += "### 📈 Top 3 Modelli\n\n"

        sorted_probs = sorted(
            result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
        )[: config.TOP_N_PREDICTIONS]

        for i, (model, prob) in enumerate(sorted_probs, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            output += f"{medal} **{model}**\n"
            output += f"`{bar}` {prob:.1%}\n\n"

    if result["confidence"] < config.CONFIDENCE_THRESHOLD:
        output += "\n---\n\n"
        output += f"⚠️ **Attenzione:** Confidenza sotto la soglia del {config.CONFIDENCE_THRESHOLD:.0%}. "
        output += "Considera di migliorare il prompt per risultati più accurati.\n"

    return output
