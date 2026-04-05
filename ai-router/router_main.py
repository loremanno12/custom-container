"""
Punto di ingresso per il Sistema Router AI.
Unisce le migliori caratteristiche dei progetti originali.
"""
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

from cache import ModelCache
from config import Config
from ollama_service import check_ollama_health
from training import should_retrain, train_model
from ui import create_gradio_interface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Avvio del Sistema Router AI Unificato")
    logger.info("=" * 60)

    config = Config()
    model_cache = ModelCache()

    if should_retrain(config):
        logger.info("Addestramento del modello in corso...")
        success, message = train_model(config, model_cache)
        if not success:
            logger.error(f"Addestramento fallito: {message}")
            logger.error(
                "Verifica che training_data.json esista con la struttura corretta"
            )
            return
        logger.info(message)
    else:
        model_cache.get_classifier(config.CLASSIFIER_PATH)
        model_cache.get_label_encoder(config.ENCODER_PATH)

    if check_ollama_health(config):
        logger.info("Ollama disponibile")
    else:
        logger.warning("Ollama non disponibile - miglioramento prompt disabilitato")

    logger.info("Avvio dell'interfaccia Gradio")
    logger.info(
        f"Accedi a http://{config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}"
    )
    logger.info("=" * 60)

    interface = create_gradio_interface(config, model_cache)
    interface.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
    )


if __name__ == "__main__":
    main()
