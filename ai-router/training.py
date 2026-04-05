"""Addestramento del modello AI Router."""
import json
import logging
import pickle
from pathlib import Path
from typing import Tuple

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from cache import ModelCache
from config import Config

logger = logging.getLogger(__name__)


def load_training_data(file_path: Path) -> Tuple[list, list]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Il file di training deve contenere una lista di oggetti")
    prompts = []
    models = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Ogni elemento del training deve essere un oggetto JSON")
        if "modello" not in item or "prompts" not in item:
            raise KeyError("Ogni elemento deve contenere le chiavi 'modello' e 'prompts'")
        if not isinstance(item["prompts"], list):
            raise ValueError("Il campo 'prompts' deve essere una lista")
        for prompt in item["prompts"]:
            prompts.append(prompt)
            models.append(item["modello"])
    return prompts, models


def validate_training_data(prompts: list, models: list) -> bool:
    if not prompts or not models:
        logger.error("I dati di addestramento sono vuoti")
        return False
    if len(prompts) != len(models):
        logger.error("Il numero di prompt e modelli non corrisponde")
        return False
    if any(not p or not isinstance(p, str) for p in prompts):
        logger.error("I dati di addestramento contengono prompt non validi")
        return False
    if any(not m or not isinstance(m, str) for m in models):
        logger.error("I dati di addestramento contengono modelli non validi")
        return False
    if len(set(models)) < 2:
        logger.error("Servono almeno 2 modelli distinti per addestrare il classificatore")
        return False
    return True


def should_retrain(config: Config) -> bool:
    if not config.CLASSIFIER_PATH.exists() or not config.ENCODER_PATH.exists():
        logger.info("File del modello non trovati, addestramento necessario")
        return True
    if config.RETRAIN_ON_DATA_CHANGE and config.TRAINING_DATA_PATH.exists():
        training_data_mtime = config.TRAINING_DATA_PATH.stat().st_mtime
        classifier_mtime = config.CLASSIFIER_PATH.stat().st_mtime
        if training_data_mtime > classifier_mtime:
            logger.info("Dati piu recenti del modello, riaddestramento necessario")
            return True
    logger.info("Utilizzo del modello addestrato esistente")
    return False


def train_model(config: Config, model_cache: ModelCache) -> Tuple[bool, str]:
    try:
        logger.info("Caricamento dati da: %s", config.TRAINING_DATA_PATH)
        if not config.TRAINING_DATA_PATH.exists():
            return False, f"File non trovato: {config.TRAINING_DATA_PATH}"
        prompts, models = load_training_data(config.TRAINING_DATA_PATH)
        if not validate_training_data(prompts, models):
            return False, "Formato dati non valido"
        logger.info("Caricati %s esempi", len(prompts))
        embedding_model = model_cache.get_embedding_model(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )
        X = embedding_model.encode(
            prompts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(models)
        classifier = MLPClassifier(
            hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
            max_iter=config.MLP_MAX_ITER,
            random_state=config.MLP_RANDOM_STATE,
        )
        classifier.fit(X, y)
        with open(config.CLASSIFIER_PATH, "wb") as f:
            pickle.dump(classifier, f)
        with open(config.ENCODER_PATH, "wb") as f:
            pickle.dump(label_encoder, f)
        model_cache.set_classifier(classifier)
        model_cache.set_label_encoder(label_encoder)
        return True, f"Modello addestrato con {len(prompts)} esempi"
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.exception("Errore addestramento")
        return False, str(e)
    except Exception as e:
        logger.exception("Errore addestramento")
        return False, str(e)
