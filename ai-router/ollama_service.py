"""Servizio Ollama per miglioramento prompt e validazione."""
import logging
import re
import time
from typing import Any, Dict, Optional

import requests

# Assumiamo che Config sia importato correttamente dal tuo progetto
from config import Config

logger = logging.getLogger(__name__)

# Pattern per rimuovere prefissi chiacchieroni che i modelli LLM spesso aggiungono
PREFIX_PATTERNS = (
    r"^\s*ecco il prompt migliorato:\s*",
    r"^\s*prompt migliorato:\s*",
    r"^\s*ecco:\s*",
    r"^\s*here is the improved prompt:\s*",
    r"^\s*improved prompt:\s*",
    r"^\s*sicuramente! ecco una versione migliorata:\s*",
)

def validate_prompt(prompt: str) -> tuple[bool, str]:
    """Valida l'input dell'utente prima dell'invio."""
    if not prompt:
        return False, "Il prompt non può essere vuoto"
    if not isinstance(prompt, str):
        return False, "Il prompt deve essere una stringa"
    
    prompt = prompt.strip()
    if not prompt:
        return False, "Il prompt non può contenere solo spazi"
    if len(prompt) > 5000:
        return False, "Il prompt è troppo lungo (max 5000 caratteri)"
    
    return True, ""

def _detect_prompt_profile(prompt: str) -> str:
    """Rileva il tipo di prompt per personalizzare le istruzioni di sistema."""
    prompt_lower = prompt.lower()
    if any(token in prompt_lower for token in ("python", "javascript", "sql", "bug", "debug", "api", "codice", "script")):
        return "technical"
    if any(token in prompt_lower for token in ("spiega", "explain", "riassumi", "summary", "analizza", "analyze")):
        return "explanatory"
    if any(token in prompt_lower for token in ("story", "creative", "campagna", "marketing", "copy", "brand", "post social")):
        return "creative"
    return "general"

def _build_system_instruction(prompt: str, target_model: Optional[str]) -> str:
    """Costruisce il system prompt per il 'Prompt Engineer'."""
    profile = _detect_prompt_profile(prompt)
    profile_rules = {
        "technical": "Privilegia vincoli chiari, input/output attesi, linguaggio o stack e criteri di qualità verificabili.",
        "explanatory": "Privilegia pubblico target, livello di profondità, struttura della risposta ed esempi concreti.",
        "creative": "Privilegia tono, audience, formato, obiettivo e stile desiderato.",
        "general": "Privilegia chiarezza, contesto, output atteso e criteri di completezza.",
    }
    
    target_hint = (
        f"Il router suggerisce come modello di destinazione '{target_model}'. "
        "Ottimizza il prompt per questo tipo di architettura senza citarla nel testo finale."
        if target_model else "Rendi il prompt robusto e universale."
    )

    return (
        "Sei un prompt engineer senior. Devi migliorare il prompt dell'utente mantenendo invariato l'obiettivo.\n"
        "Rispondi nella stessa lingua del prompt originale.\n"
        "Restituisci SOLO il prompt finale. NO spiegazioni, NO introduzioni, NO virgolette, NO markdown.\n"
        "Se il prompt è già ottimo, rifinisci solo i dettagli.\n"
        f"Regola specifica: {profile_rules[profile]}\n"
        f"{target_hint}"
    )

def _cleanup_improved_prompt(text: str) -> str:
    """Pulisce l'output del modello rimuovendo fuffa e blocchi di codice."""
    cleaned = (text or "").strip()
    
    # Rimuove prefissi comuni
    for pattern in PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # Rimuove virgolette esterne residue
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    
    # Rimuove i blocchi di codice markdown se il modello ha racchiuso tutto lì dentro
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        
    return cleaned.strip()

def _request_prompt_optimization(
    prompt: str, system_instruction: str, config: Config
) -> requests.Response:
    """Gestisce la chiamata fisica a Ollama con fallback tra API Chat e Generate."""
    
    # Prepariamo le opzioni filtrando i None che farebbero fallire Ollama (400 Bad Request)
    options = {
        "temperature": config.OLLAMA_TEMPERATURE,
        "top_p": config.OLLAMA_TOP_P,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    }
    safe_options = {k: v for k, v in options.items() if v is not None}
    headers = {"Content-Type": "application/json"}

    # Tentativo 1: API /api/chat (standard per modelli moderni)
    chat_url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Migliora questo prompt:\n\n{prompt.strip()}"},
        ],
        "stream": False,
        "options": safe_options,
    }

    response = requests.post(chat_url, json=payload, headers=headers, timeout=config.OLLAMA_TIMEOUT)
    
    # Se otteniamo un 404, controlliamo se è l'endpoint o il modello a mancare
    if response.status_code == 404:
        try:
            err_msg = response.json().get("error", "").lower()
            if "model" in err_msg:
                return response # È il modello, inutile cambiare endpoint
        except:
            pass

        logger.warning("Fallback a /api/generate per Ollama")
        generate_url = f"{config.OLLAMA_BASE_URL}/api/generate"
        generate_payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": f"Migliora questo prompt:\n\n{prompt.strip()}",
            "system": system_instruction,
            "stream": False,
            "options": safe_options,
        }
        return requests.post(generate_url, json=generate_payload, headers=headers, timeout=config.OLLAMA_TIMEOUT)
    
    return response

def check_ollama_health(config: Config) -> bool:
    """Verifica se Ollama è attivo e se il modello configurato è scaricato."""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            names = [m.get("name", "") for m in models]
            if config.OLLAMA_MODEL in names or any(config.OLLAMA_MODEL in n for n in names):
                logger.info("Modello %s trovato", config.OLLAMA_MODEL)
                return True
            logger.warning("Modello %s non installato localmente", config.OLLAMA_MODEL)
        return response.status_code == 200
    except Exception:
        return False

def improve_prompt_with_ollama(
    prompt: str, config: Config, target_model: Optional[str] = None
) -> Dict[str, Any]:
    """Punto di ingresso principale per il miglioramento del prompt."""
    try:
        # 1. Validazione
        is_valid, error_msg = validate_prompt(prompt)
        if not is_valid:
            return {"success": False, "error": error_msg, "improved_prompt": None, "original_prompt": prompt, "elapsed_time": 0}

        # 2. Preparazione
        system_instruction = _build_system_instruction(prompt, target_model)
        start_time = time.time()
        
        # 3. Esecuzione richiesta
        response = _request_prompt_optimization(prompt, system_instruction, config)
        elapsed_time = time.time() - start_time

        # 4. Gestione Errori HTTP espliciti da Ollama
        if not response.ok:
            error_detail = "Errore API"
            try:
                error_detail = response.json().get("error", response.text)
            except:
                error_detail = response.text
            return {"success": False, "error": f"Ollama Error: {error_detail}", "improved_prompt": None, "original_prompt": prompt, "elapsed_time": elapsed_time}

        # 5. Parsing Risposta (gestisce sia chat che generate)
        result = response.json()
        message = result.get("message")
        raw_text = message.get("content", "") if message else result.get("response", "")
        
        improved_prompt = _cleanup_improved_prompt(raw_text)

        if not improved_prompt:
            return {"success": False, "error": "Il modello ha restituito una risposta vuota", "improved_prompt": None, "original_prompt": prompt, "elapsed_time": elapsed_time}

        return {
            "success": True,
            "error": None,
            "improved_prompt": improved_prompt,
            "original_prompt": prompt,
            "target_model": target_model,
            "elapsed_time": elapsed_time,
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout Ollama", "improved_prompt": None, "original_prompt": prompt, "elapsed_time": config.OLLAMA_TIMEOUT}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connessione Ollama fallita", "improved_prompt": None, "original_prompt": prompt, "elapsed_time": 0}
    except Exception as e:
        logger.exception("Errore imprevisto nel servizio Ollama")
        return {"success": False, "error": str(e), "improved_prompt": None, "original_prompt": prompt, "elapsed_time": 0}