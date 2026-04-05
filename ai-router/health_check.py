#!/usr/bin/env python
"""Health check per il servizio AI Router (Gradio + Ollama)."""
import sys

from config import Config
from ollama_service import check_ollama_health


def check_gradio_health(config: Config) -> bool:
    try:
        import requests
        host = "127.0.0.1" if config.GRADIO_SERVER_NAME == "0.0.0.0" else config.GRADIO_SERVER_NAME
        r = requests.get(f"http://{host}:{config.GRADIO_SERVER_PORT}", timeout=5)
        return r.status_code in (200, 405)
    except Exception:
        return False


if __name__ == "__main__":
    conf = Config()
    gradio_ok = check_gradio_health(conf)
    print(f"Gradio: {'✓' if gradio_ok else '✗'}")
    print(f"Ollama: {'✓' if check_ollama_health(conf) else '✗'}")
    sys.exit(0 if gradio_ok else 1)
