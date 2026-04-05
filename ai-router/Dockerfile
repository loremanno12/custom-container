# AI Router - Ottimizzato per Raspberry Pi 5 (immagine leggera, torch CPU-only)
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

FROM python:3.11-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH="/opt/venv/bin:${PATH}" \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid "${APP_GID}" app && \
    useradd --uid "${APP_UID}" --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app cache.py config.py health_check.py metrics.py ollama_service.py predictor.py router_main.py training.py ui.py requirements.txt ./
COPY --chown=app:app training_data.json .
RUN mkdir -p /app/models /app/logs && chown -R app:app /app

USER app

EXPOSE 7860

CMD ["python", "router_main.py"]
