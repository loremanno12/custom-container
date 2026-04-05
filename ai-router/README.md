# 🤖 AI Router System

Sistema di routing intelligente che predice quale modello AI utilizzare in base ai prompt degli utenti. Utilizza SentenceTransformer per gli embeddings, scikit-learn per la classificazione e Ollama per il miglioramento automatico dei prompt.

## ✨ Funzionalità

- **Predizione intelligente**: Predice il modello AI migliore per ogni prompt
- **Miglioramento automatico**: Migliora i prompt usando Ollama
- **Interfaccia web moderna**: UI accattivante con Gradio
- **Optimizzato per Raspberry Pi**: Leggero, modulare e Docker-ready
- **Health checks**: Monitoraggio della salute dei servizi

## 🚀 Quick Start

### Prerequisiti

- Python 3.11+
- (Opzionale) Docker
- Ollama in esecuzione su `http://localhost:11434`

### Installazione

1. **Clone/scarica il progetto**

2. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

3. **Avvia l'applicazione**
```bash
python router_main.py
```

4. **Accedi all'interfaccia**
- Web: http://localhost:7860
- Ollama: http://localhost:11434

## 📋 Configurazione

Modifica il file `.env` per personalizzare:

```bash
# Modello di embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Modello Ollama
OLLAMA_MODEL=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=60
OLLAMA_TEMPERATURE=0.3
OLLAMA_TOP_P=0.9
OLLAMA_NUM_PREDICT=450

# Parametri della predizione
CONFIDENCE_THRESHOLD=0.5
TOP_N_PREDICTIONS=3
EMBEDDING_BATCH_SIZE=16
NORMALIZE_EMBEDDINGS=true
EMBEDDING_DEVICE=cpu

# Porta e host di Gradio
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_CONCURRENCY_LIMIT=1
GRADIO_QUEUE_SIZE=16

# Ottimizzazioni Raspberry Pi
CPU_THREADS=2
RETRAIN_ON_DATA_CHANGE=false
```

## 📚 Struttura del Progetto

```
├── router_main.py         # Entry point principale
├── config.py              # Configurazione centralizzata
├── cache.py               # Cache dei modelli ML
├── training.py            # Logica di addestramento
├── ollama_service.py      # Integrazione Ollama
├── predictor.py           # Logica di predizione
├── ui.py                  # Interfaccia Gradio (tema dark, ottimizzata)
├── health_check.py        # Script health check
├── Dockerfile             # Docker image
├── docker-compose.yml     # Avvio container
├── .dockerignore          # Riduce il build context
├── requirements.txt       # Dipendenze Python
├── training_data.json     # Dati di addestramento
└── README.md              # Questo file
```

## 🎓 Dati di Addestramento

Crea un file `training_data.json` con il seguente formato:

```json
[
  {
    "modello": "GPT-4",
    "prompts": [
      "Scrivi una funzione Python",
      "Debug questo codice",
      "Crea un algoritmo efficiente"
    ]
  },
  {
    "modello": "Claude",
    "prompts": [
      "Spiega il quantum computing",
      "Analizza questo testo",
      "Genera una storia creativa"
    ]
  }
]
```

## 🔧 Comandi Utili

### Avviare il servizio
```bash
python router_main.py
```

### Visualizzare i log
```bash
python router_main.py
```

### Health check
```bash
python health_check.py
```

### Build Docker
```bash
docker build -t ai-router .
```

### Avvio con Docker Compose
```bash
docker compose up -d --build
```

## 📊 Come Funziona

1. **Addestramento** (automatico al primo avvio):
   - Carica i dati da `training_data.json`
   - Genera embeddings con SentenceTransformer
   - Addestra un classificatore MLP

2. **Predizione**:
   - Riceve il prompt dall'utente
   - Genera l'embedding del prompt
   - Predice il modello migliore
   - Mostra i top 3 risultati con confidenza

3. **Miglioramento** (opzionale):
   - Invia il prompt a Ollama
   - Riceve un prompt ottimizzato
   - Visualizza il confronto

## 

- **Primo avvio**: ~2-3 minuti (download modelli)
- **Predizione**: <1 secondo (GPU) / ~2-3 secondi (CPU)
- **Miglioramento prompt**: ~5-15 secondi (dipende da Ollama)
- **Memoria**: ~1-2GB per i modelli

## 

### Ollama non si connette
```bash
# Verifica che Ollama sia disponibile
curl http://localhost:11434/api/tags

# Pull del modello
docker-compose exec ollama ollama pull gemma2:2b
```

### Out of memory
- Riduci `MLP_HIDDEN_LAYERS` in `.env`
- Usa un modello di embedding piu piccolo

### Gradio non risponde
```bash
docker-compose logs -f ai-router
```

### Health check Docker
Il container usa `python3 health_check.py` invece di un endpoint fisso, cosi il check resta stabile anche se Gradio cambia internamente alcune route.

## 

MIT

##
