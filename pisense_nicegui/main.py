"""
PiSense Hub — applicazione NiceGUI standalone (cartella autonoma).

Struttura
---------
- `static/`           bundle Vite (HTML/CSS/JS) servito come file statici.
- `frontend/`         sorgenti React: dopo modifiche UI eseguire `npm install` e `npm run build`
  nella cartella `frontend/` per rigenerare `static/`.

Integrazione UI
---------------
- `ui.html` carica un iframe verso `/pisense_static/index.html` (stesso aspetto del design
  PiSense Hub / Firebase Studio, senza riscrivere la UI con componenti NiceGUI).
- I dati arrivano con `ui.run_javascript`: nell'iframe viene emesso
  `CustomEvent('pisense-metrics', { detail: payload })` (vedi `frontend/src/main.jsx`).

Logica applicativa
------------------
- Nessun routing REST applicativo: `monitoring_core.build_metrics_payload()` sostituisce
  l'ex endpoint FastAPI `GET /api/metrics`.

Eventi gauge → Python
---------------------
- Click gauge → `window.__PISENSE_GAUGE_CLICK__` nel contesto dell'iframe; un timer
  legge il valore con `await ui.run_javascript(...)` e mostra `ui.notify`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nicegui import app, ui

from monitoring_core import SystemMonitor, build_metrics_payload

_BASE = Path(__file__).resolve().parent
DIST_DIR = _BASE / "static"

POLL_INTERVAL_SEC = 3.0
GAUGE_POLL_SEC = 0.25

monitor = SystemMonitor(history_size=90)


def ensure_nicegui_index_patch() -> None:
    """Inserisce `window.__PISENSE_MODE__` in `static/index.html` (prima degli script modulo)."""
    index = DIST_DIR / "index.html"
    if not index.is_file():
        raise FileNotFoundError(
            f"Bundle UI assente: {index}. "
            "Esegui: cd frontend && npm install && npm run build"
        )
    text = index.read_text(encoding="utf-8")
    if "window.__PISENSE_MODE__" in text:
        return
    if "<head>" not in text:
        raise ValueError("index.html: tag <head> non trovato")
    injected = (
        "<head>\n"
        '    <script>window.__PISENSE_MODE__="nicegui"</script>'
    )
    text = text.replace("<head>", injected, 1)
    index.write_text(text, encoding="utf-8")


if DIST_DIR.is_dir():
    app.add_static_files("/pisense_static", str(DIST_DIR))

try:
    ensure_nicegui_index_patch()
except (FileNotFoundError, OSError):
    pass


def push_metrics_to_iframe() -> None:
    """Invia al iframe lo stesso JSON dell'ex `GET /api/metrics`."""
    payload = build_metrics_payload(monitor)
    js_payload = json.dumps(payload)
    ui.run_javascript(
        f"""
        const d = {js_payload};
        const iframe = document.getElementById('pisense-frame');
        const win = iframe && iframe.contentWindow;
        if (win) {{
            win.dispatchEvent(new CustomEvent('pisense-metrics', {{ detail: d }}));
        }}
        """
    )


@ui.page("/")
def home() -> None:
    ensure_nicegui_index_patch()

    ui.dark_mode().enable()

    ui.add_head_html(
        "<style>html,body{height:100%;margin:0;}"
        "iframe#pisense-frame{border:0;width:100%;height:100vh;display:block;}</style>"
    )

    ui.html(
        '<iframe id="pisense-frame" src="/pisense_static/index.html" title="PiSense Hub"></iframe>',
        sanitize=False,
    )

    ui.timer(POLL_INTERVAL_SEC, push_metrics_to_iframe)
    ui.timer(0.8, push_metrics_to_iframe, once=True)

    async def poll_gauge_click() -> None:
        try:
            name = await ui.run_javascript(
                """
                const iframe = document.getElementById('pisense-frame');
                const win = iframe && iframe.contentWindow;
                if (!win) return null;
                const v = win.__PISENSE_GAUGE_CLICK__;
                if (v != null && v !== '') {
                    win.__PISENSE_GAUGE_CLICK__ = '';
                    return v;
                }
                return null;
                """,
                timeout=2.0,
            )
        except Exception:
            return
        if name:
            ui.notify(f"Evento gauge → Python: {name}", position="bottom")

    ui.timer(GAUGE_POLL_SEC, poll_gauge_click)


if __name__ in {"__main__", "__mp_main__"}:
    ensure_nicegui_index_patch()
    port = int(os.environ.get("PISENSE_PORT", "6969"))
    ui.run(title="PiSense Hub", port=port, reload=False, show=True)
