"""
Logica di monitoraggio di sistema (ex endpoint FastAPI /api/metrics).

Sostituisce:
    @app.get("/api/metrics")
    def metrics(): ...

con una funzione Python pura `build_metrics_payload()` richiamabile dal processo NiceGUI,
senza routing HTTP né serializzazione tramite REST.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from typing import Any

import docker
import psutil


def disk_usage_percent() -> float:
    """Percentuale disco del volume di sistema (Linux `/`, Windows drive di sistema)."""
    paths: list[str] = []
    if os.name == "nt":
        paths.append(os.environ.get("SystemDrive", "C:") + "\\")
    paths.extend(["/", "."])
    for path in paths:
        try:
            return float(psutil.disk_usage(path).percent)
        except Exception:
            continue
    return 0.0


class SystemMonitor:
    """Stesso comportamento della classe in `fast_api/backend/main.py`."""

    def __init__(self, history_size: int = 60) -> None:
        self.history_size = history_size
        self.cpu_history: deque[float] = deque(maxlen=history_size)
        self.temp_history: deque[float] = deque(maxlen=history_size)
        self.ram_history: deque[float] = deque(maxlen=history_size)
        self.net_rx_history: deque[float] = deque(maxlen=history_size)
        self.net_tx_history: deque[float] = deque(maxlen=history_size)
        self.last_net_io = psutil.net_io_counters()
        self.last_net_time = time.time()
        try:
            self.docker_client = docker.from_env()
        except Exception:
            self.docker_client = None

    def get_uptime(self) -> str:
        try:
            with open("/proc/uptime", "r") as f:
                uptime_seconds = int(float(f.read().split()[0]))
            days, remainder = divmod(uptime_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)

            parts: list[str] = []
            if days:
                parts.append(f"{days}d")
            if hours:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")
            return "up " + " ".join(parts)
        except Exception:
            return "up n/d"

    def get_cpu_temp(self) -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read()) / 1000.0
        except Exception:
            try:
                res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True)
                return float(res.stdout.replace("temp=", "").replace("'C\n", ""))
            except Exception:
                return 0.0

    def get_stats(self) -> dict[str, Any]:
        cpu_usage = psutil.cpu_percent()
        temp = self.get_cpu_temp()
        self.cpu_history.append(cpu_usage)
        self.temp_history.append(temp)

        mem = psutil.virtual_memory()
        self.ram_history.append(mem.percent)

        net_io = psutil.net_io_counters()
        now_time = time.time()
        dt = now_time - self.last_net_time
        rx_s = (net_io.bytes_recv - self.last_net_io.bytes_recv) / dt / 1024 / 1024
        tx_s = (net_io.bytes_sent - self.last_net_io.bytes_sent) / dt / 1024 / 1024
        self.net_rx_history.append(rx_s)
        self.net_tx_history.append(tx_s)
        self.last_net_io = net_io
        self.last_net_time = now_time

        return {
            "labels": list(range(len(self.cpu_history))),
            "cpu": list(self.cpu_history),
            "temp": list(self.temp_history),
            "ram": list(self.ram_history),
            "net_rx": list(self.net_rx_history),
            "net_tx": list(self.net_tx_history),
            "mem_info": f"{mem.percent}% ({mem.used // 1024**2}MB / {mem.total // 1024**2}MB)",
            "uptime": self.get_uptime(),
            "disk_percent": disk_usage_percent(),
        }

    def get_docker_containers(self) -> list[list[str]]:
        if not self.docker_client:
            return [["Docker non disponibile", "-", "-"]]
        try:
            containers = self.docker_client.containers.list(all=True)
            if not containers:
                return [["Nessun container", "-", "-"]]
            return [
                [c.name, c.status, c.image.tags[0] if c.image.tags else c.short_id]
                for c in containers
            ]
        except Exception as exc:
            return [[f"Errore Docker: {str(exc)}", "-", "-"]]


def build_metrics_payload(monitor: SystemMonitor) -> dict[str, Any]:
    """
    Payload identico a quello restituito dall'ex endpoint GET /api/metrics.
    Il frontend React (build Vite / grafica PiSense Hub) si aspetta questa forma.
    """
    stats = monitor.get_stats()

    points = [
        {
            "index": idx,
            "cpu": round(cpu, 2),
            "ram": round(stats["ram"][idx], 2),
            "temp": round(stats["temp"][idx], 2),
            "net_rx": round(stats["net_rx"][idx], 3),
            "net_tx": round(stats["net_tx"][idx], 3),
        }
        for idx, cpu in enumerate(stats["cpu"])
    ]

    last = points[-1] if points else {}

    return {
        "points": points,
        "summary": {
            "uptime": stats["uptime"],
            "memory_info": stats["mem_info"],
            "containers": monitor.get_docker_containers(),
            "disk_percent": round(float(stats.get("disk_percent", 0)), 1),
            "status": "Online",
        },
        "snapshot": {
            "cpu": last.get("cpu", 0),
            "ram": last.get("ram", 0),
            "disk": round(float(stats.get("disk_percent", 0)), 1),
            "temp": last.get("temp", 0),
            "net_rx": last.get("net_rx", 0),
            "net_tx": last.get("net_tx", 0),
        },
    }
