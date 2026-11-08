"""Live telemetry (Bonus 2): hardware and model-server metrics on the same clock as the agent's events.

A background thread writes one JSON line every `interval` seconds to `<out>_logs/telemetry.jsonl`:
  - CPU % and RAM used (psutil), RSS of the model-server processes (ollama / vllm);
  - per-GPU utilisation and memory (nvidia-smi), when an NVIDIA GPU is present;
  - vLLM's Prometheus metrics from `<server>/metrics` (running / waiting requests, KV-cache use,
    token counters), when the server publishes them (Ollama does not; the poll then stops).
Timestamps are `time.time()`, the clock the per-question logs use for every model and tool call.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

VLLM_KEYS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:gpu_cache_usage_perc",
             "vllm:kv_cache_usage_perc", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
             "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")
_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")


def parse_prometheus(text: str, keys=VLLM_KEYS) -> dict[str, float]:
    """Sum each wanted metric over its label sets (vLLM labels every metric by model name)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM_LINE.match(line.strip())
        if m and m.group(1) in keys:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def metrics_url(base_url: str) -> str:
    """http://host:8000/v1 -> http://host:8000/metrics"""
    root = base_url.rstrip("/")
    return (root[:-3] if root.endswith("/v1") else root) + "/metrics"


def gpu_rows() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        util, mem = (x.strip() for x in line.split(","))
        rows.append({"util_pct": float(util), "mem_gb": round(float(mem) / 1024, 2)})
    return rows


class Sampler(threading.Thread):
    def __init__(self, out_path: Path, metrics: str | None = None, interval: float = 1.0):
        super().__init__(daemon=True)
        self.out_path, self.metrics, self.interval = out_path, metrics, interval
        self._halt = threading.Event()  # not `_stop`: that name shadows Thread._stop (the old project crashed on it)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=10)

    def sample(self) -> dict:
        import psutil

        vm = psutil.virtual_memory()
        server_rss = 0
        for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
            try:
                name = (p.info["name"] or "").lower()
                cmd = " ".join(p.info["cmdline"] or []).lower()
                if name.startswith("ollama") or "vllm" in cmd:
                    server_rss += p.info["memory_info"].rss
            except (psutil.Error, TypeError):
                continue
        row = {"t": time.time(), "cpu_pct": psutil.cpu_percent(interval=None),
               "ram_used_gb": round((vm.total - vm.available) / 1e9, 2), "server_rss_gb": round(server_rss / 1e9, 2)}
        for i, g in enumerate(gpu_rows()):
            row[f"gpu{i}_util_pct"], row[f"gpu{i}_mem_gb"] = g["util_pct"], g["mem_gb"]
        if self.metrics:
            try:
                with urllib.request.urlopen(self.metrics, timeout=2) as r:
                    row.update(parse_prometheus(r.read().decode("utf-8", "replace")))
            except Exception:
                self.metrics = None  # no /metrics here (Ollama): stop asking
        return row

    def run(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)  # prime the counter
        with self.out_path.open("w", encoding="utf-8") as f:
            while not self._halt.is_set():
                f.write(json.dumps(self.sample()) + "\n")
                f.flush()
                self._halt.wait(self.interval)
