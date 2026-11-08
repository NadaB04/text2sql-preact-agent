"""Server speed and memory per quantization, separate from agent behaviour.

Replays the same prompts (the first model call of each logged question: system prompt with the schema +
the task and question) against each model through Ollama's native /api/chat, which reports where the
time goes:
    prompt_eval_duration -> prefill: reading the prompt (compute-bound)
    eval_duration        -> decode: generating tokens (memory-bandwidth-bound)
    load_duration        -> loading the weights (first call after an unload)
Two passes per model: `cold` puts a random nonce at the start of the system prompt so the prefix cache
cannot skip the prefill; `warm` sends the same prompt again right after, as the agent does on its next
step (only the new tokens are evaluated). Memory: /api/ps (weights + KV cache, and how much is on the
GPU) and nvidia-smi.

    python scripts/bench_native.py --logs runs/x_logs --models qwen3.5:4b-q8_0 qwen3.5:4b --n 10 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import urllib.request
import uuid
from pathlib import Path


def post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def first_prompts(log_dir: Path, n: int) -> list[list[dict]]:
    out = []
    for f in sorted(log_dir.glob("q*.json"))[:n]:
        log = json.loads(f.read_text(encoding="utf-8"))
        first = next(e for e in log["events"] if e["type"] == "llm")
        user = next(m["content"] for m in first["input"] if m["role"] == "human")
        out.append([{"role": "system", "content": log["system_prompt"]}, {"role": "user", "content": user}])
    return out


def gpu_mem_gb() -> list[float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [round(float(x) / 1024, 2) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def row(r: dict) -> dict:
    return {"prompt_tokens": r.get("prompt_eval_count", 0), "prefill_s": r.get("prompt_eval_duration", 0) / 1e9,
            "gen_tokens": r.get("eval_count", 0), "decode_s": r.get("eval_duration", 0) / 1e9,
            "load_s": r.get("load_duration", 0) / 1e9, "total_s": r.get("total_duration", 0) / 1e9}


def rate(rows: list[dict], tok: str, sec: str) -> float | None:
    vals = [r[tok] / r[sec] for r in rows if r[sec] > 0 and r[tok] > 0]
    return round(statistics.median(vals), 1) if vals else None


def bench(host: str, model: str, prompts: list[list[dict]], num_predict: int, all_models: list[str]) -> dict:
    for m in all_models:  # unload everything: GPU memory then belongs to this model, and the first call measures the load
        post(f"{host}/api/generate", {"model": m, "keep_alive": 0})
    options = {"temperature": 0, "num_predict": num_predict, "seed": 0, "num_ctx": 16384}
    cold, warm, mem = [], [], {}
    for i, msgs in enumerate(prompts):
        msgs = [dict(m) for m in msgs]
        msgs[0]["content"] = f"[run {uuid.uuid4().hex[:8]}]\n" + msgs[0]["content"]
        body = {"model": model, "messages": msgs, "stream": False, "think": False, "options": options}
        cold.append(row(post(f"{host}/api/chat", body)))
        warm.append(row(post(f"{host}/api/chat", body)))
        if i == 0:
            info = next((m for m in post_get(f"{host}/api/ps")["models"] if m["name"] == model), {})
            mem = {"server_size_gb": round(info.get("size", 0) / 1e9, 2),
                   "server_vram_gb": round(info.get("size_vram", 0) / 1e9, 2),
                   "context_length": info.get("context_length"), "nvidia_smi_used_gb": gpu_mem_gb()}
    return {"model": model, "n": len(cold), **mem, "load_s": round(cold[0]["load_s"], 1),
            "prefill_tok_s": rate(cold, "prompt_tokens", "prefill_s"),
            "decode_tok_s": rate(cold + warm, "gen_tokens", "decode_s"),
            "cold_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in cold)),
            "warm_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in warm)),
            "cold_total_s": round(statistics.median(r["total_s"] for r in cold[1:] or cold), 2),
            "warm_total_s": round(statistics.median(r["total_s"] for r in warm), 2),
            "cold": cold, "warm": warm}


def post_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--logs", required=True, help="a run's log folder; its first prompts are replayed")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    prompts = first_prompts(Path(a.logs), a.n)
    results = []
    for model in a.models:
        res = bench(a.host, model, prompts, a.num_predict, a.models)
        results.append(res)
        print(f"{model:<18} prefill {res['prefill_tok_s']} tok/s  decode {res['decode_tok_s']} tok/s  "
              f"load {res['load_s']} s  size {res['server_size_gb']} GB (GPU {res['server_vram_gb']} GB)  "
              f"nvidia-smi {res['nvidia_smi_used_gb']}  prompt tokens cold {res['cold_prompt_tokens']} / "
              f"warm {res['warm_prompt_tokens']}  call s cold {res['cold_total_s']} / warm {res['warm_total_s']}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
