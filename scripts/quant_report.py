"""Quantization comparison table: the same agent on two quantizations of one model.

    python scripts/quant_report.py --q8 runs/x/react.jsonl --q4 runs/x/react_q4km.jsonl \
        [--bench runs/x/bench_native.json] [--md out.md]

Accuracy, steps, tokens and time come from the runs (results.jsonl + the kit scorer's report);
prefill/decode rates, load time and memory from scripts/bench_native.py (Ollama's native API).
Paired accuracy difference: exact McNemar on the questions both runs answered.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from math import comb
from pathlib import Path


def load(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rep = path.with_name(path.stem + "_logs") / "score_report.json"
    correct = {r["question_id"]: r["correct"] for r in json.loads(rep.read_text(encoding="utf-8"))["per_question"]}
    return {"rows": rows, "correct": correct}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def med(xs) -> float:
    xs = list(xs)
    return statistics.median(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q8", type=Path, required=True)
    ap.add_argument("--q4", type=Path, required=True)
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    runs = {"Q8_0": load(a.q8), "Q4_K_M": load(a.q4)}
    bench = {}
    if a.bench and a.bench.exists():
        for b in json.loads(a.bench.read_text(encoding="utf-8")):
            bench["Q4_K_M" if b["model"] == "qwen3.5:4b" else "Q8_0" if "q8" in b["model"] else b["model"]] = b
    lines = ["| | Q8_0 | Q4_K_M |", "|---|---|---|"]

    def add(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(k) for k in ("Q8_0", "Q4_K_M")) + " |")

    n = len(runs["Q8_0"]["rows"])
    add("Correct", lambda k: f"{sum(runs[k]['correct'].values())}/{len(runs[k]['correct'])}")
    add("Status", lambda k: ", ".join(f"{s} {c}" for s, c in Counter(r["status"] for r in runs[k]["rows"]).most_common()))
    add("Median wall s / question", lambda k: f"{med(r['wall_time_s'] for r in runs[k]['rows']):.1f}")
    add("Median model-wait s / question", lambda k: f"{med(r['llm_time_s'] for r in runs[k]['rows']):.1f}")
    add("Mean model calls / question", lambda k: f"{statistics.mean(r['steps'] for r in runs[k]['rows']):.2f}")
    add("Mean prompt / output tokens", lambda k: f"{statistics.mean(r['prompt_tokens'] for r in runs[k]['rows']):,.0f} / "
                                                f"{statistics.mean(r['completion_tokens'] for r in runs[k]['rows']):,.0f}")
    add("Output tok/s seen by the agent", lambda k: f"{sum(r['completion_tokens'] for r in runs[k]['rows']) / max(1e-9, sum(r['llm_time_s'] for r in runs[k]['rows'])):.1f}")
    if bench:
        add("Prefill tok/s (native API, cache defeated)", lambda k: str(bench.get(k, {}).get("prefill_tok_s", "-")))
        add("Decode tok/s (native API)", lambda k: str(bench.get(k, {}).get("decode_tok_s", "-")))
        add("Load time s", lambda k: str(bench.get(k, {}).get("load_s", "-")))
        add("Server memory GB (weights + KV cache, /api/ps)", lambda k: str(bench.get(k, {}).get("server_size_gb", "-")))
        add("Of which on GPU GB", lambda k: str(bench.get(k, {}).get("server_vram_gb", "-")))
        add("nvidia-smi used GB", lambda k: str(bench.get(k, {}).get("nvidia_smi_used_gb", "-")))
        add("One call, cold prefix / warm prefix (s)", lambda k: f"{bench.get(k, {}).get('cold_total_s', '-')} / {bench.get(k, {}).get('warm_total_s', '-')}")
    c8, c4 = runs["Q8_0"]["correct"], runs["Q4_K_M"]["correct"]
    both = set(c8) & set(c4)
    b = sum(1 for q in both if c8[q] and not c4[q])
    c = sum(1 for q in both if c4[q] and not c8[q])
    text = "\n".join(lines) + f"\n\nPaired on {len(both)} questions: only Q8_0 right {b}, only Q4_K_M right {c}, exact McNemar p = {mcnemar(b, c):.3g} ({n} questions in the Q8_0 run)."
    print(text)
    if a.md:
        a.md.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
