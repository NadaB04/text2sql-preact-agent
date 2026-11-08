"""One picture per run: every question's model and tool calls on a shared clock, above the telemetry panels.

    python scripts/plot_timeline.py runs/<run>_logs [--out timeline.png]

Top panel: one lane per question, blue = waiting for the model, orange = running a tool. Below it,
one panel per metric (CPU, RAM, GPU util/memory, vLLM running/waiting requests, KV-cache use), all on
the same time axis, each with its own y-axis (small multiples, no dual axes). Also prints a few numbers
the picture shows: model calls in flight over time and the GPU's busy share.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LLM, TOOL, SERVER = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(log_dir: Path) -> tuple[list[dict], list[dict]]:
    events = []
    for f in sorted(log_dir.glob("q*.json")):
        log = json.loads(f.read_text(encoding="utf-8"))
        for e in log.get("events", []):
            if e.get("type") in ("llm", "llm_error", "tool") and "duration_s" in e:
                events.append({"q": log["question_id"], "kind": "tool" if e["type"] == "tool" else "llm",
                               "start": e["start"], "end": e["start"] + e["duration_s"]})
    tp = log_dir / "telemetry.jsonl"
    tele = [json.loads(line) for line in tp.read_text(encoding="utf-8").splitlines() if line.strip()] if tp.exists() else []
    return events, tele


def panels_for(tele: list[dict]) -> list[tuple[str, str]]:
    keys = set().union(*tele) if tele else set()
    out = [("cpu_pct", "CPU %"), ("ram_used_gb", "RAM used GB")]
    for i in range(8):
        if f"gpu{i}_util_pct" in keys:
            out += [(f"gpu{i}_util_pct", f"GPU{i} util %"), (f"gpu{i}_mem_gb", f"GPU{i} mem GB")]
    for k, label in (("vllm:num_requests_running", "vLLM running"), ("vllm:num_requests_waiting", "vLLM waiting"),
                     ("vllm:kv_cache_usage_perc", "KV cache use"), ("vllm:gpu_cache_usage_perc", "KV cache use")):
        if k in keys and label not in [p[1] for p in out]:
            out.append((k, label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    events, tele = load(a.log_dir)
    if not events:
        raise SystemExit(f"no events in {a.log_dir}")
    t0 = min([e["start"] for e in events] + [r["t"] for r in tele])
    t1 = max(e["end"] for e in events)
    tele = [r for r in tele if r["t"] <= t1 + 5]
    order = sorted({e["q"] for e in events}, key=lambda q: min(e["start"] for e in events if e["q"] == q))
    lane = {q: i for i, q in enumerate(order)}
    panels = panels_for(tele)
    heights = [max(2.5, 0.16 * len(order))] + [1.0] * len(panels)
    fig, axes = plt.subplots(len(heights), 1, figsize=(12, sum(heights) + 1), sharex=True,
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    axes = axes[:, 0]
    ax = axes[0]
    for e in events:
        ax.barh(lane[e["q"]], max(e["end"] - e["start"], 0.2), left=e["start"] - t0, height=0.75,
                color=LLM if e["kind"] == "llm" else TOOL, linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel(f"{len(order)} questions", fontsize=8, color=MUTED)
    ax.invert_yaxis()
    ax.set_title(f"Timeline: {a.log_dir.name.removesuffix('_logs')}", loc="left", color=INK, fontsize=11)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LLM), plt.Rectangle((0, 0), 1, 1, color=TOOL)],
              labels=["waiting for the model", "running a tool"], loc="upper right", frameon=False, fontsize=8)
    ts = [r["t"] - t0 for r in tele]
    for axp, (key, label) in zip(axes[1:], panels):
        axp.plot(ts, [r.get(key, float("nan")) for r in tele], color=SERVER if key.startswith("vllm") else MUTED,
                 linewidth=1.2)
        axp.set_ylabel(label, fontsize=8, color=MUTED, rotation=0, ha="right", va="center")
    for axp in axes:
        axp.grid(axis="x", color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            axp.spines[s].set_visible(False)
        axp.tick_params(colors=MUTED, labelsize=7)
    axes[-1].set_xlabel("seconds since the run started", color=MUTED, fontsize=8)
    fig.tight_layout()
    out = a.out or a.log_dir / "timeline.png"
    fig.savefig(out, dpi=130)

    # numbers behind the picture
    llm = [e for e in events if e["kind"] == "llm"]
    grid = [t0 + s for s in range(int(t1 - t0) + 1)]
    inflight = [sum(1 for e in llm if e["start"] <= g < e["end"]) for g in grid]
    llm_s = sum(e["end"] - e["start"] for e in llm)
    tool_s = sum(e["end"] - e["start"] for e in events if e["kind"] == "tool")
    print(f"wrote {out}: wall {t1 - t0:.0f}s, model calls {len(llm)}, model-wait {llm_s:.0f}s vs tools {tool_s:.1f}s, "
          f"model calls in flight mean {sum(inflight) / len(inflight):.1f} max {max(inflight)}")
    for key, label in panels:
        vals = [r[key] for r in tele if isinstance(r.get(key), (int, float))]
        if vals:
            print(f"  {label}: mean {sum(vals) / len(vals):.2f} max {max(vals):.2f}")


if __name__ == "__main__":
    main()
