"""Compare runs: accuracy, cost, and whether the agent's mechanism actually happened.

    python scripts/summarize.py runs/final_ollama/preact_q8.jsonl runs/final_ollama/preact_q4km.jsonl [--baseline preact_q8] [--traces 1] [--json out.json]

Correctness comes from each run's score_report.json (written by the kit's scoring.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from math import comb
from pathlib import Path


def load(path: Path) -> dict:
    logs = path.with_name(path.stem + "_logs")
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads((logs / "score_report.json").read_text(encoding="utf-8")) if (logs / "score_report.json").exists() else {}
    correct = {r["question_id"]: r["correct"] for r in report.get("per_question", [])}
    qlogs = {r["question_id"]: json.loads((logs / f"q{r['question_id']}.json").read_text(encoding="utf-8"))
             for r in results if (logs / f"q{r['question_id']}.json").exists()}
    return {"name": path.stem, "results": results, "correct": correct, "logs": qlogs}


def mechanism(run: dict) -> str:
    """Did the mechanism happen: a thought on every call, a prediction on every query, free text dropped?"""
    ev = [e for lg in run["logs"].values() for e in lg["events"]]
    llm = [e for e in ev if e["type"] == "llm" and e.get("label") != "answer"]
    tools = [e for e in ev if e["type"] == "tool"]
    out = []
    agent_calls = [c for e in llm for c in e["tool_calls"]]
    if agent_calls:
        with_thought = sum(1 for c in agent_calls if len(str(c["args"].get("thought", "")).strip()) >= 3)
        multi = sum(1 for e in llm if len(e["tool_calls"]) > 1)
        rejected = sum(1 for t in tools if t["output"].startswith("ERROR: invalid arguments"))
        agent_turns = [e for e in llm if e["tool_calls"]]
        extra_text = sum(1 for e in agent_turns if str(e.get("content") or "").strip())
        out.append(f"tool calls with a thought {with_thought}/{len(agent_calls)}; turns with >1 tool call {multi}; "
                   f"rejected for missing fields {rejected}; turns with free text next to the call (dropped from "
                   f"history) {extra_text}/{len(agent_turns)}")
        queries = [c for c in agent_calls if c["name"] in ("run_query", "column_values")]
        preds = sum(1 for c in queries if len(str(c["args"].get("prediction", "")).strip()) >= 3)
        out.append(f"queries with a prediction {preds}/{len(queries)}")
        repeats = sum(1 for t in tools if t["output"].startswith("You already ran"))
        if repeats:
            out.append(f"repeat-guard hits {repeats}")
    kinds = Counter(e["type"] for e in ev)
    if kinds["nudge"]:
        out.append(f"nudges {kinds['nudge']}")
    return "; ".join(out) or "-"


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired correctness."""
    ids = set(a) & set(b)
    x = sum(1 for i in ids if a[i] and not b[i])
    y = sum(1 for i in ids if b[i] and not a[i])
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


def trace(lg: dict, limit: int = 12) -> str:
    """A compact view of one question: each model call's tool calls with thoughts, and observations."""
    lines = [f"  Q{lg['question_id']} [{lg['status']}] {lg['question'][:110]}"]
    for e in lg["events"][: limit * 3]:
        if e["type"] == "llm":
            for c in e["tool_calls"]:
                a = dict(c["args"])
                th, pr = a.pop("thought", None), a.pop("prediction", None)
                lines.append(f"    {e['label']}> {c['name']}({json.dumps(a, ensure_ascii=False)[:160]})")
                if th:
                    lines.append(f"       thought: {str(th)[:200]}")
                if pr:
                    lines.append(f"       prediction: {str(pr)[:160]}")
            if not e["tool_calls"] and e.get("content"):
                lines.append(f"    {e['label']}> text: {e['content'][:160]!r}")
        elif e["type"] == "tool":
            lines.append(f"       -> {e['output'][:160]!r}")
        elif e["type"] == "nudge":
            lines.append(f"    [{e['type']}] {json.dumps({k: v for k, v in e.items() if k not in ('type', 'start')})[:200]}")
    lines.append(f"    final: {lg['predicted_sql']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--baseline", default=None, help="run name to compare others with (McNemar)")
    ap.add_argument("--traces", type=int, default=0, help="print N wrong-answer traces per run")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs = [load(p) for p in a.runs]

    print("| run | correct | acc | status | steps/q | prompt tok/q | output tok/q | median wall s | model s total |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for r in runs:
        res, ok = r["results"], [v for v in r["correct"].values() if v is not None]
        row = {
            "correct": sum(ok), "n": len(ok), "accuracy": round(sum(ok) / len(ok), 4) if ok else None,
            "status": dict(Counter(x["status"] for x in res)),
            "steps": round(statistics.mean(x["steps"] for x in res), 2),
            "prompt_tokens": round(statistics.mean(x["prompt_tokens"] for x in res)),
            "completion_tokens": round(statistics.mean(x["completion_tokens"] for x in res)),
            "median_wall_s": round(statistics.median(x["wall_time_s"] for x in res), 1),
            "llm_s": round(sum(x["llm_time_s"] for x in res), 1), "mechanism": mechanism(r),
            "correct_ids": sorted(q for q, v in r["correct"].items() if v),
        }
        summary[r["name"]] = row
        st = " ".join(f"{k}={v}" for k, v in sorted(row["status"].items()))
        acc = f"{row['accuracy']:.0%}" if row["accuracy"] is not None else "-"
        print(f"| {r['name']} | {row['correct']}/{row['n']} | {acc} | {st} | {row['steps']} | {row['prompt_tokens']} | "
              f"{row['completion_tokens']} | {row['median_wall_s']} | {row['llm_s']} |")
    print("\nMechanism checks (from the logs):")
    for r in runs:
        print(f"- {r['name']}: {summary[r['name']]['mechanism']}")
    if a.baseline and any(r["name"] == a.baseline for r in runs):
        base = next(r for r in runs if r["name"] == a.baseline)
        print(f"\nPaired with {a.baseline} (exact McNemar): only-this / only-baseline / p")
        for r in runs:
            if r is not base:
                x, y, p = mcnemar(r["correct"], base["correct"])
                print(f"- {r['name']}: +{x} / -{y}, p = {p:.3f}")
    ids = sorted({q for r in runs for q in r["correct"]})
    print("\nPer question (1 = correct):")
    print("qid    " + " ".join(f"{r['name'][:6]:>6}" for r in runs))
    for q in ids:
        print(f"{q:<6} " + " ".join(f"{('1' if r['correct'].get(q) else '.'):>6}" for r in runs))
    if a.traces:
        for r in runs:
            wrong = [lg for q, lg in r["logs"].items() if r["correct"].get(q) is False][: a.traces]
            right = [lg for q, lg in r["logs"].items() if r["correct"].get(q)][:1]
            print(f"\n=== traces: {r['name']} ===")
            for lg in right + wrong:
                print(trace(lg))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
