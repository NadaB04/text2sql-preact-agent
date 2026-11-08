"""Runner (CONTRACT.md): answer every question, write results.jsonl and one log per question, then
score with the kit's scoring.py when the questions carry gold_sql.

    python -m text2sql_v2 --questions Q.json --db-dir DBS --base-url URL --model M --out results.jsonl

One full turn per question (all inside the 300 s limit):
    open the database read-only -> system prompt with the schema -> agent.solve -> final SQL
    -> answer step: the result rows in one or two plain sentences
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from . import agent, prompts
from .db import Database, db_path
from .llm import Config, OutOfSteps, OutOfTime, Session
from .telemetry import Sampler, metrics_url
from .tools import Toolbox

# CONTRACT.md's fields; the result also carries fallback_sql, answer and error.
RESULT_FIELDS = ("question_id", "db_id", "predicted_sql", "status", "steps", "tool_calls", "prompt_tokens",
                 "completion_tokens", "llm_time_s", "tool_time_s", "wall_time_s")


def answer_question(cfg: Config, q: dict, db_dir: str, client=None) -> tuple[dict, dict]:
    """One full turn: question -> SQL -> plain-English answer. Never raises.
    Returns (the results.jsonl line, the full log). ``client`` lets tests script the model."""
    t0 = time.monotonic()
    session = Session.start(cfg, client)
    status, error, box, answer, system = "ok", None, None, None, ""
    try:
        db = Database(db_path(db_dir, q["db_id"]))
    except Exception as exc:
        db, status, error = None, "error", f"{type(exc).__name__}: {exc}"
    if db is not None:
        try:
            system = prompts.SYSTEM.format(schema=db.schema())
            question = prompts.QUESTION.format(question=q["question"], evidence=q.get("evidence") or "(none)")
            box = Toolbox(db, session)  # made here, so the SQL found so far survives a step or time limit
            agent.solve(session, box, system, question)  # gold_sql is never passed
            if box.final_sql is None:
                status, error = "error", "no final SQL"
        except OutOfSteps:
            status = "max_steps"
        except OutOfTime:
            status = "timeout"
        except Exception as exc:  # one failing question must not stop the run
            status, error = "error", f"{type(exc).__name__}: {exc}"[:500]
        sql = final_sql(box)
        if cfg.answer and sql and session.left() > 10:
            answer = answer_in_words(session, db, q["question"], sql)
        db.close()
    sql = final_sql(box)
    result = {
        "question_id": q["question_id"], "db_id": q["db_id"], "predicted_sql": sql, "status": status,
        "steps": session.steps, "tool_calls": session.tool_calls, "prompt_tokens": session.prompt_tokens,
        "completion_tokens": session.completion_tokens, "llm_time_s": round(session.llm_time_s, 3),
        "tool_time_s": round(session.tool_time_s, 4), "wall_time_s": round(time.monotonic() - t0, 3),
        "fallback_sql": bool(box and box.final_sql is None and sql), "answer": answer, "error": error,
    }
    log = {**result, "question": q["question"], "evidence": q.get("evidence"), "difficulty": q.get("difficulty"),
           "system_prompt": system, "events": session.events}
    return result, log


def final_sql(box: Toolbox | None) -> str | None:
    """The submitted SQL, or else the last query that ran (best effort after a step/time limit)."""
    return (box.final_sql or box.last_ok_sql) if box else None


def answer_in_words(session: Session, db: Database, question: str, sql: str) -> str | None:
    """The last part of the turn: the result rows in words (one call, no tools, short)."""
    try:
        res = db.query(sql, max_rows=20, timeout_s=min(20, session.left() - 5))
        messages = [SystemMessage(prompts.ANSWER_SYSTEM),
                    HumanMessage(prompts.ANSWER.format(question=question, result=res.table(20)))]
        session.count_steps = False
        reply = session.chat(messages, label="answer", max_tokens=150)
        return str(reply.content).strip() or None
    except Exception as exc:
        session.note(type="answer_failed", error=str(exc)[:300])
        return None


# ---------------------------------------------------------------- command line
def parse_args(argv=None) -> argparse.Namespace:
    d = Config(base_url="", model="")
    p = argparse.ArgumentParser(prog="python -m text2sql_v2", description=__doc__.split("\n\n")[0])
    p.add_argument("--questions", required=True)
    p.add_argument("--db-dir", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=1, help="questions answered at the same time")
    p.add_argument("--max-steps", type=int, default=d.max_steps, help="model calls per question")
    p.add_argument("--time-limit", type=float, default=d.time_limit_s, help="seconds per question")
    p.add_argument("--call-timeout", type=float, default=d.call_timeout_s, help="seconds per model call")
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--think", action="store_true", help="turn on the model's built-in reasoning")
    p.add_argument("--no-answer", action="store_true", help="skip the plain-English answer")
    p.add_argument("--api-key", default=d.api_key)
    p.add_argument("--ids", help="comma-separated question_ids to run")
    p.add_argument("--limit", type=int, help="run only the first N questions")
    p.add_argument("--telemetry", action="store_true",
                   help="sample CPU/RAM/GPU and the server's /metrics every second into <out>_logs/telemetry.jsonl")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = Config(base_url=a.base_url, model=a.model, api_key=a.api_key, temperature=a.temperature,
                 max_tokens=a.max_tokens, time_limit_s=a.time_limit, call_timeout_s=a.call_timeout,
                 max_steps=a.max_steps, think=a.think, answer=not a.no_answer)
    questions = load_questions(a.questions, a.ids, a.limit)
    out = Path(a.out)
    logs = out.with_name(out.stem + "_logs")
    logs.mkdir(parents=True, exist_ok=True)
    run_info = {"config": {k: v for k, v in asdict(cfg).items() if k != "api_key"},
                "questions": a.questions, "n": len(questions), "concurrency": a.concurrency,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (logs / "run_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    write_lock = threading.Lock()

    def one(q: dict) -> dict:
        result, log = answer_question(cfg, q, a.db_dir)
        (logs / f"q{q['question_id']}.json").write_text(json.dumps(log, ensure_ascii=False, default=str, indent=1),
                                                         encoding="utf-8")
        with write_lock:  # one line as soon as the question is done, so a stopped run keeps its results
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        print(f"q{q['question_id']}: {result['status']} steps={result['steps']} {result['wall_time_s']:.0f}s", flush=True)
        return result

    sampler = Sampler(logs / "telemetry.jsonl", metrics_url(a.base_url)) if a.telemetry else None
    if sampler:
        sampler.start()
    try:  # questions run in parallel threads; each has its own Session, database connection and Toolbox
        with out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as pool:
            results = list(pool.map(one, questions))
    finally:
        if sampler:
            sampler.stop()
    print(f"wrote {out} ({len(results)} questions)")

    if questions and all("gold_sql" in q for q in questions):  # skip scoring when gold_sql is missing
        print(score(a.questions, a.db_dir, out, logs / "score_report.json"))
    return 0


def load_questions(path: str, ids: str | None, limit: int | None) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if ids:
        wanted = {s.strip() for s in ids.split(",")}
        questions = [q for q in questions if str(q["question_id"]) in wanted]
    return questions[:limit] if limit else questions


def score(questions: str, db_dir: str, results: Path, report: Path) -> str:
    """The kit's scoring.py (an unmodified copy sits next to this file)."""
    scorer = Path(__file__).with_name("scoring.py")
    cmd = [sys.executable, str(scorer), "--questions", questions, "--db-dir", db_dir, "--results", str(results),
           "--report", str(report)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + (f"\nscoring.py failed:\n{p.stderr}" if p.returncode else "")
