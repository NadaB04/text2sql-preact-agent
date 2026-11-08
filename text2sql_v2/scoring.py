#!/usr/bin/env python3
"""Scoring for the text-to-SQL home assignment.

Use it from Python:

    from scoring import is_correct
    is_correct("data/databases/superhero/superhero.sqlite", predicted_sql, gold_sql)

or from the command line, to check a whole results file (format and accuracy):

    python scoring.py --questions data/questions_dev.json \
                      --db-dir data/databases --results results.jsonl

A prediction is correct when it returns the same set of rows as the gold query.
This is BIRD's execution-accuracy metric: row order and duplicate rows are
ignored, and values must match exactly. Both queries run on a read-only
connection with a time limit. Standard library only, Python 3.11+.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

DEFAULT_TIMEOUT_S = 30.0
STATUSES = ("ok", "error", "timeout", "max_steps")
NUMBER = (int, float)
REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "question_id": (int, str),
    "db_id": (str,),
    "predicted_sql": (str, type(None)),
    "status": (str,),
    "steps": (int,),
    "tool_calls": (int,),
    "prompt_tokens": (int,),
    "completion_tokens": (int,),
    "llm_time_s": NUMBER,
    "tool_time_s": NUMBER,
    "wall_time_s": NUMBER,
}
# Errors a bad prediction can raise; any of them simply means "incorrect".
PREDICTION_ERRORS = (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError)


class QueryTimeout(Exception):
    """The query ran longer than the time limit."""


def db_path_for(db_dir: str | Path, db_id: str) -> Path:
    """Where CONTRACT.md says a database lives."""
    return Path(db_dir) / db_id / f"{db_id}.sqlite"


def run_query(db_path: str | Path, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[tuple]:
    """Run one SQL statement on a read-only connection and return all rows.

    Raises sqlite3.Error for invalid SQL (including any attempt to write),
    and QueryTimeout if the query runs longer than timeout_s.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    expired = False

    def check_deadline() -> int:
        nonlocal expired
        expired = time.monotonic() > deadline
        return 1 if expired else 0  # a non-zero return aborts the query

    con.set_progress_handler(check_deadline, 10_000)
    try:
        return con.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        if expired:
            raise QueryTimeout(f"query exceeded {timeout_s:g} s") from exc
        raise
    finally:
        con.close()


@lru_cache(maxsize=512)
def _gold_rows(db_path: str, gold_sql: str, timeout_s: float) -> frozenset:
    return frozenset(run_query(db_path, gold_sql, timeout_s))


def is_correct(db_path: str | Path, predicted_sql: str | None, gold_sql: str,
               timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    """True if predicted_sql returns the same set of rows as gold_sql.

    An empty prediction, invalid SQL, a write attempt or a timeout all count
    as incorrect. A failing gold query raises, because that is a problem with
    the kit, not with the prediction.
    """
    if not isinstance(predicted_sql, str) or not predicted_sql.strip():
        return False
    gold = _gold_rows(str(Path(db_path).resolve()), gold_sql, float(timeout_s))
    try:
        predicted = run_query(db_path, predicted_sql, timeout_s)
    except (QueryTimeout, *PREDICTION_ERRORS):
        return False
    return set(predicted) == gold


def validate_record(record: object) -> list[str]:
    """Format problems in one results.jsonl record (empty list if none)."""
    if not isinstance(record, dict):
        return ["line is not a JSON object"]
    problems = []
    for field, types in REQUIRED_FIELDS.items():
        if field not in record:
            problems.append(f"missing field '{field}'")
        elif isinstance(record[field], bool) or not isinstance(record[field], types):
            expected = " or ".join("null" if t is type(None) else t.__name__ for t in types)
            problems.append(f"'{field}' is {type(record[field]).__name__}, expected {expected}")
    status = record.get("status")
    if isinstance(status, str) and status not in STATUSES:
        problems.append(f"'status' is '{status}', expected one of: {', '.join(STATUSES)}")
    return problems


def _accuracy(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    correct = sum(r["correct"] for r in scored)
    return {"scored": len(scored), "correct": correct,
            "accuracy": round(correct / len(scored), 4) if scored else None}


def score_file(questions_path: str | Path, db_dir: str | Path, results_path: str | Path,
               timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Check a results.jsonl file against a questions file and score it."""
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    by_id = {str(q["question_id"]): q for q in questions}
    records: dict[str, dict] = {}
    problems: list[str] = []
    duplicates: list[str] = []
    unknown: list[str] = []

    lines = Path(results_path).read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {n}: not valid JSON ({exc.msg})")
            continue
        problems += [f"line {n}: {p}" for p in validate_record(record)]
        if not isinstance(record, dict) or "question_id" not in record:
            continue
        qid = str(record["question_id"])
        if qid not in by_id:
            unknown.append(qid)
        elif qid in records:
            duplicates.append(qid)
        else:
            records[qid] = record

    per_question = []
    for qid, q in by_id.items():
        record = records.get(qid)
        row = {"question_id": q["question_id"], "db_id": q["db_id"],
               "difficulty": q.get("difficulty", "unknown"),
               "status": record.get("status") if record else "missing", "correct": None}
        if q.get("gold_sql"):
            row["correct"] = record is not None and is_correct(
                db_path_for(db_dir, q["db_id"]), record.get("predicted_sql"), q["gold_sql"], timeout_s)
        per_question.append(row)

    def grouped(key: str) -> dict:
        groups = defaultdict(list)
        for r in per_question:
            groups[r[key]].append(r)
        return {k: _accuracy(v) for k, v in sorted(groups.items())}

    def numbers(field: str) -> list[float]:
        return [r[field] for r in records.values()
                if isinstance(r.get(field), NUMBER) and not isinstance(r.get(field), bool)]

    wall, llm, tool = numbers("wall_time_s"), numbers("llm_time_s"), numbers("tool_time_s")
    return {
        "questions": len(by_id),
        "answered": len(records),
        **_accuracy(per_question),
        "by_difficulty": grouped("difficulty"),
        "by_db": grouped("db_id"),
        "status_counts": dict(Counter(r["status"] for r in per_question)),
        "missing": [r["question_id"] for r in per_question if r["status"] == "missing"],
        "format_problems": problems,
        "duplicate_ids": duplicates,
        "unknown_ids": unknown,
        "timing": {
            "median_wall_time_s": round(statistics.median(wall), 2) if wall else None,
            "total_llm_time_s": round(sum(llm), 2),
            "total_tool_time_s": round(sum(tool), 2),
        },
        "per_question": per_question,
    }


def format_summary(report: dict) -> str:
    def pct(block: dict) -> str:
        if block["accuracy"] is None:
            return "not scored (no gold_sql)"
        return f"{block['correct']}/{block['scored']} = {block['accuracy']:.1%}"

    lines = [f"Questions: {report['questions']}, answered: {report['answered']}",
             f"Accuracy: {pct(report)}"]
    lines += [f"  {level:<12} {pct(block)}" for level, block in report["by_difficulty"].items()]
    lines += [f"  {db:<24} {pct(block)}" for db, block in report["by_db"].items()]
    lines.append("Status: " + ", ".join(f"{k}={v}" for k, v in sorted(report["status_counts"].items())))
    timing = report["timing"]
    lines.append(f"Time: median {timing['median_wall_time_s']} s per question, "
                 f"{timing['total_llm_time_s']} s in the model, {timing['total_tool_time_s']} s in tools")
    issues = (report["format_problems"] + [f"duplicate question_id {i}" for i in report["duplicate_ids"]]
              + [f"unknown question_id {i}" for i in report["unknown_ids"]])
    if report["missing"]:
        issues.append(f"{len(report['missing'])} question(s) have no result line: {report['missing'][:10]}")
    lines.append("Format: OK" if not issues else f"Format: {len(issues)} problem(s)")
    lines += [f"  - {p}" for p in issues[:20]]
    if len(issues) > 20:
        lines.append(f"  ... and {len(issues) - 20} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check and score a results.jsonl file.")
    parser.add_argument("--questions", required=True, help="questions JSON file")
    parser.add_argument("--db-dir", required=True, help="folder with <db_id>/<db_id>.sqlite")
    parser.add_argument("--results", required=True, help="your results.jsonl")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per query")
    parser.add_argument("--report", help="also write the full report as JSON to this path")
    parser.add_argument("--strict", action="store_true",
                        help="exit with code 1 if there is any format problem or missing result")
    args = parser.parse_args(argv)

    report = score_file(args.questions, args.db_dir, args.results, args.timeout)
    print(format_summary(report))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = report["format_problems"] or report["duplicate_ids"] or report["unknown_ids"] or report["missing"]
    return 1 if args.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
