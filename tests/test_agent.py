"""Offline tests of the PreAct agent: a scripted fake model replaces the server, so no GPU is needed.

    python -m pytest -q
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from text2sql_v2 import llm
from text2sql_v2.db import Database, db_path
from text2sql_v2.llm import Config
from text2sql_v2.runner import RESULT_FIELDS, answer_question
from text2sql_v2.telemetry import Sampler, metrics_url, parse_prometheus

KIT = Path(os.environ.get("KIT_DIR") or Path(__file__).resolve().parents[2] / "sql_agent_kit") / "data"
DB_DIR = str(KIT / "databases")
if not (KIT / "databases" / "superhero" / "superhero.sqlite").is_file():
    pytest.skip(f"the kit's databases are not at {KIT}; set KIT_DIR to the sql_agent_kit folder",
                allow_module_level=True)
Q = {"question_id": 1, "db_id": "superhero", "question": "How many superheroes are there?",
     "evidence": "", "difficulty": "simple", "gold_sql": "SELECT COUNT(*) FROM superhero"}
GOOD = "SELECT COUNT(*) FROM superhero"
_ids = itertools.count()


def call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"c{next(_ids)}"}


def ai(*calls, content="") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(calls))


def fake(responder):
    """A ChatOpenAI stand-in: responder(messages) -> AIMessage."""
    class Fake:
        def __init__(self, **_):
            pass

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            msg = responder(messages)
            msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return msg
    return Fake


def run(responder, **cfg):
    c = Config(base_url="http://fake/v1", model="fake", answer=False, **cfg)
    return answer_question(c, Q, DB_DIR, client=fake(responder))


def script(*turns):
    it = iter(turns)
    return lambda messages: next(it)


def outputs(log) -> list[str]:
    return [e["output"] for e in log["events"] if e["type"] == "tool"]


# ---------------------------------------------------------------- the loop
def test_a_full_turn_writes_every_contract_field():
    result, log = run(script(ai(call("run_query", thought="count rows", prediction="1 row", sql=GOOD)),
                             ai(call("submit", thought="that is the answer", sql=GOOD))))
    assert result["status"] == "ok" and result["predicted_sql"] == GOOD
    assert result["steps"] == 2 and result["tool_calls"] == 2
    assert set(RESULT_FIELDS) <= set(result)
    assert log["system_prompt"].startswith("You write SQLite queries")


def test_thought_and_prediction_are_required():
    result, log = run(script(
        ai(call("run_query", sql=GOOD)),                                       # no thought, no prediction
        ai(call("run_query", thought="count rows", sql=GOOD)),                 # no prediction
        ai(call("run_query", thought="count rows", prediction="1 row, about 750", sql=GOOD)),
        ai(call("submit", thought="matches the prediction", sql=GOOD))))
    out = outputs(log)
    assert out[0].startswith("ERROR: invalid arguments (thought, prediction)")
    assert out[1].startswith("ERROR: invalid arguments (prediction)")
    assert out[2].startswith("Your prediction: 1 row, about 750\nActual result:\nCOUNT(*)\n750")
    assert result["status"] == "ok"


def test_one_thought_per_step_free_text_is_dropped_from_history():
    seen = []

    def respond(messages):
        seen.append(messages)
        if len(seen) == 1:
            return ai(call("list_tables", thought="see the tables"), content="Let me think out loud first.")
        return ai(call("submit", thought="the count is the answer", sql=GOOD))
    result, log = run(respond)
    replayed = [m for m in seen[1] if m.type == "ai"]
    assert replayed[0].content == "" and replayed[0].tool_calls[0]["args"]["thought"] == "see the tables"
    assert next(e for e in log["events"] if e["type"] == "llm")["content"] == "Let me think out loud first."
    assert any(e["type"] == "discarded_text" for e in log["events"]) and result["status"] == "ok"


def test_several_tool_calls_in_one_turn():
    result, log = run(script(
        ai(call("describe_table", thought="see heroes", table="superhero"),
           call("column_values", thought="publisher names", prediction="names", table="publisher",
                column="publisher_name", contains="Marv")),
        ai(call("submit", thought="done", sql=GOOD))))
    assert [e["name"] for e in log["events"] if e["type"] == "tool"] == ["describe_table", "column_values", "submit"]
    assert "Marvel Comics" in outputs(log)[1] and result["steps"] == 2


def test_repeat_guard_and_empty_result_warning():
    empty = "SELECT superhero_name FROM superhero WHERE superhero_name = 'Nobody'"
    result, log = run(script(
        ai(call("run_query", thought="try it", prediction="one row", sql=GOOD)),
        ai(call("run_query", thought="again", prediction="one row", sql=GOOD)),
        ai(call("submit", thought="sure?", sql=empty)),
        ai(call("submit", thought="fine", sql=GOOD))))
    out = outputs(log)
    assert out[1].startswith("You already ran exactly this query")  # the guard answers before the prediction
    assert out[2].startswith("NOT submitted: the query returns 0 rows")
    assert result["predicted_sql"] == GOOD


def test_no_tool_call_gets_a_nudge():
    result, log = run(script(ai(content="I think the answer is 750."), ai(call("submit", thought="submit it", sql=GOOD))))
    assert any(e["type"] == "nudge" for e in log["events"]) and result["status"] == "ok"


# ---------------------------------------------------------------- limits and failures
def test_step_limit_gives_max_steps_and_keeps_the_last_working_sql():
    result, _ = run(lambda m: ai(call("run_query", thought="loop", prediction="one row", sql="SELECT 1")), max_steps=3)
    assert result["status"] == "max_steps" and result["steps"] == 3
    assert result["predicted_sql"] == "SELECT 1" and result["fallback_sql"]


def test_time_limit_gives_timeout():
    def slow(messages):
        time.sleep(0.2)
        return ai(call("run_query", thought="slow step", prediction="one row", sql=GOOD))
    result, _ = run(slow, time_limit_s=5.3)  # a call needs 5 s left
    assert result["status"] == "timeout" and result["predicted_sql"] == GOOD


def test_unreadable_tool_call_is_retried_with_a_note():
    class ServerError(Exception):
        status_code = 500
    turns = iter(["boom", ai(call("submit", thought="valid this time", sql=GOOD))])

    def respond(messages):
        t = next(turns)
        if t == "boom":
            raise ServerError("Error code: 500 - XML syntax error on line 22: unexpected EOF")
        assert "could not be used" in messages[-1].content
        return t
    result, log = run(respond)
    assert result["status"] == "ok" and result["predicted_sql"] == GOOD
    assert [e["type"] for e in log["events"]][:2] == ["llm_error", "llm"]


def test_server_that_rejects_reasoning_effort_gets_the_call_again_without_it():
    """vLLM may refuse Ollama's reasoning_effort="none"; the call is resent without it, no flag needed."""
    llm._REJECTED_KEYS.clear()
    bodies = []

    class Picky:
        def __init__(self, **kw):
            self.body = kw["extra_body"]
            bodies.append(self.body)

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            if "reasoning_effort" in self.body:
                raise RuntimeError("Error code: 400 - reasoning_effort: Input should be 'low', 'medium' or 'high'")
            msg = ai(call("submit", thought="count rows", sql=GOOD))
            msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return msg
    try:
        result, log = answer_question(Config(base_url="http://fake/v1", model="fake", answer=False), Q, DB_DIR,
                                      client=Picky)
    finally:
        llm._REJECTED_KEYS.clear()
    assert result["status"] == "ok", result["error"]
    assert "reasoning_effort" in bodies[0] and "reasoning_effort" not in bodies[-1]
    assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert any(e["type"] == "server_rejected_key" for e in log["events"])


def test_missing_database_is_an_error_not_a_crash():
    result, _ = answer_question(Config(base_url="x", model="y", answer=False), {**Q, "db_id": "nope"}, DB_DIR,
                                client=fake(lambda m: ai()))
    assert result["status"] == "error" and "database not found" in result["error"]


def test_answer_step_turns_rows_into_a_sentence():
    turns = iter([ai(call("submit", thought="count", sql=GOOD)), AIMessage(content="There are 750 superheroes.")])
    c = Config(base_url="http://fake/v1", model="fake", answer=True)
    result, log = answer_question(c, Q, DB_DIR, client=fake(lambda m: next(turns)))
    assert result["answer"] == "There are 750 superheroes." and result["steps"] == 1  # the answer is not a step


# ---------------------------------------------------------------- safety and prompt layout
def test_database_is_read_only():
    db = Database(db_path(DB_DIR, "superhero"))
    for sql in ("DELETE FROM superhero", "PRAGMA query_only = OFF", "DROP TABLE superhero",
                "ATTACH DATABASE ':memory:' AS x"):
        with pytest.raises(Exception):
            db.query(sql)


def test_system_prompt_is_a_static_prefix_and_gold_sql_never_reaches_the_model():
    seen = []

    def respond(messages):
        seen.append(messages)
        return ai(call("submit", thought="submit it", sql=GOOD))
    run(respond)
    system, user = seen[0][0].content, seen[0][1].content
    assert system.startswith("You write SQLite queries") and "TABLE superhero" in system
    assert Q["question"] in user and Q["question"] not in system  # the question comes after the shared prefix
    assert all(Q["gold_sql"] not in m.content for msgs in seen for m in msgs if isinstance(m.content, str))


def test_did_you_mean_names_the_table():
    db = Database(db_path(DB_DIR, "superhero"))
    with pytest.raises(Exception) as exc:
        db.query("SELECT s.publisher_nam FROM superhero s")
    assert "publisher.publisher_name" in db.error_hint(str(exc.value))


# ---------------------------------------------------------------- telemetry (Bonus 2)
def test_telemetry_parses_vllm_metrics_and_samples(tmp_path):
    text = ('# HELP x\nvllm:num_requests_running{model_name="m"} 3.0\n'
            'vllm:kv_cache_usage_perc{model_name="m"} 0.25\nother_metric 7\n')
    assert parse_prometheus(text) == {"vllm:num_requests_running": 3.0, "vllm:kv_cache_usage_perc": 0.25}
    assert metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"
    pytest.importorskip("psutil")
    s = Sampler(tmp_path / "t.jsonl", metrics="http://127.0.0.1:9/metrics", interval=0.05)
    s.start()
    time.sleep(0.3)
    s.stop()
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows and {"t", "cpu_pct", "ram_used_gb"} <= set(rows[0]) and s.metrics is None


def test_a_dropped_connection_is_retried_once():
    import httpx
    import openai
    state = {"n": 0}

    def respond(messages):
        state["n"] += 1
        if state["n"] == 1:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://fake/v1/chat/completions"))
        return ai(call("submit", thought="count rows", sql=GOOD))
    result, log = run(respond)
    assert result["status"] == "ok" and state["n"] == 2
    assert [e["type"] for e in log["events"]][:2] == ["connection_retry", "llm"]
