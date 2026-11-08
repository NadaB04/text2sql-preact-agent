"""The agent's five tools, their argument schemas, and the Toolbox that runs them on one database.

    list_tables     tables and row counts
    describe_table  columns, types, meanings (from the kit's CSVs), example values, foreign keys
    column_values   distinct values of a column (exact spelling), optionally filtered by text
    run_query       run a read-only SELECT, see up to 20 rows and the total row count
    submit          the final SQL (must run); ends the task

Reasoning is part of the tool schema. Every tool has a required, non-empty ``thought`` argument, and
run_query / column_values also need a ``prediction`` of the result (PreAct). The arguments are pydantic
models: a call without them fails validation and the error goes back to the model as the observation,
so no tool runs without a thought. The tool then shows "Your prediction: ... / Actual result: ...".
"""

from __future__ import annotations

import sqlite3
from pydantic import BaseModel, Field, ValidationError

from .db import Database, Result

THOUGHT = Field(min_length=3, description="Your reasoning for this step: what you know, what is unclear, why this action.")
PREDICTION = Field(min_length=3, description="What you expect the result to be: number of rows, columns, rough values.")


# ---------- tool arguments (validated before the tool runs)
class ListTablesArgs(BaseModel):
    thought: str = THOUGHT


class DescribeTableArgs(BaseModel):
    thought: str = THOUGHT
    table: str = Field(description="Table name.")


class ColumnValuesArgs(BaseModel):
    thought: str = THOUGHT
    table: str
    column: str
    contains: str = Field("", description="Optional text to search for (case-insensitive). Empty: most common values.")
    prediction: str = PREDICTION


class RunQueryArgs(BaseModel):
    thought: str = THOUGHT
    sql: str = Field(min_length=6, description="One read-only SQLite SELECT.")
    prediction: str = PREDICTION


class SubmitArgs(BaseModel):
    thought: str = THOUGHT
    sql: str = Field(min_length=6, description="The final SQLite query.")


DESCRIPTIONS = {
    "list_tables": "List the tables and their row counts.",
    "describe_table": "Show a table's columns, types, meanings, example values and foreign keys.",
    "column_values": "Show distinct values of a column, optionally only those containing some text. Use it to get the exact spelling of a value.",
    "run_query": "Run a read-only SELECT and see up to 20 rows plus the total row count.",
    "submit": "Submit the final SQL. It must run. This ends the task.",
}

ARGS: dict[str, type[BaseModel]] = {
    "list_tables": ListTablesArgs,
    "describe_table": DescribeTableArgs,
    "column_values": ColumnValuesArgs,
    "run_query": RunQueryArgs,
    "submit": SubmitArgs,
}

SUBMITTED = "Submitted."  # the observation that ends the agent loop


def tool_spec(name: str, args: type[BaseModel], description: str) -> dict:
    """The OpenAI tool format, so the tool name is exactly ``name``."""
    schema = args.model_json_schema()
    schema.pop("title", None)
    for p in schema.get("properties", {}).values():
        p.pop("title", None)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


class Toolbox:
    """Runs the tools against one read-only database and remembers the SQL that worked.

    ``final_sql`` is set by a successful submit; ``last_ok_sql`` is the last query that ran, used as a
    best-effort answer when the agent hits the step or time limit before submitting.
    """

    def __init__(self, db: Database, session):
        self.db, self.session = db, session
        self.final_sql: str | None = None
        self.last_ok_sql: str | None = None
        self._seen: dict[str, str] = {}       # sql -> its observation (repeat guard)
        self._warned_empty: set[str] = set()  # queries already warned about returning 0 rows

    def specs(self, names: list[str]) -> list[dict]:
        return [tool_spec(n, ARGS[n], DESCRIPTIONS[n]) for n in names]

    def call(self, name: str, args: dict) -> str:
        """Validate the arguments (thought / prediction are required), run the tool, log it."""
        def run() -> str:
            if name not in ARGS:
                return f"ERROR: unknown tool {name!r}."
            try:
                a = ARGS[name](**args)
            except ValidationError as exc:
                missing = ", ".join(str(e["loc"][0]) for e in exc.errors())
                return (f"ERROR: invalid arguments ({missing}). Every call needs a non-empty thought"
                        " and, for run_query and column_values, a prediction. Call again.")
            return getattr(self, "_" + name)(a)
        return self.session.tool(name, args, run)

    # ---------- the tools
    def _list_tables(self, a) -> str:
        with self.db.lock:
            return "\n".join(f"{t} ({self.db.con.execute(f'SELECT COUNT(*) FROM `{t}`').fetchone()[0]} rows)"
                             for t in self.db.tables())

    def _describe_table(self, a) -> str:
        t = self.db.find_table(a.table)
        return self.db.describe(t) if t else f"ERROR: no table {a.table!r}. Tables: {', '.join(self.db.tables())}"

    def _column_values(self, a) -> str:
        t = self.db.find_table(a.table)
        if not t:
            return f"ERROR: no table {a.table!r}."
        col = next((c for c in self.db.column_names(t) if c.lower() == a.column.strip('`"').lower()), None)
        if not col:
            return f"ERROR: no column {a.column!r} in {t}. Columns: {', '.join(self.db.column_names(t))}"
        where = f"WHERE `{col}` LIKE ? " if a.contains else ""
        sql = f"SELECT `{col}`, COUNT(*) AS n FROM `{t}` {where}GROUP BY `{col}` ORDER BY n DESC LIMIT 25"
        with self.db.lock:
            rows = self.db.con.execute(sql, (f"%{a.contains}%",) if a.contains else ()).fetchall()
        return self._with_prediction(a, Result([col, "n"], rows, len(rows)).table(25))

    def _run_query(self, a) -> str:
        sql = a.sql.strip().rstrip(";")
        if sql in self._seen:  # repeat guard: the same query again teaches nothing new
            return ("You already ran exactly this query. Its result was:\n" + self._seen[sql]
                    + "\nChange the query, or submit it if it is right.")
        try:
            text = self.db.query(sql, timeout_s=self._query_budget()).table()
            self.last_ok_sql = sql
        except (sqlite3.Error, TimeoutError) as exc:
            text = f"ERROR: {exc}.{self.db.error_hint(str(exc))}"
        self._seen[sql] = text
        return self._with_prediction(a, text)

    def _submit(self, a) -> str:
        sql = a.sql.strip().rstrip(";")
        try:
            res = self.db.query(sql, timeout_s=self._query_budget())
        except (sqlite3.Error, TimeoutError) as exc:
            return f"NOT submitted, the query fails: {exc}.{self.db.error_hint(str(exc))} Fix it and submit again."
        if res.total == 0 and sql not in self._warned_empty:  # asks once; the same query again is accepted
            self._warned_empty.add(sql)
            return "NOT submitted: the query returns 0 rows, which usually means a wrong value or join. " \
                   "Check with column_values; if you are sure, submit the same query again."
        self.final_sql = self.last_ok_sql = sql
        return SUBMITTED

    # ---------- helpers
    def _query_budget(self) -> float:
        return min(30, max(self.session.left() - 5, 1))

    def _with_prediction(self, a, text: str) -> str:
        if getattr(a, "prediction", None):
            return f"Your prediction: {a.prediction}\nActual result:\n{text}"
        return text
