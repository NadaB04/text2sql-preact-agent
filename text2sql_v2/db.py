"""Read-only SQLite access and the schema text the model sees.

Read-only is enforced in code, three ways: the file is opened with ``mode=ro``, ``PRAGMA query_only``
is on, and an authorizer denies everything that is not a read. Every query gets a time budget.
"""

from __future__ import annotations

import csv
import difflib
import io
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                 sqlite3.SQLITE_RECURSIVE, sqlite3.SQLITE_PRAGMA}
_READ_PRAGMAS = {"table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info"}


def _authorizer(action, arg1, *_):
    if action not in _READ_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() not in _READ_PRAGMAS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def db_path(db_dir: str | Path, db_id: str) -> Path:
    """CONTRACT.md layout: <db-dir>/<db_id>/<db_id>.sqlite"""
    return Path(db_dir) / db_id / f"{db_id}.sqlite"


def quote(name: str) -> str:
    """Backtick-quote a name that is not a plain identifier."""
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f"`{name}`"


@dataclass
class Result:
    columns: list[str]
    rows: list[tuple]
    total: int  # rows the query returned (len(rows) is capped)

    def table(self, limit: int = 20) -> str:
        """The result as a small text table, as the model sees it."""
        cell = lambda v: (str(v)[:80] + "…") if len(str(v)) > 80 else str(v)  # noqa: E731
        lines = [" | ".join(self.columns)]
        lines += [" | ".join(cell(v) for v in r) for r in self.rows[:limit]]
        more = f", showing {limit}" if self.total > limit else ""
        return "\n".join(lines) + f"\n({self.total} rows{more})"


class Database:
    def __init__(self, path: str | Path, query_timeout_s: float = 30.0):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"database not found: {self.path}")
        self.timeout = query_timeout_s
        self.con = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.con.execute("PRAGMA query_only = ON")
        self.con.set_authorizer(_authorizer)
        self._docs: dict[str, dict[str, str]] | None = None
        self.lock = threading.RLock()  # guards the one connection

    def close(self) -> None:
        self.con.close()

    # ---------- queries
    def query(self, sql: str, max_rows: int = 1000, timeout_s: float | None = None) -> Result:
        """Run one statement. Raises sqlite3.Error (or TimeoutError) on failure."""
        with self.lock:
            return self._query(sql, max_rows, timeout_s)

    def _query(self, sql: str, max_rows: int, timeout_s: float | None) -> Result:
        budget = min(timeout_s or self.timeout, self.timeout)
        deadline = time.monotonic() + max(budget, 0.5)
        self.con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = self.con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows, total = [], 0
            for row in cur:
                total += 1
                if total <= max_rows:
                    rows.append(row)
                if total >= 100_000:
                    break
            return Result(cols, rows, total)
        except sqlite3.OperationalError as exc:
            if time.monotonic() > deadline:
                raise TimeoutError(f"query took longer than {budget:.0f} s") from exc
            raise
        finally:
            self.con.set_progress_handler(None, 0)

    def error_hint(self, error: str) -> str:
        """'did you mean' for an unknown table or column (the most common small-model error)."""
        m = re.search(r"no such (column|table): ([\w.`\"\[\] ]+)", error)
        if not m:
            return ""
        bad = m.group(2).split(".")[-1].strip('`"[] ').lower()
        if m.group(1) == "table":
            names = {t: quote(t) for t in self.tables()}
        else:  # name the table too: the same column name often exists in several tables
            names = {f"{t}.{c}": f"{quote(t)}.{quote(c)}" for t in self.tables() for c in self.column_names(t)}
        short = {k: k.split(".")[-1].lower() for k in names}
        close = set(difflib.get_close_matches(bad, set(short.values()), n=4, cutoff=0.5))
        hits = [names[k] for k in names if short[k] in close][:6]
        return f" Did you mean: {', '.join(hits)}?" if hits else ""

    # ---------- schema
    def tables(self) -> list[str]:
        with self.lock:
            rows = self.con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                    "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def column_names(self, table: str) -> list[str]:
        with self.lock:
            return [r[1] for r in self.con.execute(f'PRAGMA table_info("{table}")')]

    def find_table(self, name: str) -> str | None:
        clean = name.strip().strip('`"[]').lower()
        return next((t for t in self.tables() if t.lower() == clean), None)

    def describe(self, table: str, examples: int = 3) -> str:
        """One table: columns with type, key, meaning (from the kit's CSVs) and example values."""
        with self.lock:
            return self._describe(table, examples)

    def _describe(self, table: str, examples: int) -> str:
        docs = self.docs().get(table.lower(), {})
        lines = [f"TABLE {quote(table)} ({self.con.execute(f'SELECT COUNT(*) FROM {quote(table)}').fetchone()[0]} rows)"]
        for _, name, ctype, _, _, pk in self.con.execute(f'PRAGMA table_info("{table}")'):
            ex = [r[0] for r in self.con.execute(
                f"SELECT DISTINCT {quote(name)} FROM {quote(table)} WHERE {quote(name)} IS NOT NULL LIMIT {examples}")]
            notes = [docs.get(name.lower(), "")] if docs.get(name.lower()) else []
            if ex:
                notes.append("e.g. " + ", ".join(repr(v)[:40] for v in ex))
            lines.append(f"  {quote(name)} {ctype or ''}{' PRIMARY KEY' if pk else ''}"
                         + (f"  -- {'; '.join(notes)}" if notes else ""))
        for r in self.con.execute(f'PRAGMA foreign_key_list("{table}")'):
            lines.append(f"  FOREIGN KEY ({quote(r[3])}) REFERENCES {quote(r[2])}({quote(r[4]) if r[4] else ''})")
        return "\n".join(lines)

    def schema(self, budget_chars: int = 24_000) -> str:
        """All tables. Falls back to names only if the database is too big for the prompt."""
        full = "\n\n".join(self.describe(t) for t in self.tables())
        if len(full) <= budget_chars:
            return full
        names = "\n".join(f"TABLE {quote(t)}: " + ", ".join(quote(c) for c in self.column_names(t))
                          for t in self.tables()) + "\n(Large database: use describe_table for details.)"
        if len(names) <= budget_chars:
            return names
        return ("TABLES: " + ", ".join(quote(t) for t in self.tables())
                + "\n(Very large database: use describe_table to see a table's columns.)")

    def docs(self) -> dict[str, dict[str, str]]:
        """{table: {column: 'meaning; values: ...'}} from database_description/*.csv (may be empty)."""
        if self._docs is None:
            self._docs = _load_docs(self.path.parent / "database_description")
        return self._docs


def _squash(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("commonsense evidence:", "")).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _load_docs(folder: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for f in sorted(folder.glob("*.csv")) if folder.is_dir() else []:
        raw = f.read_bytes()
        text = next((raw.decode(e) for e in ("utf-8-sig", "cp1252", "latin-1") if _decodes(raw, e)), "")
        cols = {}
        for row in csv.DictReader(io.StringIO(text)):
            row = {(k or "").strip().lower(): v or "" for k, v in row.items()}
            name = row.get("original_column_name", "").strip()
            desc, values = _squash(row.get("column_description", "")), _squash(row.get("value_description", ""))
            parts = [desc] if desc and desc.lower() != name.lower() else []
            if values and values.lower() not in ("not useful", "useless"):
                parts.append("values: " + values)
            if name and parts:
                cols[name.lower()] = "; ".join(parts)
        out[f.stem.lower()] = cols
    return out


def _decodes(raw: bytes, enc: str) -> bool:
    try:
        raw.decode(enc)
        return True
    except UnicodeDecodeError:
        return False
