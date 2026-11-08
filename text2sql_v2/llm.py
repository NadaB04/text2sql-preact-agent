"""Settings, the model client, and one question's budget and log.

The agent talks to the model only through ``Session.chat`` and runs tools only through
``Session.tool``. A Session belongs to one question: it knows the time left, enforces the step limit,
counts tokens, times every model and tool call, and keeps the event log that becomes q<id>.json.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI


@dataclass
class Config:
    base_url: str
    model: str
    api_key: str = "unused"        # Ollama and vLLM ignore it, the OpenAI client needs one
    temperature: float = 0.0
    max_tokens: int = 1024
    time_limit_s: float = 300.0    # one full turn: question -> SQL -> answer in words
    call_timeout_s: float = 120.0  # one model call; a stuck generation must not eat the whole turn
    max_steps: int = 20            # model calls per question (the answer step is not counted)
    think: bool = False            # the model's built-in reasoning (slower); off by default
    answer: bool = True            # finish with a plain-English answer

    def extra_body(self) -> dict:
        """Thinking on/off for both servers: Ollama reads reasoning_effort, vLLM reads chat_template_kwargs.
        A server that rejects one of these keys gets the request again without it (see Session._invoke),
        so switching servers needs only --base-url and --model."""
        if self.think:
            body = {"reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True}}
        else:
            body = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        return {k: v for k, v in body.items() if k not in _REJECTED_KEYS}


_REJECTED_KEYS: set[str] = set()  # extra_body keys this server refused (learned once per process)


class OutOfTime(Exception):
    """Less than a few seconds left of the question's time limit."""


class OutOfSteps(Exception):
    """The question used its step limit (max_steps model calls)."""


class BadReply(Exception):
    """No usable reply, but time is left: the server could not parse the model's output (Ollama's
    tool-call parser answers HTTP 500 on a malformed call), or one call hit call_timeout_s. The agent
    tells the model and lets it try again."""


@dataclass
class Session:
    """One question's budget and log."""
    cfg: Config
    deadline: float
    events: list[dict] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_time_s: float = 0.0
    tool_time_s: float = 0.0
    count_steps: bool = True  # False for the answer step, which is not an agent step
    client: Callable[..., Any] | None = None  # tests swap in a scripted model here
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def start(cls, cfg: Config, client: Callable[..., Any] | None = None) -> "Session":
        return cls(cfg, time.monotonic() + cfg.time_limit_s, client=client)

    def left(self) -> float:
        return self.deadline - time.monotonic()

    # ---------- model calls
    def chat(self, messages: list[BaseMessage], tools: list | None = None, label: str = "",
             max_tokens: int | None = None) -> AIMessage:
        """One model call. Raises OutOfTime / OutOfSteps when the budget is spent, BadReply when the
        reply is unusable but there is time to try again. Every call is logged as an event."""
        step = self._reserve_step()
        t0, wall = time.perf_counter(), time.time()
        try:
            reply = self._invoke(messages, tools, max_tokens)
        except Exception as exc:
            self._log_error(exc, label, step, wall, time.perf_counter() - t0)
            translated = self._classify_error(exc)
            if translated is exc:
                raise
            raise translated from exc
        self._log_reply(reply, messages, label, step, wall, time.perf_counter() - t0)
        return reply

    def _reserve_step(self) -> int:
        if self.left() < 5:
            raise OutOfTime()
        with self.lock:
            if self.count_steps and self.steps >= self.cfg.max_steps:
                raise OutOfSteps()
            if self.count_steps:
                self.steps += 1  # reserved before the call, so parallel calls cannot overshoot the limit
            return self.steps

    def _invoke(self, messages, tools, max_tokens) -> AIMessage:
        make = self.client or ChatOpenAI  # a new client per call, so its timeout fits the time left
        for _ in range(3):
            llm = make(base_url=self.cfg.base_url, api_key=self.cfg.api_key, model=self.cfg.model,
                       temperature=self.cfg.temperature,
                       max_tokens=max_tokens or self.cfg.max_tokens,
                       timeout=max(min(self.left() - 2, self.cfg.call_timeout_s), 1),
                       max_retries=0, extra_body=self.cfg.extra_body())
            runnable = llm.bind_tools(tools) if tools else llm
            try:
                return runnable.invoke(messages)
            except Exception as exc:
                key = _rejected_key(exc)
                if key is None:
                    raise
                _REJECTED_KEYS.add(key)  # e.g. vLLM refusing reasoning_effort="none": resend without it
                self.note(type="server_rejected_key", key=key)

    def _classify_error(self, exc: Exception) -> Exception:
        """Which exception the agent sees for a failed call."""
        if self.left() < 10:
            return OutOfTime()
        if "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
            return BadReply(f"the call took longer than {self.cfg.call_timeout_s:.0f} s")
        if getattr(exc, "status_code", None) == 500 or "Error code: 500" in str(exc):
            return BadReply(str(exc)[:300])
        return exc

    def _log_error(self, exc: Exception, label: str, step: int, wall: float, dt: float) -> None:
        with self.lock:
            self.llm_time_s += dt
            self.events.append({"type": "llm_error", "label": label, "step": step, "start": wall,
                                "duration_s": round(dt, 3), "error": f"{type(exc).__name__}: {exc}"[:500]})

    def _log_reply(self, reply: AIMessage, messages, label: str, step: int, wall: float, dt: float) -> None:
        usage = reply.usage_metadata or {}
        with self.lock:
            self.llm_time_s += dt
            self.prompt_tokens += usage.get("input_tokens", 0)
            self.completion_tokens += usage.get("output_tokens", 0)
            self.events.append({
                "type": "llm", "label": label, "step": step, "start": wall, "duration_s": round(dt, 3),
                "prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0),
                "input": [_msg(m) for m in messages[1:]],  # the system prompt is logged once per question
                "content": reply.content if isinstance(reply.content, str) else str(reply.content),
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls],
            })

    # ---------- tool calls and other events
    def tool(self, name: str, args: dict, fn: Callable[[], str]) -> str:
        """Run one tool function, time it and log it. A crash becomes an ERROR observation."""
        t0, wall = time.perf_counter(), time.time()
        try:
            out = fn()
        except Exception as exc:
            out = f"ERROR: {short_error(exc)}"
        dt = time.perf_counter() - t0
        with self.lock:
            self.tool_time_s += dt
            self.tool_calls += 1
            self.events.append({"type": "tool", "name": name, "args": args, "start": wall,
                                "duration_s": round(dt, 4), "output": out[:2000]})
        return out

    def note(self, **event: Any) -> None:
        with self.lock:
            self.events.append({"start": time.time(), **event})


def _rejected_key(exc: Exception) -> str | None:
    """The extra_body key named in an HTTP 400, if the server refused one of ours."""
    if getattr(exc, "status_code", None) != 400 and "Error code: 400" not in str(exc):
        return None
    return next((k for k in ("reasoning_effort", "chat_template_kwargs") if k in str(exc) and k not in _REJECTED_KEYS), None)


def _msg(m: BaseMessage) -> dict:
    d = {"role": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
    if getattr(m, "tool_call_id", None):
        d["tool_call_id"] = m.tool_call_id
    return d


def short_error(exc: Exception) -> str:
    text = str(exc).split("\n For further information")[0]
    return re.sub(r"\s+", " ", text)[:300]
