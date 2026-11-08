"""The agent: PreAct, a ReAct loop with predictions, as a LangGraph graph.

ReAct (Yao et al. 2022): Thought -> Action -> Observation, repeated until the model submits.
PreAct (Fu et al. 2024): before each query the model also predicts its result, and the observation
shows the prediction next to the actual result, so a surprise is visible.

    START -> model --tool calls--> tools --submitted--> END
               ^ \\--no tool call--> nudge   |
               +---------------------------+

One step = one model call. Each tool call carries exactly one thought (and, for queries, one
prediction): both are required arguments (see tools.py). If the model also writes free text next to the
call, that text is a second, unrequested thought: it stays in the log but is removed from the history,
so the trajectory the model reads is strictly Thought -> Action -> Observation.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import prompts
from .llm import BadReply, Session
from .tools import SUBMITTED, Toolbox

TOOLS = ["list_tables", "describe_table", "column_values", "run_query", "submit"]


def solve(session: Session, box: Toolbox, system: str, question: str) -> None:
    """Answer one question. The result is left in box.final_sql (or box.last_ok_sql)."""
    messages = [SystemMessage(system), HumanMessage(prompts.PREACT + "\n\n" + question)]
    tool_loop(session, box, messages)


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    done: bool


def tool_loop(session: Session, box: Toolbox, messages: list, label: str = "preact") -> list:
    """Run model -> tools -> model ... until submit succeeds; return the final message list.
    The step and time limits end it from inside session.chat (OutOfSteps / OutOfTime)."""
    specs = box.specs(TOOLS)
    calls = {"bad": 0}  # unreadable replies in a row

    def model(state: LoopState) -> dict:
        try:
            reply = session.chat(state["messages"], tools=specs, label=label)
        except BadReply:  # the server could not parse the tool call: say so, let the model retry
            calls["bad"] += 1
            if calls["bad"] > 2:
                raise
            return {"messages": [HumanMessage("Your last reply could not be used (malformed or too long). "
                                              "Make one tool call with valid JSON arguments and a short thought.")]}
        calls["bad"] = 0
        return {"messages": [one_thought(reply, session, label)]}

    def tools(state: LoopState) -> dict:
        out, done = [], False
        for call in state["messages"][-1].tool_calls:  # every call of the turn, in order
            result = box.call(call["name"], call["args"])
            out.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
            if call["name"] == "submit" and result == SUBMITTED:
                done = True
                break
        return {"messages": out, "done": done}

    def nudge(state: LoopState) -> dict:
        session.note(type="nudge", label=label)
        return {"messages": [HumanMessage(prompts.NUDGE.format(finish="submit"))]}

    def after_model(state: LoopState) -> str:
        last = state["messages"][-1]
        if state.get("done"):
            return END
        if isinstance(last, HumanMessage):  # an unreadable reply was replaced by a retry request
            return "model"
        return "tools" if last.tool_calls else "nudge"

    g = StateGraph(LoopState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_node("nudge", nudge)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", after_model, ["model", "tools", "nudge", END])
    g.add_conditional_edges("tools", lambda s: END if s["done"] else "model", ["model", END])
    g.add_edge("nudge", "model")
    final = g.compile().invoke({"messages": messages, "done": False}, {"recursion_limit": 200})
    return final["messages"]


def one_thought(reply: AIMessage, session: Session, label: str) -> AIMessage:
    """Keep only the tool calls (with their required thought) in the history; free text next to a
    call is dropped there and noted in the log. A reply without tool calls is kept (it gets a nudge)."""
    if reply.tool_calls and str(reply.content).strip():
        session.note(type="discarded_text", label=label, chars=len(str(reply.content).strip()))
        return AIMessage(content="", tool_calls=reply.tool_calls, id=reply.id, usage_metadata=reply.usage_metadata)
    return reply
