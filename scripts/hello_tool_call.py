"""Step 3 of the assignment: one tool call with no framework (kept as the setup smoke test)."""
import json
import sys
import time

from openai import OpenAI

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:4b-q8_0"
client = OpenAI(base_url=BASE_URL, api_key="unused")
NO_THINKING = {"reasoning_effort": "none"}

if MODEL not in [m.id for m in client.models.list()]:
    sys.exit(f"{MODEL} is not on this server. Run: ollama pull {MODEL}")

tools = [{"type": "function", "function": {
    "name": "list_tables", "description": "List the tables in the current SQLite database.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}]
messages = [{"role": "user", "content": "Which tables does this database have?"}]

t = time.perf_counter()
reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools,
                                       extra_body=NO_THINKING).choices[0].message
print(f"[{time.perf_counter() - t:.1f}s] The model asked for:", reply.tool_calls)
if reply.tool_calls:
    call = reply.tool_calls[0]
    messages.append(reply.model_dump(exclude_none=True))
    messages.append({"role": "tool", "tool_call_id": call.id,
                     "content": json.dumps(["superhero", "publisher", "superpower"])})
    t = time.perf_counter()
    final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools,
                                           extra_body=NO_THINKING).choices[0].message
    print(f"[{time.perf_counter() - t:.1f}s] Final answer:", final.content)
