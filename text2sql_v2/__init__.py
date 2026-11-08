"""text2sql_v2: a PreAct text-to-SQL agent (LangGraph, any OpenAI-compatible server such as Ollama or vLLM).

Read in this order:
    runner.py     CLI (CONTRACT.md) and one full turn per question
    agent.py      the PreAct loop: model -> tools -> model ... -> submit
    tools.py      the five tools; thought and prediction are required arguments
    prompts.py    every prompt
    llm.py        Config, and the Session that times, counts and logs every call
    db.py         read-only SQLite and the schema text
    telemetry.py  CPU/RAM/GPU and vLLM metrics while a run executes (Bonus 2)
    scoring.py    unmodified copy of the kit's scorer
"""
