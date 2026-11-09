# text2sql_v2: a PreAct text-to-SQL agent

A question in plain English about a SQLite database goes in; the agent explores the database with
tools, writes and tests SQL, submits it, and answers in one or two plain sentences. The agent is
**PreAct** (ReAct with a prediction before every query) built with **LangGraph**, served locally by
**Ollama** (`qwen3.5:4b-q8_0`) or **vLLM**, through the OpenAI-compatible API.

The package is **`text2sql_v2`**.

- [Setup and run](#setup-and-run)
- [How the agent works](#how-the-agent-works)
- [Logs](#logs)
- [Results](#results)
- [Quantization: Q8_0 vs Q4_K_M](#quantization-q8_0-vs-q4_k_m)
- [Running many questions at once](#running-many-questions-at-once)
- [Bonus 1: vLLM](#bonus-1-vllm) · [Bonus 2: telemetry](#bonus-2-live-telemetry)
- [What didn't work](#what-didnt-work) · [AI tools used](#ai-tools-used) · [Repository map](#repository-map)

## Setup and run

```bash
pip install -r requirements.txt                   # Python 3.10+ (tested with 3.12)
# Ollama, with a context window large enough for the schema + the agent's history:
OLLAMA_CONTEXT_LENGTH=16384 ollama serve          # PowerShell: $env:OLLAMA_CONTEXT_LENGTH=16384; ollama serve
ollama pull qwen3.5:4b-q8_0                       # main setup (Q8_0); qwen3.5:4b = Q4_K_M for the comparison
ollama show qwen3.5:4b-q8_0                       # check: quantization Q8_0, capabilities include tools

python -m text2sql_v2 \
  --questions ../sql_agent_kit/data/questions_dev.json \
  --db-dir ../sql_agent_kit/data/databases \
  --base-url http://localhost:11434/v1 \
  --model qwen3.5:4b-q8_0 \
  --out runs/results.jsonl
```

The kit (`sql_agent_kit/`, with the databases and the dev questions) is expected next to this folder.
This is the command from `CONTRACT.md`, run from the project root. It writes `results.jsonl`, one log
per question in `runs/results_logs/`, and, when the questions carry `gold_sql`, scores the run with the
kit's `scoring.py` (an unmodified copy is `text2sql_v2/scoring.py`). Each line of `results.jsonl` is
written as soon as its question finishes, so a stopped run keeps the questions it finished. Without `gold_sql` it answers every
question and skips scoring. Switching to vLLM changes only `--base-url` and `--model`
(e.g. `--base-url http://localhost:8000/v1 --model qwen3.5-4b`).

| Optional flag | Default | Meaning |
|---|---|---|
| `--max-steps` | **20** | the step limit: model calls per question (the answer step is not counted); reaching it records `max_steps` |
| `--time-limit` | 300 | seconds per question for the whole turn (question -> SQL -> answer); records `timeout` |
| `--call-timeout` | 120 | seconds for one model call, so one stuck generation cannot eat the whole turn |
| `--concurrency` | 1 | questions answered at the same time |
| `--telemetry` | off | sample CPU, RAM, GPU and the server's `/metrics` every second (Bonus 2) |
| `--no-answer` | off | skip the plain-English answer |
| `--think` | off | the model's built-in reasoning (slower) |
| `--ids`, `--limit` | | run a subset (scored against just those questions) |

## How the agent works

One question = one full turn, all inside the 300-second limit (`runner.answer_question`):

```
open the database read-only -> system prompt: rules + schema -> PreAct loop -> final SQL
    -> answer step: the result rows in one or two sentences -> results.jsonl line + q<id>.json log
```

**The PreAct loop** (`agent.py`), one LangGraph graph:

```
START -> model --tool calls--> tools --submitted--> END
           ^ \--no tool call--> nudge   |
           +---------------------------+
```

Every step is one model call. The model answers with tool calls; the tools run; their results go back
as observations; repeat until `submit` succeeds. ReAct (Yao et al., 2022) is Thought -> Action ->
Observation; PreAct (Fu et al., 2024) adds a prediction of each query's result, and the observation
shows `Your prediction: ... / Actual result: ...`, so a surprise (0 rows when one was expected, two
columns instead of one) is in front of the model before it submits.

**Five tools** (`tools.py`):

| Tool | What it returns |
|---|---|
| `list_tables` | tables and row counts |
| `describe_table` | columns, types, meanings from the kit's `database_description/*.csv`, 3 example values, foreign keys |
| `column_values` | the most common distinct values of a column, optionally only those containing some text (exact spelling of a value) |
| `run_query` | up to 20 rows of a read-only SELECT and the total row count |
| `submit` | the final SQL; it must run. A query returning 0 rows is refused once with a warning |

**Thought and prediction are enforced, not requested.** Every tool's arguments are a pydantic model
with a required, non-empty `thought`; `run_query` and `column_values` also require a `prediction`. A
call without them fails validation and the error is the observation ("invalid arguments (prediction)
... Call again"), so no tool ever runs without a thought. With native tool calling a model can
otherwise return a bare call with no reasoning: my first version only asked for "Thought:" lines in the
prompt and got them in about 12% of calls.

**One thought per step.** The small model often also writes free text in the message next to its tool
call: a second, unrequested thought. No API setting can force that text to be empty
(`tool_choice="required"` forces a call, not silence), so the prompt says the `thought` argument is
the one place to reason, and any free text next to a call is removed from the history (it stays in
the log as the call's `content` plus a `discarded_text` event). The model always reads a strict
Thought -> Action -> Observation trajectory. On the held-out run the model still wrote such text in 40%
of turns (77% before the prompt change), which costs output tokens but never reaches the history.

**The prompt** (`prompts.py`). The system message is static: five SQL rules (use the evidence's
columns and formulas, select exactly the asked columns, backtick-quote odd names, SQLite date
functions, no rounding) plus the full schema with column meanings and example values. It is identical
for every question on a database, so the server's prefix cache can reuse it. The user message holds
the PreAct task, the question and its evidence. No few-shot examples: in my first version, full
example trajectories made the agent explore less (fewer `column_values` calls) and lowered held-out
accuracy from 109 to 90 of 176 (p = 0.002).

**Guards** (all in `tools.py` / `agent.py`): a repeated identical query returns its earlier result with a
note; an unknown table or column gets a table-qualified "did you mean" (`frpm.\`School Name\``); a reply
with no tool call gets a nudge; a tool call the server cannot parse (Ollama answers HTTP 500) or a call
that hits `--call-timeout` is reported to the model, which tries again; if the server cannot be reached,
the call is retried once after 3 s.

**Limits and safety.**
- **Read-only in code**, three ways (`db.py`): the file is opened with `mode=ro`, `PRAGMA query_only=ON`,
  and an SQLite authorizer denies anything that is not a read (tested with DELETE, DROP, ATTACH, PRAGMA).
  Every query also has a time budget.
- **`gold_sql` never reaches the agent**: the runner passes only the question and evidence (checked
  on every message the model receives).
- **300 s per question** for the whole turn: each model call gets at most the time left; with less
  than 5 s left the question stops with `timeout`. **Step limit 20** model calls: `max_steps`. Either way
  the last query that ran is kept as the best-effort answer (`fallback_sql: true` in the result).
- **One failing question never stops the run**: it records `error` and the run continues.
- **Switching servers**: thinking is turned off for Ollama (`reasoning_effort: "none"`) and vLLM
  (`chat_template_kwargs.enable_thinking: false`) in the same request; if a server rejects one of these
  keys (HTTP 400 naming it), the request is resent without it. No flag or code change is needed.

## Logs

`<out>_logs/q<id>.json` holds the results line plus the question, evidence, the system prompt (once),
and every event in order with wall-clock start and duration:

| Event | Fields |
|---|---|
| `llm` | step, the input messages, the reply's text and tool calls (with thought and prediction), prompt and completion tokens, seconds |
| `tool` | name, arguments, output (first 2,000 characters), seconds |
| `discarded_text`, `nudge`, `llm_error`, `server_rejected_key`, `connection_retry`, `answer_failed` | what happened and when |

Per question the results line has `steps`, `tool_calls`, `prompt_tokens`, `completion_tokens`,
`llm_time_s` (waiting for the model), `tool_time_s` (running tools), `wall_time_s` and `status`
(`ok`, `error`, `timeout`, `max_steps`), plus `answer`, `fallback_sql` and `error`. `run_config.json` records
the settings, `score_report.json` the kit scorer's report, and `telemetry.jsonl` the hardware samples.
Each run folder also has a `summary_*.json`: accuracy, cost, and checks counted from the logs that the
mechanism happened (thoughts, predictions, dropped free text, repeat-guard hits). Runs are compared with
an exact McNemar test on the per-question correctness in `score_report.json`.

## Results

Final runs of this code (Kaggle `notebook85bc7ddd8c`, `kaggle/final_ollama_quant.ipynb`, logs in
`runs/final_ollama/`): Ollama on a T4, one question at a time, thinking off, temperature 0, step limit
20, 300 s per full turn including the answer. Scored with the kit's `scoring.py` (`Format: OK`).

| Questions | Model | Correct | Status | Median s / full turn | Slowest | Model calls / q |
|---|---|---|---|---|---|---|
| 20 dev (kit) | `qwen3.5:4b-q8_0` | **17/20 = 85%** | ok 20 | 30.0 | 125 s | 5.5 |
| 20 dev (kit) | `qwen3.5:4b` (Q4_K_M) | 15/20 = 75% | ok 19, max_steps 1 | 23.0 | 107 s | 6.45 |
| 100 held-out (BIRD mini-dev, same DBs) | `qwen3.5:4b-q8_0` | **64/100** | ok 95, max_steps 5 | 78.0 * | 233 s * | 7.48 |

\* 4 questions at a time on one GPU, so per-question times include waiting for the other three. The
held-out run (`runs/heldout100/`) used an earlier revision of this code with the same prompts, tools and
limits. Its 100 questions are a seeded sample (stratified by database, seed 2026) of the 176 BIRD mini-dev
questions on the same four databases, none shared with dev. I also looked at these 100 questions while
choosing between agent variants, so 64% is slightly optimistic; the other 76 were never run.
On the held-out run 801/801 tool calls carried a thought and 486/488 queries a prediction.

Where the time goes: in the Q8_0 dev run the agent waited 698.8 s for the model and 0.47 s for tools,
in total. Every answer is produced by the model; the database is never the bottleneck.

**Mechanism checks** (counted from the logs, `runs/final_ollama/summary_dev.json`, Q8_0 dev run): 118/118 tool calls carried a thought,
67/67 queries a prediction, 0 calls rejected; free text next to a call appeared in 52 of 110 turns and
was dropped from the history every time.

**The three dev misses** (Q8_0), all typical small-model errors:
- q46 "Which state special schools have the highest number of enrollees": the agent returned all three
  schools with their enrollment; the gold answer is the top school's name only (extra column, no `LIMIT 1`).
- q989 2008 Canadian Grand Prix winner's finish time: it took `races.time` (the start time) instead of
  `results.time`. The answer step faithfully reported the wrong value ("17:00:00").
- q1352 members of the "Business" major: `LIKE '%Business%'` instead of `= 'Business'`, which also matches
  other majors containing the word.

## Quantization: Q8_0 vs Q4_K_M

The PreAct agent on the 20 dev questions, once per quantization, each on its own T4 with one
question at a time (so the times are not inflated by queueing). Speed and memory also from
`scripts/bench_native.py`: it replays the first prompt of 10 logged questions (system prompt with the
schema + task + question, about 4,050 tokens) through **Ollama's native `/api/chat`**, which reports
`prompt_eval_duration` (reading the prompt) and `eval_duration` (generating). A random nonce at the start
of the prompt defeats the prefix cache for the "cold" call; the same prompt is then sent again ("warm").
Table: `scripts/quant_report.py` -> `runs/final_ollama/quant_table.md`.

| | Q8_0 (~8.5 bits/weight) | Q4_K_M (~4.5 bits/weight) |
|---|---|---|
| Correct (20 dev) | **17/20** | 15/20 (only-Q8 right 3, only-Q4 right 1; McNemar p = 0.63) |
| Status | ok 20 | ok 19, max_steps 1 |
| Model calls / question | 5.50 | 6.45 |
| Prompt / output tokens per question | 32,918 / 1,005 | 45,071 / 1,189 |
| Median s per full turn | 30.0 | 23.0 |
| **Prefill** (`prompt_eval`), tok/s | 1,553 | 1,459 |
| **Decode** (`eval`), tok/s | 32.8 | **42.8** (1.30x) |
| Prompt read, cold / warm (prefix cached), s | 2.61 / 0.09 | 2.77 / 0.09 |
| Load time (after unload), s | 4.6 | 4.0 |
| **Memory**: weights + KV cache for 16k context x 4 slots (`/api/ps`), all on the GPU | 7.25 GB | **5.61 GB** |
| GPU memory in use (`nvidia-smi`, that GPU) | 7.52 GB | 6.0 GB |

**What caused the differences, as I read them:**
- **Decode is 1.3x faster with Q4_K_M.** Generating a token reads every weight once, so decoding is
  limited by memory bandwidth, and Q4_K_M has about half the bytes. It is not the ideal 1.9x
  (8.5/4.5 bits) because each token also reads the KV cache and activations, which do not shrink, and the
  4-bit blocks must be dequantized, which costs compute.
- **Prefill is not faster** (1,459 vs 1,553 tok/s). Reading the prompt processes thousands of tokens in
  one batch, so it is limited by compute, not by bytes; the dequantization even makes Q4_K_M slightly
  slower. Prefix caching matters far more than quantization here: reusing the cached schema prefix cut
  the prompt read from 2.6 s to 0.09 s on both.
- **Memory: 1.6 GB less** with Q4_K_M (download 5.2 vs 3.4 GB; the rest is the same KV cache, compute
  buffers and the unused vision tower). Both fit easily on a 15 GB T4; on my 16 GB laptop, Q8_0 did not
  leave enough RAM to finish a run in version 1.
- **Accuracy and the real speedup.** Q4_K_M lost 2 questions here (not significant on 20), and it needed
  more steps: 17% more model calls and 37% more prompt tokens per question. So a 30% faster decode became
  only a 23% shorter turn. The 20-question difference is noise-level, but my first version measured the
  same model on 176 held-out questions: Q4_K_M lost 28 questions (109 vs 81 correct, p = 0.0001). A 4B
  model loses real quality at 4 bits, which is why Q8_0 stays the main setup.

## Running many questions at once

**How it is done:** `--concurrency N` answers N questions in parallel threads
(`ThreadPoolExecutor` in `runner.main`). Each question has its own `Session` (budget, log), its own
read-only database connection and its own `Toolbox`, so nothing is shared but the model server. The
agent is I/O-bound on the client side: 99.9% of a question's time is waiting for the model (tools took
0.5 s of 700 s in the Q8_0 run), so threads are enough; no processes are needed.

**What limits the speedup on my machine** (and on any single GPU):
1. **The server must batch.** With one request at a time the GPU decodes one token per pass over the
   weights. Running several requests together reuses each pass for all of them, which is why vLLM
   (continuous batching) speeds up with more agents. Ollama batches only up to `OLLAMA_NUM_PARALLEL`
   requests per loaded model, and every parallel slot reserves its own share of the context window.
2. **Memory.** Each concurrent request needs KV cache for its whole history (here 30-45k prompt tokens
   per question over its steps, up to ~10k per call). On my 16 GB laptop, Q8_0 plus Chrome left too
   little RAM to finish a 20-question run one question at a time (version 1), so there is no room for more slots.
3. **Each call gets slower.** Batched requests share the GPU, so a single step takes longer as N grows
   (vLLM median call 9.7 s at 1 agent, 21.5 s at 8). The 300 s limit is per question, so questions with
   many steps start to time out (one at 4 and at 8 agents below).
4. **The tail.** With 20 questions, the run ends when the slowest question ends; in the 8-agent timeline
   the last 150 s run only 1-2 requests while one long question finishes.
5. **Two GPUs = two servers.** A 4B model fits on one T4, so the Kaggle notebooks start one Ollama server
   per GPU and split the work between them (data parallel), which doubles throughput.

Measured with vLLM on one T4 (the 20 dev questions, below): 1 -> 2 -> 4 -> 8 agents gave 1.8x, 2.3x and
2.9x the throughput, with the same 17/20 correct.

## Bonus 1: vLLM

Same agent and code, only `--base-url http://localhost:8000/v1 --model qwen3.5-4b` changed (Kaggle
`notebook023c437457`, `kaggle/final_vllm_concurrency.ipynb`, logs in `runs/final_vllm/`). vLLM serves the
Hugging Face weights `Qwen/Qwen3.5-4B` in **float16** (the T4 has no bfloat16), with
`--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching`,
16k context, at most 8 sequences, CUDA graphs captured only for batch sizes 1/2/4/8 (my first vLLM run in
version 1 ran out of memory capturing 51 sizes). Startup took 602 s on Kaggle (install + load + graph
capture). The server accepted both thinking-off keys, so the automatic resend was not needed.

| One agent, 20 dev questions, T4 | Ollama `qwen3.5:4b-q8_0` | vLLM `Qwen3.5-4B` fp16 |
|---|---|---|
| Correct | 17/20 | 17/20 (ok 19, max_steps 1) |
| Misses | q46, q989, q1352 | q46, q989, q1323 |
| Median s per full turn | **30.0** | 57.0 |
| Wall time, 20 questions | **701 s** | 1,369 s |
| Model calls / question | 5.5 | 5.7 |
| GPU memory in use | 7.5 GB | 12.1 GB |

**Why the differences:**
- **Same accuracy, different misses.** The weights differ (8-bit GGUF vs 16-bit) and the two servers use
  different chat templates and tool-call parsers, so trajectories diverge on hard questions; on 20
  questions the totals are the same.
- **vLLM is about 2x slower for a single agent.** One agent decodes one token at a time, which is
  memory-bandwidth bound, and the fp16 weights are twice the bytes of Q8_0 (about 8 GB vs 4.5 GB per
  token pass). The ratio matches: 57 vs 30 s per question.
- **vLLM uses more memory by design**: it preallocates `--gpu-memory-utilization 0.88` of the GPU for
  weights + KV cache, where Ollama allocates what the loaded model needs.
- **vLLM wins with several agents** (next section): at 8 agents it finished the 20 questions in 476 s,
  faster than one Ollama Q8_0 agent (701 s), because continuous batching serves all of them in each pass.

## Bonus 2: live telemetry

`--telemetry` starts a background sampler (`telemetry.py`) that writes one line per second to
`<out>_logs/telemetry.jsonl`: CPU %, RAM, per-GPU utilization and memory (`nvidia-smi`), and vLLM's
Prometheus `/metrics` (running and waiting requests, KV-cache use, prefix-cache hits, preemptions, token
counters). It uses `time.time()`, the same clock as the start of every model and tool call in the
per-question logs, so `scripts/plot_timeline.py` puts all of them on one time axis
(`runs/final_vllm/figures/`, `runs/final_ollama/figures/`).

| vLLM, one T4, 20 dev questions | 1 agent | 2 agents | 4 agents | 8 agents |
|---|---|---|---|---|
| Wall time | 1,369 s | 766 s | 590 s | **476 s** |
| Speedup | 1.0x | 1.8x | 2.3x | **2.9x** |
| Generated tokens / s (all agents) | 14.8 | 26.1 | 34.0 | 40.1 |
| Prompt tokens / s (all agents) | 547 | 1,004 | 1,278 | 1,431 |
| Median / p90 model call | 9.7 / 16.6 s | 10.2 / 18.5 s | 13.8 / 26.8 s | 21.5 / 47.2 s |
| vLLM running requests, mean (max) | 1.0 (1) | 1.9 (2) | 3.4 (4) | 5.9 (8) |
| KV-cache use, mean (max) | 9% (19%) | 16% (26%) | 24% (41%) | 36% (60%) |
| Prefix-cache hit rate | 87% | 88% | 88% | 87% |
| Preemptions | 0 | 0 | 0 | 0 |
| Correct / not ok | 17/20 / max_steps 1 | 17/20 / max_steps 1 | 17/20 / max_steps 1, timeout 1 | 17/20 / timeout 1 |

**What the picture shows:**
- **Throughput grows, but less than linearly.** 8 agents give 2.9x, not 8x. The KV cache never passes
  60% and nothing is preempted, so memory is not the limit at this scale. Each decode pass now serves
  several sequences, and the prompt (prefill) work of every new step competes with them, so each call
  gets longer: median 9.7 s at 1 agent, 21.5 s at 8.
- **The tail limits the speedup.** In the 8-agent timeline vLLM runs 7-8 requests for the first ~330 s,
  then only 1-2 while the slowest questions finish (the last one started at 220 s and ended at 476 s).
- **Latency becomes the risk.** Slower calls push long questions toward the 300 s limit: q32 timed out
  at 4 and 8 agents (298 s) after finishing in time at 1 and 2. For this workload 4-8 agents on one T4
  is the useful range; past that, the 300 s limit (and the step count of hard questions) costs answers.
- **Prefix caching works.** 87-88% of prompt tokens were served from the cache: every step resends the
  same system prompt (rules + schema) and the growing history, so only the new tokens are computed.
- **The CPU is idle and tools are invisible.** CPU stayed mostly under 30% while 8 agents ran (a spike to 67% at the start), and tool calls
  are thin orange slivers: the agent's own code and SQLite cost nothing next to the model.
- `nvidia-smi` showed 100% GPU "utilization" even with one agent. That metric is the share of time any
  kernel is running, not how much of the GPU is used, so it cannot show the batching gain; vLLM's own
  running-requests and token counters can.

## What didn't work

- **Other agent designs.** PreAct was chosen from six designs built on the same model, tools and prompt
  rules. Four are workflows, where code or one model decision fixes the path: **single** (one call writes
  the SQL, code runs it), **chain** (analyse the question, check the names in code, write, check that the
  SQL runs, repair if not), **route** (the model classifies the question once: simple goes to the chain,
  complex to the agent) and **evaluator** (one call writes the SQL, a second call judges it and asks for
  revisions). Two are agents, where the model decides each next step: **plan-and-execute** (write a plan,
  run its steps with tools, replan) and **PreAct**. On the 20 dev questions:

  | Design | single | chain | route | evaluator | PreAct | plan-and-execute |
  |---|---|---|---|---|---|---|
  | Correct (20 dev) | 11 | 14 | 16 | **18** | 16 | 14 |
  | Model calls / question | 1 | 2.4 | 4.6 | 3.7 | 6.6 | 6.8 |

  With 20 questions almost none of these gaps is significant (the same design moved by one or two
  questions between identical runs), and the agents spent six to seven calls per question without being
  more accurate than the evaluator. The top three then ran on the 100 held-out questions: PreAct 64,
  route 61, evaluator 60, a tie. The difference was behaviour: PreAct looked at real values in the
  database on 47 of the 100 questions, route on 19 and the evaluator on none, which is the grounding the
  BIRD paper says models lack. I kept PreAct because the assignment asks for a tool-using agent and its
  reasoning is visible and checkable in every log; the evaluator was better at avoiding extra columns.
  The comparison code and its logs are kept outside this repo.
- **Decoding settings.** The final runs are greedy (temperature 0): a question has one right SQL, so the
  most likely token is what we want, and top-p / top-k only matter once you sample. In version 1 (20 dev
  questions, older prompt) sampling at temperature 0.7 gave the same accuracy as greedy (85%), and
  Qwen's thinking mode 90% with about 60% more output tokens: within noise, at a cost the 300 s limit
  cannot carry on a laptop (~7 tok/s). Beam search is not offered by Ollama's OpenAI endpoint, and in an
  agent each step is short and conditioned on tool results, so searching over alternatives pays off at
  the level of whole queries, not tokens.
- **A richer tokenizer.** I looked at adding SQL keywords and schema names as tokens. Two reasons
  stopped it: (1) BIRD's error analysis (Li et al., 2023) puts the failures in schema linking (41.6%),
  database content (40.8%) and evidence (17.6%), not in writing SQL, and my misses agree (an extra
  column, a wrong column, `LIKE` instead of `=`); (2) new tokens need new embedding and output rows,
  i.e. training the model. What needs no training was done instead: a compact schema text and cell
  values cut at 80 characters, so prompts cost fewer tokens.
- **Fine-tuning (PEFT / LoRA).** I considered a LoRA adapter trained on BIRD's train split. The model is
  served as a quantized GGUF, while the adapter would be trained on the full-precision weights (or in
  QLoRA's own 4-bit format) and then merged and re-quantized or converted, a pipeline with its own
  quality risk and GPU time I did not have before the deadline. The error analysis above also says the
  gap is knowing the schema and the data, which the agent's tools address at run time.
- **Asking for reasoning in the prompt** (version 1): "Thought:" lines appeared in about 12% of model
  calls. Making `thought` a required tool argument fixed it and using pyDantic (100% in every run since).
- **Few-shot example trajectories** (version 1): the demos taught the model to explore less, and held-out
  accuracy fell from 109 to 90 of 176 (p = 0.002). This version uses none. The article
- **An empty `sql` argument** (5 of 100 held-out questions): the call arrived with a thought and a
  prediction but no SQL, most likely lost by Ollama's parser for Qwen's XML-style tool calls. In one
  question the model repeated the same broken call 18 times until the step limit. Next fix: after two
  identical invalid calls, say which argument arrived empty.
- **First held-out attempt sat in Kaggle's GPU queue** for 30 minutes; the P100 option is gone, so all
  runs use T4 x2 and wait when it is busy.
- **Telemetry gap**: host RAM in the Kaggle container rose from 4 to 22 GB during the Ollama runs and then
  stayed flat, but the sampler's process filter caught only the 0.09 GB `ollama serve` process, not the
  model runners, so I cannot say what held that memory.

## AI tools used

Fable 5.1 was used to define the Opus agent which wrote this version with me: the code, the Kaggle notebooks and
the log analysis; I decided the design, read the traces, and chose what to change. Later Fable 5.1 was used to
review and analyze the work of Opus.

## Repository map

```
text2sql_v2/         the agent (read in this order: runner, agent, tools, prompts, llm, db)
  runner.py          CLI (CONTRACT.md), one full turn per question, results.jsonl, scoring
  agent.py           the PreAct LangGraph loop; one thought per step
  tools.py           the five tools; thought / prediction required by schema
  prompts.py         every prompt
  llm.py             Config; Session: model calls, tool calls, time and step budget, the log
  db.py              read-only SQLite, schema text, "did you mean"
  telemetry.py       hardware + vLLM metrics sampler (Bonus 2)
  scoring.py         unmodified copy of the kit's scorer
scripts/             bench_native.py + quant_report.py (quantization), plot_timeline.py (Bonus 2)
kaggle/              the notebooks behind the final Ollama and vLLM runs (each embeds the code)
runs/                logs of every run quoted here (final_ollama, final_vllm, heldout100)
data/                questions_heldout.json (176 BIRD mini-dev), questions_heldout100.json (the sample)
```
