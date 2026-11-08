"""Build a self-contained Kaggle notebook (GPU T4 x2) that runs the agent with Ollama or vLLM and prints the summary.

The project code is embedded (base64), so the notebook always runs the current code. The kit
(databases + questions) comes from the private Kaggle dataset that holds kit.zip.

    python scripts/make_kaggle_notebook.py --runs preact_q8@qwen3.5:4b-q8_0 preact_q4km@qwen3.5:4b \
        --bench-models qwen3.5:4b-q8_0 qwen3.5:4b --concurrency preact_q8=1 preact_q4km=1 \
        --extra=--telemetry --out kaggle/final_ollama_quant.ipynb

Each run is ``NAME@MODEL`` (``NAME`` alone uses ``--model``). A ``:dev`` suffix runs it on the kit's 20
dev questions even when ``--questions-file`` is given.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("text2sql_v2", "scripts", "tests"):
            for p in (ROOT / sub).rglob("*.py"):
                z.write(p, "text2sql_V2/" + p.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def cell(src: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


SETUP = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, threading, time, urllib.request, zipfile
os.chdir("/kaggle/working")
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall(".")
zips = glob.glob("/kaggle/input/**/kit.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(".")
else:  # Kaggle may auto-extract an uploaded zip
    q = glob.glob("/kaggle/input/**/sql_agent_kit/data/questions_dev.json", recursive=True)[0]
    shutil.copytree(q.rsplit("/data/", 1)[0], "sql_agent_kit", dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langgraph", "langchain-openai", "psutil", "matplotlib"], check=True)
os.chdir("/kaggle/working/text2sql_V2")
print(subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True).stdout[-600:])
'''

OLLAMA = r'''
subprocess.run("apt-get install -y -qq zstd pciutils lshw > /dev/null 2>&1", shell=True)
r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
assert shutil.which("ollama"), "ollama install failed (see the installer output above)"
MODELS = sorted({model for _, _, model in RUNS})
URLS = []
for g in range(GPUS):  # one server per GPU: a 4B model fits on one T4, so this doubles throughput
    port = 11434 + g
    env = {**os.environ, "OLLAMA_CONTEXT_LENGTH": "16384", "OLLAMA_NUM_PARALLEL": str(PARALLEL),
           "OLLAMA_KEEP_ALIVE": "120m", "OLLAMA_HOST": f"127.0.0.1:{port}", "CUDA_VISIBLE_DEVICES": str(g)}
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"/kaggle/working/ollama{g}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}"); break
        except OSError:
            time.sleep(1)
    URLS.append(f"http://127.0.0.1:{port}/v1")
for m in MODELS:  # the servers share ~/.ollama: one pull serves both GPUs
    subprocess.run(["ollama", "pull", m], check=True, capture_output=True, env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"})
print(URLS, subprocess.run(["ollama", "list"], capture_output=True, text=True, env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"}).stdout)
# Warm-up: load each model on each server before any timed question, so the load time is not charged
# to the first questions (POC round 2 lost q32 that way: its first call waited about 146 s).
for url in URLS:
    for m in MODELS:
        body = json.dumps({"model": m, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
                           "reasoning_effort": "none"}).encode()
        req = urllib.request.Request(url + "/chat/completions", body, {"Content-Type": "application/json"})
        t = time.time()
        urllib.request.urlopen(req, timeout=600).read()
        print(f"warm-up {m} on {url}: {time.time() - t:.0f}s", flush=True)
# The assignment's "Getting started" step 3: one tool call with no framework.
print(subprocess.run([sys.executable, "scripts/hello_tool_call.py", URLS[0], MODELS[0]], capture_output=True, text=True).stdout)
'''

VLLM = r'''
# vLLM serves Hugging Face weights (not GGUF). The T4 (compute 7.5) has no bfloat16 -> float16.
# One server on GPU 0. Earlier project: capturing CUDA graphs for 51 batch sizes ran out of memory on a
# T4, so capture only the sizes we use; if that still fails, run eager (no graphs).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM_PIP], check=True)
base = ["vllm", "serve", HF_MODEL, "--served-model-name", SERVED_NAME, "--dtype", "float16",
        "--max-model-len", "16384", "--gpu-memory-utilization", "0.88", "--max-num-seqs", "8",
        "--enable-auto-tool-choice", "--tool-call-parser", TOOL_PARSER, "--reasoning-parser", "qwen3",
        "--language-model-only", "--enable-prefix-caching", "--port", "8000"]
attempts = [base + ["--compilation-config", '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'], base + ["--enforce-eager"]]
VLLM_MODE = None
for k, cmd in enumerate(attempts):
    print(" ".join(cmd), flush=True)
    log = f"/kaggle/working/vllm_attempt{k}.log"
    server = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    t0 = time.time()
    while time.time() - t0 < 1500:
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models"); VLLM_MODE = "cudagraphs" if k == 0 else "eager"
            print(f"vllm up after {time.time() - t0:.0f}s ({VLLM_MODE})", flush=True); break
        except OSError:
            if server.poll() is not None:
                print(open(log).read()[-3000:], flush=True); break
            time.sleep(2)
    if VLLM_MODE:
        break
    server.kill(); time.sleep(10)
assert VLLM_MODE, "vllm did not start"
URLS = ["http://localhost:8000/v1"]
body = json.dumps({"model": SERVED_NAME, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
urllib.request.urlopen(urllib.request.Request(URLS[0] + "/chat/completions", body, {"Content-Type": "application/json"}), timeout=600).read()
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
print(subprocess.run([sys.executable, "scripts/hello_tool_call.py", URLS[0], SERVED_NAME], capture_output=True, text=True).stdout)
'''

RUN = r'''
Q = DEV_Q = "../sql_agent_kit/data/questions_dev.json"; DBS = "../sql_agent_kit/data/databases"
if QUESTIONS_B64:
    Q = "/kaggle/working/questions.json"; open(Q, "wb").write(base64.b64decode(QUESTIONS_B64))
os.makedirs("/kaggle/working/runs", exist_ok=True)

def lane(k):  # runs k, k+n, ... go one after another to server k
    for i in range(k, len(RUNS), 1 if SERIAL else len(URLS)):
        name, model = RUNS[i]
        conc = CONCURRENCY.get(name, PARALLEL)
        q = DEV_Q if name in ON_DEV else Q  # runs marked :dev use the kit's 20 dev questions
        cmd = [sys.executable, "-m", "text2sql_v2", "--questions", q, "--db-dir", DBS, "--base-url", URLS[k],
               "--model", model, "--out", f"/kaggle/working/runs/{name}.jsonl",
               "--concurrency", str(conc)] + EXTRA.split()
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        tail = "\n".join(p.stdout.splitlines()[-12:])  # the scorer's summary
        print(f"=== {name} on server {k} [{time.time() - t:.0f}s]\n{tail}\n{p.stderr[-1500:]}", flush=True)

threads = [threading.Thread(target=lane, args=(k,)) for k in range(1 if SERIAL else len(URLS))]
[t.start() for t in threads]; [t.join() for t in threads]
'''

REPORT = r'''
for group, tag in ((set(ON_DEV), "dev"), ({n for n, _ in RUNS} - set(ON_DEV), "main")):  # one table per question set
    runs = [f"/kaggle/working/runs/{n}.jsonl" for n, _ in RUNS if n in group and os.path.exists(f"/kaggle/working/runs/{n}.jsonl")]
    if runs:
        p = subprocess.run([sys.executable, "scripts/summarize.py", *runs, "--baseline", BASELINE, "--traces", str(TRACES),
                            "--json", f"/kaggle/working/runs/summary_{tag}.json"], capture_output=True, text=True)
        print(f"##### {tag} questions", p.stdout, p.stderr[-2000:])
os.makedirs("/kaggle/working/runs/figures", exist_ok=True)
for n, _ in RUNS:  # Bonus 2: one timeline per run (agent events + telemetry on one clock)
    if os.path.isdir(f"/kaggle/working/runs/{n}_logs"):
        r = subprocess.run([sys.executable, "scripts/plot_timeline.py", f"/kaggle/working/runs/{n}_logs",
                            "--out", f"/kaggle/working/runs/figures/{n}_timeline.png"], capture_output=True, text=True)
        print(r.stdout, r.stderr[-500:])
if BENCH_MODELS and SERVER == "ollama":  # quantization: prefill/decode rates and memory from Ollama's native API
    src = next(f"/kaggle/working/runs/{n}_logs" for n, _ in RUNS if os.path.isdir(f"/kaggle/working/runs/{n}_logs"))
    r = subprocess.run([sys.executable, "scripts/bench_native.py", "--host", URLS[0][:-3], "--logs", src,
                        "--models", *BENCH_MODELS, "--n", "10", "--out", "/kaggle/working/runs/bench_native.json"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-1500:])
shutil.make_archive("/kaggle/working/runs_bundle", "zip", "/kaggle/working/runs")
print(os.path.getsize("/kaggle/working/runs_bundle.zip"), "bytes in runs_bundle.zip")
'''


def plan_runs(specs: list[str], default_model: str) -> list[tuple[str, str]]:
    """'NAME' or 'NAME@MODEL' (optionally ending in ':dev') -> (run name, model)."""
    runs = []
    for s in specs:
        name, _, model = s.removesuffix(":dev").partition("@")
        runs.append((name, model or default_model))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME@MODEL[:dev], one per run")
    ap.add_argument("--model", default="qwen3.5:4b-q8_0", help="model for runs without @MODEL")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=4, help="questions at once per run (and OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--extra", default="", help="extra runner flags for every run")
    ap.add_argument("--baseline", default=None, help="run the others are compared with (default: the first)")
    ap.add_argument("--traces", type=int, default=2)
    ap.add_argument("--questions-file", type=Path)
    ap.add_argument("--title", default="text2sql_V2 on Kaggle T4")
    ap.add_argument("--server", choices=["ollama", "vllm"], default="ollama")
    ap.add_argument("--hf-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--served-name", default="qwen3.5-4b", help="vLLM model name (= --model of the runs)")
    ap.add_argument("--tool-parser", default="qwen3_coder")
    ap.add_argument("--vllm-pip", default="vllm")
    ap.add_argument("--serial", action="store_true", help="all runs one after another on server 0")
    ap.add_argument("--concurrency", nargs="*", default=[], help="NAME=N: questions at once for that run")
    ap.add_argument("--bench-models", nargs="*", default=[], help="ollama models for scripts/bench_native.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.server == "vllm":
        a.model, a.gpus = a.served_name, 1
    runs = plan_runs(a.runs, a.model)
    a.baseline = a.baseline or runs[0][0]
    conc = {k: int(v) for k, v in (c.split("=") for c in a.concurrency)}
    on_dev = [r[0] for r, s in zip(runs, a.runs) if s.endswith(":dev")]
    qb64 = base64.b64encode(a.questions_file.read_bytes()).decode() if a.questions_file else ""
    params = (f"CODE = {code_b64()!r}\nRUNS = {runs!r}\nGPUS = {a.gpus}\nPARALLEL = {a.parallel}\n"
              f"EXTRA = {a.extra!r}\nBASELINE = {a.baseline!r}\nTRACES = {a.traces}\nQUESTIONS_B64 = {qb64!r}\n"
              f"SERVER = {a.server!r}\nHF_MODEL = {a.hf_model!r}\nSERVED_NAME = {a.served_name!r}\n"
              f"ON_DEV = {on_dev!r}\nTOOL_PARSER = {a.tool_parser!r}\nVLLM_PIP = {a.vllm_pip!r}\nSERIAL = {a.serial}\n"
              f"CONCURRENCY = {conc!r}\nBENCH_MODELS = {a.bench_models!r}\n")
    lanes = [[n for n, _ in runs]] if a.serial else [[n for n, _ in runs[k::a.gpus]] for k in range(a.gpus)]
    header = (f"# {a.title}\n\n"
              + "\n".join(f"- `{n}`: the PreAct agent on `{m}`" for n, m in runs)
              + f"\n\nGPU lanes (one model server each): {lanes}. "
              + f"Questions: {a.questions_file.name if a.questions_file else 'the 20 dev questions'}.\n\n"
              "Generated by `text2sql_V2/scripts/make_kaggle_notebook.py`. Needs GPU T4 x2, Internet on, "
              "and the private dataset holding kit.zip.")
    nb = {"cells": [cell(header, "markdown"), cell(params), cell(SETUP), cell(OLLAMA if a.server == "ollama" else VLLM), cell(RUN), cell(REPORT)],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KB); lanes {lanes}")


if __name__ == "__main__":
    main()
