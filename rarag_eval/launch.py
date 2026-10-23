"""Run the grid on every GPU of the machine. Each worker is one `rarag_eval.run` with its own Ollama server, both
pinned to one GPU (the LLM, AlignScore, the entailment model and the retriever all run there); the workers take
the grid's setups one at a time from a shared queue, so they finish together. Workers per GPU: one per 21 GB
of its memory (about 18.5 GB measured for Llama 3 8B fp16 and the lab's models), so one on a 24 GB card, two on
a 48 GB card, three on an 80 GB card. Even a single worker gets its own Ollama, so its settings (4 requests at
once, 2048-token context) are the same on every machine, whatever the installed Ollama service is set to.

  python -m rarag_eval.launch configs/confirm.yaml                 # every GPU
  python -m rarag_eval.launch configs/confirm.yaml --gpus 2        # the first 2 GPUs
  python -m rarag_eval.launch configs/confirm.yaml --plan          # what it would start, nothing more
  python -m rarag_eval.launch configs/confirm.yaml --status        # how far the run is, time left (any terminal)
  python -m rarag_eval.launch configs/confirm.yaml --tmux          # the same run, inside a detached tmux session
  (any other option goes to rarag_eval.run, e.g. --only dataset=NQ_paper)
"""
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from . import share
from .progress import Progress, only_filters
from .run import load_config, log
from .thesis import ROOT

SLOT_MIB = 21 * 1024
FIRST_PORT = 11500
DEFAULT_OLLAMA = "127.0.0.1:11434"


def parse_smi(text, visible=None):
    """nvidia-smi's "index, name, memory.total[, compute_mode]" CSV lines as [{index, name, mib, mode}], only the
    GPUs that CUDA_VISIBLE_DEVICES leaves visible when it lists indices."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) in (3, 4) and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1], "mib": int(float(parts[2])),
                         "mode": parts[3] if len(parts) == 4 else "Default"})
    keep = [v.strip() for v in (visible or "").split(",") if v.strip()]
    if keep and all(v.isdigit() for v in keep):
        gpus = [g for g in gpus if str(g["index"]) in keep]
    return gpus


def detect():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_mode",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_smi(out, os.environ.get("CUDA_VISIBLE_DEVICES"))


CUDA_CHECK = "import torch; torch.ones(8, device='cuda').sum().item()"


def last_error(text):
    """The exception line of a traceback ("torch.AcceleratorError: CUDA error: ..."), not a hint that only mentions
    an error name; the last line when there is none."""
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    found = [l for l in lines if re.match(r"^[\w.]*(Error|Exception)\b.*:", l)]
    return (found or lines or [""])[-1][:240]


def healthy(gpus, logfile, python=sys.executable):
    """The GPUs a worker can use. Each GPU holds two processes, its Ollama server and its worker, so one that
    allows a single process (compute mode Exclusive_Process or Prohibited) is left out, and so is one where a
    small CUDA test fails (a faulty or busy card), each with the reason in the log."""
    ok = []
    for g in gpus:
        if g.get("mode", "Default") != "Default":
            log(f"GPU {g['index']} left out: compute mode {g['mode']} allows only one process "
                f"(nvidia-smi -i {g['index']} -c DEFAULT, as root, would allow it)", logfile)
        else:
            ok.append(g)
    procs = [(g, subprocess.Popen([python, "-c", CUDA_CHECK], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g["index"])},
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)) for g in ok]
    out = []
    for g, p in procs:
        try:
            _, err = p.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            err, p.returncode = "no answer in 3 minutes", 1
        if p.returncode == 0:
            out.append(g)
        else:
            log(f"GPU {g['index']} left out: a small CUDA test failed there ({last_error(err)})", logfile)
    return out


def plan(gpus, n_gpus=None, per_gpu=None, workers=None):
    """[(gpu or None, port)] for each worker."""
    gpus = gpus[:n_gpus] if n_gpus else gpus
    if workers:   # an explicit count, spread over the GPUs (or none, e.g. offline on a laptop)
        return [(gpus[w % len(gpus)] if gpus else None, FIRST_PORT + w) for w in range(workers)]
    slots = [(g, per_gpu or max(1, g["mib"] // SLOT_MIB)) for g in gpus]
    return [(g, FIRST_PORT + w) for w, g in enumerate(g for g, n in slots for _ in range(n))] or [(None, FIRST_PORT)]


def http(url, body=None, timeout=5):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def up(host):
    try:
        http(f"http://{host}/api/tags")
        return True
    except Exception:
        return False


def unload(host, logfile):
    """Models the default Ollama keeps on a GPU (it loads them for good) would leave a worker's GPU short."""
    try:
        for m in http(f"http://{host}/api/ps").get("models", []):
            http(f"http://{host}/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=60)
            log(f"unloaded {m['name']} from the Ollama at {host} (it frees its GPU memory; the server keeps running)", logfile)
    except Exception:
        pass


def start_ollama(gpu, port, logfile, context=2048):
    host = f"127.0.0.1:{port}"
    if up(host):
        log(f"Ollama at {host}: already running, reused", logfile)
        return host, None
    # OLLAMA_VULKAN off: through Vulkan Ollama sees every GPU, whatever CUDA_VISIBLE_DEVICES says, and puts the model
    # on the one with the most free memory, another worker's (2026-10-23: out of memory, thousands of empty answers)
    env = {**os.environ, "OLLAMA_HOST": host, "OLLAMA_NUM_PARALLEL": os.environ.get("OLLAMA_NUM_PARALLEL", "4"),
           "OLLAMA_CONTEXT_LENGTH": str(context), "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_VULKAN": "false"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    out = open(os.path.join(ROOT, "logs", f"ollama.{port}.log"), "w")   # this run's only: it logs every request
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(60):
        if up(host):
            break
        time.sleep(1)
    else:
        raise SystemExit(f"Ollama at {host} did not start; see logs/ollama.{port}.log")
    log(f"Ollama at {host} on GPU {gpu['index'] if gpu else '-'} started", logfile)
    return host, proc


def verify(gpu, port, tag, logfile):
    """Before any worker starts: the server loads the model on its own GPU through CUDA and answers a question. Run
    on the wrong backend or out of memory, it answers every request with nothing, which the lab reads as "I don't
    know": a whole grid of setups where no source seems to know anything."""
    host = f"127.0.0.1:{port}"
    try:
        r = http(f"http://{host}/api/generate", {"model": tag, "prompt": "Answer with one word: what is the capital of France?",
                                                 "stream": False, "options": {"num_predict": 8, "temperature": 0}}, timeout=900)
    except Exception as e:
        raise SystemExit(f"Ollama at {host} could not answer a test question ({e}); see logs/ollama.{port}.log")
    text = (r.get("response") or "").strip()
    if not text:
        raise SystemExit(f"Ollama at {host} answered a test question with nothing; see logs/ollama.{port}.log")
    backend = None
    try:
        with open(os.path.join(ROOT, "logs", f"ollama.{port}.log"), errors="replace") as f:
            for line in f:
                if "selecting GPU backend" in line:
                    m = re.search(r"library=(\w+)", line)
                    backend = m[1] if m else backend
    except OSError:
        pass
    where = f"GPU {gpu['index']}" if gpu else "this machine"
    if gpu is not None and backend and backend.lower() != "cuda":
        raise SystemExit(f"Ollama at {host} put the model on {backend}, not on {where} through CUDA; see logs/ollama.{port}.log")
    vram = sum(m.get("size_vram", 0) for m in http(f"http://{host}/api/ps").get("models", []))
    if gpu is not None and not vram:
        raise SystemExit(f"Ollama at {host} runs the model on the CPU, not on {where}; see logs/ollama.{port}.log")
    log(f"Ollama at {host}: {tag} on {where}{' (' + backend + ')' if backend else ''}, {vram / 2**30:.1f} GB, "
        f"test answer {text!r}", logfile)


def stop(procs):
    for p in procs:
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for p in procs:
        if p:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def wait(procs, progress, logfile, every=30 * 60, names=None, poll=30):
    """Until the workers end: the progress and time left each time a setup finishes, and every 30 minutes; a worker
    that stops with an error at once, in the run's log and in its own (its last line is what make status shows)."""
    last_n, last_t, reported = 0, time.time(), set()
    while [p.poll() for p in procs].count(None):   # every worker polled (any() would stop at the first running one)
        time.sleep(poll)
        for w, p in enumerate(procs):
            if p.returncode not in (None, 0) and w not in reported:
                reported.add(w)
                name, own, err = (names or {}).get(w, (f"worker {w}", None, None))
                last = ""
                try:
                    with open(err) as f:
                        last = last_error(f.read())
                except (OSError, TypeError):
                    pass
                msg = f"{name} stopped with an error (exit {p.returncode}){': ' + last if last else ''}; the others go on"
                log(msg, logfile)
                if own:
                    log(msg, own)
        n = len(progress.done()[0])
        if n != last_n or time.time() - last_t >= every:
            log(progress.line(), logfile)
            last_n, last_t = n, time.time()
    return [p.returncode for p in procs]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def running():
    try:
        return alive(Progress.load().pid)
    except (OSError, ValueError, KeyError):
        return False


def in_tmux(argv, session, module="rarag_eval.launch", busy=None, what="a run", log="logs/run.log"):
    """Start `python -m module argv` (argv: its options, less --tmux) inside a detached tmux session: it keeps going
    when the terminal closes or the connection drops. The window stays open at the end, to read how it ended."""
    busy = busy or running
    if not shutil.which("tmux"):
        raise SystemExit("tmux is not installed (Ubuntu, WSL: sudo apt install tmux; Mac: brew install tmux)")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        if busy():
            raise SystemExit(f"{what} is going in tmux session {session}: make attach SESSION={session} to watch it")
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)   # the window of one that ended
    # a tmux server already running would give the session its own environment, not this shell's
    keep = [f"{k}={os.environ[k]}" for k in ("OLLAMA_MODEL", "OLLAMA_NUM_PARALLEL", "THESIS_REPO", "CUDA_VISIBLE_DEVICES")
            if k in os.environ]
    cmd = shlex.join(["env", *keep, sys.executable, "-m", module, *argv])
    # then a shell in the same window: the last lines stay above it, and stray input cannot close it
    cmd += (f"; echo; echo 'It has ended (log: {log}). This window is now a shell here: exit closes it.'"
            "; exec \"${SHELL:-/bin/bash}\"")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", ROOT, cmd], check=True)
    attach = "make attach" + ("" if session == "eval" else f" SESSION={session}")
    status = ("make status          how far it is and the time left" if module == "rarag_eval.launch"
              else "make claims-status   how far it is")
    print(f"started in tmux session {session}.\n  {attach}   watch it (then Ctrl-b, d: leave it running)\n"
          f"  {status}, from any terminal")


def status():
    try:
        p = Progress.load()
    except (OSError, ValueError):
        print("no run started here yet")
        return
    alive_now = alive(p.pid)
    print(f"{os.path.relpath(p.config, ROOT)}, {p.workers} worker{'s' if p.workers > 1 else ''}, started "
          f"{time.strftime('%a %d %b %H:%M', time.localtime(p.t0))}"
          + ("" if alive_now else f", ended {time.strftime('%H:%M', time.localtime(p.t_end))}" if p.t_end else " (not running now)"))
    print(p.line())
    if alive_now:   # what each worker is doing now: its log's last line
        logs = ([os.path.join(ROOT, "logs", f"run.w{w}.log") for w in range(p.workers)] if p.workers > 1
                else [os.path.join(ROOT, "logs", "run.log")])
        for w, path in enumerate(logs):
            try:
                if os.path.getmtime(path) >= p.t0:
                    with open(path) as f:
                        print(f"  worker {w}: {f.read().rstrip().splitlines()[-1].strip()}")
            except (OSError, IndexError):
                pass


def interrupted(*_):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--gpus", type=int, help="use only the first N GPUs")
    ap.add_argument("--per-gpu", type=int, help="workers per GPU (default: one per 21 GB of its memory)")
    ap.add_argument("--workers", type=int, help="this many workers, spread over the GPUs (testing: works without one)")
    ap.add_argument("--plan", action="store_true", help="print the GPUs and workers, start nothing")
    ap.add_argument("--status", action="store_true", help="how far the run is and how long it has left")
    ap.add_argument("--tmux", action="store_true", help="run inside a detached tmux session (see make attach)")
    ap.add_argument("--session", default="eval", help="the tmux session's name")
    args, rest = ap.parse_known_args()
    cfg = load_config(args.config)
    if args.status:
        return status()
    if args.tmux:
        argv = [args.config, *(f"--{k.replace('_', '-')}={v}" for k, v in
                               (("gpus", args.gpus), ("per_gpu", args.per_gpu), ("workers", args.workers)) if v), *rest]
        return in_tmux(argv, args.session)
    signal.signal(signal.SIGTERM, interrupted)   # e.g. the tmux session closed: stop the workers too
    logfile = os.path.join(ROOT, "logs", "run.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    gpus = healthy(detect(), logfile)
    workers = plan(gpus, args.gpus, args.per_gpu, args.workers)
    offline = "--offline" in rest
    tag = os.environ.get("OLLAMA_MODEL") or cfg.get("ollama_model") or "(the model config's)"

    gpu_text = ", ".join(f"GPU {g['index']} {g['name']} {g['mib'] / 1024:.0f} GB" for g in gpus) or "no NVIDIA GPU"
    log(f"{gpu_text}: {len(workers)} worker{'s' if len(workers) > 1 else ''}, LLM {tag}"
        + (", offline" if offline else ""), logfile)
    if args.plan:
        for w, (g, port) in enumerate(workers):
            print(f"  worker {w}: GPU {g['index'] if g else '-'}, "
                  + ("no LLM (offline)" if offline else f"its own Ollama at 127.0.0.1:{port}"))
        return

    run = [sys.executable, "-m", "rarag_eval.run", args.config, *rest]
    progress = Progress(args.config, only_filters(rest), len(workers), len({g["index"] for g, _ in workers if g}) or 1)
    if not progress.start:
        log(f"nothing to run: every analysis of {args.config}{' (--only ' + ' '.join(f'{k}={v}' for k, v in progress.only.items()) + ')' if progress.only else ''} is saved", logfile)
        return
    progress.save()
    log(progress.line(), logfile)
    share.clear_claims()
    servers, procs = [], []
    try:
        if not offline:
            unload(os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA).replace("http://", ""), logfile)
            for g, port in workers:
                host, proc = start_ollama(g, port, logfile)
                servers.append(proc)
            if cfg.get("ollama_model") or os.environ.get("OLLAMA_MODEL"):
                for g, port in workers:
                    verify(g, port, os.environ.get("OLLAMA_MODEL") or cfg["ollama_model"], logfile)

        def env_for(w, g, port):
            env = {**os.environ, "RARAG_WORKER": str(w)}
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g["index"])
            if not offline:
                env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
            return env

        if len(workers) == 1:   # its own log lines in this terminal too
            procs.append(subprocess.Popen(run, cwd=ROOT, env=env_for(0, *workers[0]), start_new_session=True))
            code = wait(procs, progress, logfile)[0]
            log(f"run done (exit code {code})", logfile)
            return
        names = {}
        for w, (g, port) in enumerate(workers):
            env = env_for(w, g, port)
            wlog = os.path.join("logs", f"run.w{w}.log")
            err = open(os.path.join(ROOT, "logs", f"run.w{w}.err"), "a")   # warnings, and a crash's traceback
            procs.append(subprocess.Popen(run + ["--claim", "--log", wlog], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                                          stderr=err, start_new_session=True))
            log(f"worker {w} on GPU {g['index'] if g else '-'}: started, log {wlog}", logfile)
            names[w] = (f"worker {w} on GPU {g['index'] if g else '-'}", os.path.join(ROOT, wlog), err.name)
        codes = wait(procs, progress, logfile, names=names)
        log(f"workers done (exit codes {codes})", logfile)
    except KeyboardInterrupt:
        log("stopped: stopping the workers (saved analyses are kept; make run resumes)", logfile)
    finally:
        stop(procs)
        stop(servers)
        progress.finish()
    log(progress.line(), logfile)
    log("DONE", logfile)


if __name__ == "__main__":
    main()
