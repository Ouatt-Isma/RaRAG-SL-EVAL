"""Optional: the fact-checking datasets LIAR-RAW and RAWFC, built once on the GPU machine with the local LLM (the
thesis repo's scripts/build_claims_dataset.py, FINDINGS F-57), then labelled with news topics for the experts
setup. Once a dataset is built, the grid includes it (`optional_datasets` in the config).

It starts its own Ollama for the build, on the first GPU, with a 4,096-token context (a claim's prompt and its nine
passages do not fit in the 2,048 tokens the runs use), and stops it at the end. Not while a grid run is going: both
need the GPU's memory. A dataset already built is skipped, and every LLM call is cached, so running it again after
an interruption carries on where it stopped.

  python -m rarag_eval.factcheck configs/confirm.yaml             # RAWFC, then LIAR-RAW (about 3-4 h each), then topics
  python -m rarag_eval.factcheck configs/confirm.yaml --check 3   # a first look: 3 RAWFC claims built and printed
  python -m rarag_eval.factcheck configs/confirm.yaml --tmux      # the same inside tmux session "claims"
"""
import argparse
import os
import shutil
import subprocess
import sys

from . import launch
from .run import load_config, log
from .thesis import ROOT, thesis_dir

DATASETS = [("rawfc", "RAWFC"), ("liar_raw", "LIAR_RAW")]
PORT = 11600
PIDFILE = os.path.join(ROOT, "logs", "claims.pid")
LOG = os.path.join(ROOT, "logs", "claims.log")


def building():
    try:
        with open(PIDFILE) as f:
            return launch.alive(int(f.read()))
    except (OSError, ValueError):
        return False


def logged(cmd, cwd, env):
    """Run cmd, its output both here and in logs/claims.log."""
    with open(LOG, "a") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            out.write(line)
            out.flush()
        return proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--check", type=int, default=0, help="build this many RAWFC claims and print them; keep nothing")
    ap.add_argument("--tmux", action="store_true", help="run inside a detached tmux session")
    ap.add_argument("--session", default="claims")
    args = ap.parse_args()
    if args.tmux:
        argv = [args.config] + (["--check", str(args.check)] if args.check else [])
        return launch.in_tmux(argv, args.session, module="rarag_eval.factcheck", busy=building, what="a claims build",
                              log="logs/claims.log")
    cfg = load_config(args.config)
    thesis = thesis_dir()
    raw = os.path.join(thesis, "dataset", "raw", "CofCED")
    if not os.path.isdir(raw) and not any(shutil.which(t) for t in ("bsdtar", "unar", "unrar")):
        raise SystemExit("the CofCED release holds RAR archives: apt install -y libarchive-tools (for bsdtar), then again")
    if launch.running():
        raise SystemExit("a grid run is going: build the claims before or after it (both need the GPU's memory)")
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

    tag = os.environ.get("OLLAMA_MODEL") or cfg.get("ollama_model")
    gpus = launch.detect()
    gpu = gpus[0] if gpus else None
    proc = None
    try:
        launch.unload(os.environ.get("OLLAMA_HOST", launch.DEFAULT_OLLAMA).replace("http://", ""), LOG)
        host, proc = launch.start_ollama(gpu, PORT, LOG, context=4096)
        if tag:
            launch.verify(gpu, PORT, tag, LOG)
        env = {**os.environ, "OLLAMA_HOST": host, **({"OLLAMA_MODEL": tag} if tag else {}),
               **({"CUDA_VISIBLE_DEVICES": str(gpu["index"])} if gpu else {})}
        build = [sys.executable, "scripts/build_claims_dataset.py"]
        if args.check:
            logged(build + ["--dataset", "rawfc", "--download", "--check", str(args.check)], thesis, env)
            return
        built = []
        for name, file in DATASETS:
            if os.path.exists(os.path.join(thesis, "dataset", f"{file}.json")):
                log(f"{file}: already built (dataset/{file}.json in the thesis repo), skipped", LOG)
                built.append(file)
                continue
            log(f"{file}: building with {tag or 'the model config'} (log: logs/claims.log)", LOG)
            code = logged(build + ["--dataset", name, "--download", "--out", f"dataset/{file}.json"], thesis, env)
            if code:
                log(f"{file}: FAILED (exit {code}); run make claims again to carry on", LOG)
            else:
                built.append(file)
        todo = [f for f in built if not os.path.exists(os.path.join(thesis, "dataset", f"{f}.domains.json"))]
        if todo:
            log(f"domain labels for {', '.join(todo)}", LOG)
            logged([sys.executable, "scripts/label_domains.py", "--scheme", "claims", "--datasets", ",".join(todo)],
                   thesis, env)
        log(f"done: {', '.join(built) or 'nothing'} built. make start runs the grid on them too", LOG)
    finally:
        launch.stop([proc])
        try:
            os.remove(PIDFILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
