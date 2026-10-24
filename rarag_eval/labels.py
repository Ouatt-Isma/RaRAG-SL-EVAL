"""The fair domain test's independent labeller (thesis FINDINGS F-58). The experts are built from one LLM's domain
labels (A), so placing questions by those labels is an oracle. A second model, from another family, labels every
question with the same categories and prompt (B: dataset/<name>.domains_b.json in the thesis repo); the fair
placements read only B. Runs on its own Ollama (the first GPU, Vulkan off), pulling the model if it is missing.
A dataset already labelled by B is skipped. Not while a grid run is going (both need the GPU).

  python -m rarag_eval.labels configs/confirm.yaml      # a few minutes per dataset
"""
import argparse
import os
import subprocess
import sys

from . import launch, store
from .factcheck import logged
from .run import load_config, log
from .thesis import ROOT, thesis_dir

PORT = 11650
LOG = os.path.join(ROOT, "logs", "labels.log")
CLAIMS = ("LIAR_RAW", "RAWFC")   # labelled with news topics (label_domains.py --scheme claims)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tag = cfg.get("labeller_b")
    if not tag:
        raise SystemExit("set labeller_b (an Ollama model tag from another family) in the config")
    thesis = thesis_dir()
    names = [store.dataset_name(d) for d in cfg["datasets"]]
    todo = [n for n in names if not os.path.exists(os.path.join(thesis, "dataset", f"{n}.domains_b.json"))]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if not todo:
        log(f"labeller B ({tag}): every dataset is labelled already", LOG)
        return
    if launch.running():
        raise SystemExit("a grid run is going: label before or after it (both need the GPU's memory)")
    gpus = launch.detect()
    gpu = gpus[0] if gpus else None
    proc = None
    try:
        launch.unload(os.environ.get("OLLAMA_HOST", launch.DEFAULT_OLLAMA).replace("http://", ""), LOG)
        host, proc = launch.start_ollama(gpu, PORT, LOG)
        env = {**os.environ, "OLLAMA_HOST": host, "OLLAMA_MODEL": tag,
               **({"CUDA_VISIBLE_DEVICES": str(gpu["index"])} if gpu else {})}
        log(f"labeller B: {tag}, pulled if missing", LOG)
        if subprocess.call(["ollama", "pull", tag], env=env):
            raise SystemExit(f"could not pull {tag}; see the message above")
        launch.verify(gpu, PORT, tag, LOG)
        for scheme, group in (("trivia", [n for n in todo if n not in CLAIMS]), ("claims", [n for n in todo if n in CLAIMS])):
            if group:
                log(f"labeller B on {', '.join(group)} ({scheme} categories)", LOG)
                code = logged([sys.executable, "scripts/label_domains.py", "--labeller", "b", "--scheme", scheme,
                               "--datasets", ",".join(group)], thesis, env, LOG)
                if code:
                    raise SystemExit(f"labelling failed (exit {code}); see logs/labels.log")
        log("done: make start runs the fair placements on them", LOG)
    finally:
        launch.stop([proc])


if __name__ == "__main__":
    main()
