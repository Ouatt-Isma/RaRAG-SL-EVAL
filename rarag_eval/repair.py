"""Find the setups whose results were computed from empty LLM answers (rarag_eval/answers.py) and, with --apply, make
the next run redo them: their empty answers and what was computed from them are dropped from the run's cache, and
their results are deleted. Good answers stay cached, so the rerun only asks the LLM the questions that failed.
Runs on the machine that holds the runs' caches (the GPU server).

  python -m rarag_eval.repair configs/confirm.yaml            # report only
  python -m rarag_eval.repair configs/confirm.yaml --apply    # then: make start
"""
import argparse
import gzip
import json
import os
import shutil

from . import answers, store
from .run import load_config, units
from .thesis import thesis_dir


def run_of(dataset, seed, setup_id):
    """The lab run (folder name) and model of a setup's saved results."""
    d = os.path.dirname(store.path(dataset, seed, setup_id, "x"))
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.endswith(".json.gz"):
            with gzip.open(os.path.join(d, name), "rt") as f:
                meta = json.load(f)["meta"]
            return meta["run"], meta["model"], d
    return None, None, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--runs-dir", help="the lab's runs folder (default: the thesis repo's vector_cache)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    runs_dir = args.runs_dir or os.path.join(thesis_dir(), "vector_cache")
    bad, missing, good, shown = [], [], 0, 0
    for dataset, seed, setup in units(cfg, {}):
        run, model, results_dir = run_of(dataset, seed, setup["id"])
        if run is None:
            continue
        name = f"{store.dataset_name(dataset)} seed {seed} {setup['id']}"
        run_dir = os.path.join(runs_dir, run)
        if not os.path.isdir(run_dir):
            missing.append(name)
            continue
        rep = answers.check(run_dir, model)
        if not answers.has_empty(rep):
            good += 1
            continue
        bad.append((name, run_dir, model, results_dir, rep))
        (we, wn), (te, tn) = rep["warmup"], rep["test"]
        print(f"{name}: empty answers, warm-up {we}/{wn}, test {te}/{tn}")
        if shown < 2:   # what a failed answer looks like
            for ex in rep["examples"]:
                print(f"    {ex}")
            shown += 1
    print(f"\n{good} setups with no empty answer, {len(bad)} with some"
          + (f", {len(missing)} whose cache is not on this machine" if missing else ""))
    if not bad:
        return
    if not args.apply:
        print("nothing changed; with --apply their empty answers and results are dropped, then make start redoes them")
        return
    for name, run_dir, model, results_dir, _ in bad:
        answers.clean(run_dir, model)
        shutil.rmtree(results_dir)
    print(f"done: {len(bad)} setups will be redone by make start (good answers stay cached)")


if __name__ == "__main__":
    main()
