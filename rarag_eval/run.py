"""Run the evaluation grid of a config: every dataset x seed x setup, each through its analyses. Saved
results are skipped, so it resumes where it stopped. LLM answers come from the local model through the
thesis lab (Ollama on a GPU server); --offline uses cached answers only and skips what would need the LLM.
`make run` starts it through launch.py, once per GPU (--claim: the workers share the grid).

  python -m rarag_eval.run configs/confirm.yaml
  python -m rarag_eval.run configs/confirm.yaml --only dataset=NQ_paper seed=0 setup=attackers_k2
  python -m rarag_eval.run configs/pilot.yaml --offline --runs-dir ../Ra-Rag-Thesis-Adaptation/server_backup/vector_cache
"""
import argparse
import os
import re
import time
import traceback

import yaml

from . import analyses, answers, setups, share, store
from .thesis import ROOT, Thesis


def available(dataset):
    """An optional dataset runs once it is built (its file in the thesis repo's dataset/) and is reported once it
    has results (a machine that only reports has no thesis datasets)."""
    from .thesis import thesis_dir
    try:
        built = os.path.exists(os.path.join(thesis_dir(), "dataset", os.path.basename(dataset)))
    except SystemExit:   # no thesis repo here
        built = False
    return built or os.path.isdir(os.path.join(store.RESULTS, store.dataset_name(dataset)))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = {"kappa": 4, "top_k": 3, "filtering_threshold": 0.1, "coverage": 0.6, "target_fraction": 0.5,
                "n_est": 200, "n_test": 600}
    cfg = {**defaults, **cfg}
    cfg["datasets"] = list(cfg["datasets"]) + [d for d in cfg.get("optional_datasets") or []
                                               if d not in cfg["datasets"] and available(d)]
    ids = [s["id"] for s in cfg["setups"]]
    if len(set(ids)) != len(ids):
        raise SystemExit("setup ids must be unique")
    return cfg


def units(cfg, only):
    """(dataset, seed, setup) in run order: dataset by dataset, seed by seed, so a dataset and seed finish
    together and can be reported."""
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            for setup in cfg["setups"]:
                unit = {"dataset": store.dataset_name(dataset), "seed": str(seed), "setup": setup["id"]}
                if all(unit.get(k) == v for k, v in only.items()):
                    yield dataset, seed, setup


def log(msg, path):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


class LogJob:
    """The lab reports a long step's progress on a job (its web page polls it: .progress, .message); here it goes to
    the log, when the step changes and at least every 3 minutes, so building the sources, the warm-up answers and
    the test answers are not silent for half an hour."""

    def __init__(self, name, path, every=180):
        self.name, self.path, self.every = name, path, every
        self.progress, self._message, self._step, self._at = 0.0, "", None, 0.0

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, text):
        self._message = text
        step = re.sub(r"[\d.,/%]+", "#", text or "")
        if step != self._step or time.time() - self._at >= self.every:
            self._step, self._at = step, time.time()
            pct = f" ({self.progress:.0%})" if self.progress else ""
            log(f"  {self.name}: {text}{pct}", self.path)


def no_empty_answers(sc, name):
    """Stop a setup computed from failed LLM requests (empty answers, read as "I don't know") before anything of it
    is saved; its empty answers are dropped from the cache, so the next run asks those questions again."""
    rep = answers.check(sc.run_dir, sc.model)
    if answers.has_empty(rep):
        answers.clean(sc.run_dir, sc.model)
        (we, wn), (te, tn) = rep["warmup"], rep["test"]
        raise RuntimeError(f"the LLM gave empty answers (warm-up {we}/{wn}, test {te}/{tn}): failed requests, see "
                           f"logs/ollama.*.log; dropped from the cache, the next make start asks them again")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*", default=[], help="filters: dataset=NQ_paper seed=0 setup=attackers_k2")
    ap.add_argument("--offline", action="store_true", help="cached LLM answers only; skip setups that need more")
    ap.add_argument("--runs-dir", help="the lab's runs directory (default: the thesis repo's vector_cache)")
    ap.add_argument("--embeddings", help="a question-embedding cache to use (e.g. a server backup's)")
    ap.add_argument("--nli", default=os.path.join(ROOT, "cache", "nli.pkl"), help="the entailment decisions kept between runs")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--claim", action="store_true", help="one of several workers: take only setups no other worker took")
    ap.add_argument("--log", default=os.path.join("logs", "run.log"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    only = dict(kv.split("=", 1) for kv in args.only)
    logfile = os.path.join(ROOT, args.log)
    if cfg.get("ollama_model"):   # the Ollama tag (the model config names a 4-bit one); OLLAMA_MODEL overrides
        os.environ.setdefault("OLLAMA_MODEL", cfg["ollama_model"])
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    todo = []
    for dataset, seed, setup in units(cfg, only):
        missing = [a for a in analyses.plan(setup, cfg["n_est"]) if not store.exists(dataset, seed, setup["id"], a[0])]
        if missing:
            todo.append((dataset, seed, setup, missing))
    log(f"{args.config}: {len(todo)} setups with analyses to run", logfile)
    if args.dry_run:
        for dataset, seed, setup, missing in todo:
            print(f"  {store.dataset_name(dataset)} seed {seed} {setup['id']}: {len(missing)} analyses")
        return
    if not todo:
        return
    th = Thesis(cfg["model"], runs_dir=args.runs_dir, offline=args.offline, embeddings=args.embeddings, nli_memo=args.nli)
    th.nli()
    tag = os.environ.get("OLLAMA_MODEL")
    log(f"thesis {th.path} @ {th.commit}, runs in {th.runs_dir}, model {th.model}{f' ({tag})' if tag else ''}"
        f"{', offline' if args.offline else ''}, GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}", logfile)
    for dataset, seed, setup, missing in todo:
        name = f"{store.dataset_name(dataset)} seed {seed} {setup['id']}"
        if args.claim and not share.claim(dataset, seed, setup["id"]):
            continue
        try:
            t0 = time.perf_counter()
            # building writes caches shared by every setup of a dataset (its topic model, the embeddings):
            # one worker at a time
            with share.locked(os.path.join(th.runs_dir, ".build")):
                sc = setups.scenario(th, dataset, seed, setup, cfg)
                if args.offline and not os.path.exists(os.path.join(sc.run_dir, "corpora.pkl")):
                    log(f"{name}: skipped (offline, sources never built)", logfile)
                    continue
                sc.build(LogJob(name, logfile))
            if sc.estimation is None and not sc.load_cached_estimation():
                if args.offline:
                    log(f"{name}: skipped (offline, no cached warm-up answers)", logfile)
                    continue
                sc.estimate(LogJob(name, logfile))
            no_empty_answers(sc, name)
            log(f"{name}: run {sc.run_name} ready in {time.perf_counter() - t0:.0f} s", logfile)
            for analysis, kind, settings in missing:
                t1, memo0 = time.perf_counter(), len(th.nli()._memo)
                job = LogJob(f"{name} {analysis}", logfile)
                body = (analyses.swap(th, sc, settings, cfg["kappa"], job) if kind == "swap"
                        else analyses.compare(th, sc, settings, job))
                no_empty_answers(sc, name)
                body["meta"] = {"dataset": store.dataset_name(dataset), "seed": seed, "setup": setup["id"], "setup_spec": setup,
                                "analysis": analysis, "kind": kind, "settings": {**analyses.BASE, **settings},
                                "kappa": cfg["kappa"], "n_est": cfg["n_est"], "n_test": cfg["n_test"], "run": sc.run_name,
                                "model": th.model, "llm_tag": tag, "thesis_commit": th.commit,
                                "seconds": round(time.perf_counter() - t1, 1)}
                store.save(body, dataset, seed, setup["id"], analysis)
                if len(th.nli()._memo) > memo0:
                    th.save_nli()
                log(f"  {name} {analysis}: {body['meta']['seconds']:.0f} s", logfile)
        except Exception as e:   # one failing setup must not stop the grid
            if args.offline and "offline" in str(e):
                log(f"{name}: skipped ({e})", logfile)
            else:
                log(f"{name}: FAILED {e!r}\n{traceback.format_exc()}", logfile)
    log("DONE", logfile)


if __name__ == "__main__":
    main()
