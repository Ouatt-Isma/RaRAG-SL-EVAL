"""How each trust function's estimate of a source moves over the test: the paper's trust-over-time figure.
The saved results keep one outcome per test question; this replays the reliability-only analysis (the swap) on a
run's cached answers, without the LLM, and keeps what they drop: every source's trust before each test question,
and whether its answer was right.

  python -m rarag_eval.trace configs/confirm.yaml --runs-dir DIR [--embeddings FILE] [--only setup=attackers_k3]

Writes results/traces/<dataset>__s<seed>__<setup>.json.gz for every unit of SETUPS whose answers DIR holds
(a server backup's vector_cache; work on a copy, the lab rewrites its caches as it goes).
"""
import argparse
import gzip
import json
import os
import time

from . import analyses, setups, store
from .run import load_config, units
from .thesis import ROOT, Thesis

OUT = os.path.join(ROOT, "results", "traces")
SETUPS = ("unreliable_k3", "attackers_k3", "switch")


def path(dataset, seed, setup):
    return os.path.join(OUT, f"{store.dataset_name(dataset)}__s{seed}__{setup}.json.gz")


def trace(th, sc):
    """Each trust function's P per source before every test question (601 steps: after the warm-up, then after
    each question), RA-RAG's frozen estimate, and each source's answers: right, wrong, or None (IDK or ungrounded)."""
    out = {}
    for name, settings in (("plain", {}), ("aged", analyses.AGED)):
        res = th.sl_minimal.MinimalRun(sc, {**analyses.BASE, **settings}).swap()
        v = res["variants"]
        if name == "plain":
            out.update(sl_plain=v["sl_online"]["trajectory"]["P"], rarag_online=v["rarag_online"]["trajectory"]["P"],
                       rarag_frozen=v["rarag"]["reliability"],
                       correct=[q["source_correct"] for q in res["questions"]],
                       target=[q["is_target"] for q in res["questions"]])
        else:
            out.update(sl_aged=v["sl_online"]["trajectory"]["P"], rarag_window=v["rarag_window"]["trajectory"]["P"])
    out["sources"] = [{"name": s["name"], "kind": s.get("kind"), "reliability": s.get("reliability")} for s in sc.sources]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--runs-dir", required=True, help="the lab's runs directory holding the cached answers")
    ap.add_argument("--embeddings", help="a question-embedding cache to use (e.g. a server backup's)")
    ap.add_argument("--nli", default=os.path.join(ROOT, "cache", "nli.pkl"), help="the entailment decisions kept between runs")
    ap.add_argument("--only", nargs="*", default=[], help="filters: dataset=NQ_paper seed=0 setup=attackers_k3")
    args = ap.parse_args()
    cfg = load_config(args.config)
    only = dict(kv.split("=", 1) for kv in args.only)
    th = Thesis(cfg["model"], runs_dir=args.runs_dir, offline=True, embeddings=args.embeddings, nli_memo=args.nli)
    th.nli()
    os.makedirs(OUT, exist_ok=True)
    for dataset, seed, setup in units(cfg, only):
        if setup["id"] not in SETUPS:
            continue
        name = f"{store.dataset_name(dataset)} seed {seed} {setup['id']}"
        sc = setups.scenario(th, dataset, seed, setup, cfg)
        if not os.path.exists(os.path.join(sc.run_dir, "corpora.pkl")):
            continue   # not in this backup
        t0 = time.perf_counter()
        sc.build()
        if sc.estimation is None and not sc.load_cached_estimation():
            print(f"{name}: skipped (no cached warm-up answers)")
            continue
        body = {**trace(th, sc), "meta": {"dataset": store.dataset_name(dataset), "seed": seed, "setup": setup["id"],
                                          "run": sc.run_name, "n_est": cfg["n_est"], "n_test": cfg["n_test"],
                                          "thesis_commit": th.commit}}
        with gzip.open(path(dataset, seed, setup["id"]), "wt") as f:
            json.dump(body, f)
        print(f"{name}: {time.perf_counter() - t0:.0f} s -> {path(dataset, seed, setup['id'])}")
    th.save_nli()


if __name__ == "__main__":
    main()
