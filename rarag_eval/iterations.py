"""Does RA-RAG online need its estimator run to convergence? Replays RA-RAG online on exported runs (no LLM)
with cheaper re-estimations and compares each with the full one.

RA-RAG online (webapp/backend/sl_minimal.RaragOnline) re-runs iterative weighted majority voting from equal
weights until no majority changes (max_iter 100, tol 1e-20) on every answer seen so far, before every test
question. The cheaper re-estimations here:
  scratch1, scratch2   the same, stopped after 1 or 2 passes
  warm1                one pass over the whole record, starting from the previous question's weights
  incremental          only the new question's weighted majority is counted; past majorities are never revisited
                       (constant cost, but no longer RA-RAG's estimator)
Each test question: the weights before it, RA-RAG's decision (sources by weight until kappa raw answers, weighted
vote on the answer groups), its outcome with the fair matching. The exports (the thesis repository's exports/,
webapp/backend/sl_redesign.export_records) group a question's answers once over all its answers, where the eval
groups the kappa answers asked; the full re-estimation's row is checked against the stored result where the run
is in the grid.

  python -m rarag_eval.iterations EXPORTS_DIR      (make iterations; needs the thesis repository, like make run)
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import sys
import time

import numpy as np

from .thesis import thesis_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDK = "i don't know"
KAPPA = 4
CODE = {"correct": "c", "wrong": "w", "attacked": "a", "abstained": "x"}


def estimators():
    """name -> class(warm-up rows, number of sources) with .acc and .learn(row)"""
    from src.rarag_trace import iterative_wmv_trace
    from src.reliability_utils import update_reliability, weighted_majority_voting

    class Full:
        def __init__(self, warm, n, max_iter=100):
            self.n, self.max_iter, self.hist = n, max_iter, list(warm)
            self.acc = np.asarray(iterative_wmv_trace(self.hist, n, max_iter, 1e-20)[1], float)

        def learn(self, row):
            self.hist.append(row)
            self.acc = np.asarray(iterative_wmv_trace(self.hist, self.n, self.max_iter, 1e-20)[1], float)

    class Warm1(Full):
        def learn(self, row):
            self.hist.append(row)
            v = self.n * self.acc - 1
            consensus = weighted_majority_voting(self.hist, np.tile(v, (len(self.hist), 1)))
            self.acc = np.asarray(update_reliability(self.hist, consensus), float)

    class Incremental:
        def __init__(self, warm, n):
            self.n = n
            v, _, _ = iterative_wmv_trace(list(warm), n, 100, 1e-20)
            self.agree, self.seen = np.zeros(n), np.zeros(n)
            for row, c in zip(warm, weighted_majority_voting(warm, np.tile(v, (len(warm), 1)))):
                self._count(row, c)

        def _count(self, row, c):
            for s, a in enumerate(row):
                if a != IDK:
                    self.seen[s] += 1
                    self.agree[s] += a == c
            self.acc = np.divide(self.agree, self.seen, out=np.full(self.n, 0.5), where=self.seen != 0)

        def learn(self, row):
            self._count(row, weighted_majority_voting([row], [self.n * self.acc - 1])[0])

    return {"full": Full, "scratch1": lambda w, n: Full(w, n, 1), "scratch2": lambda w, n: Full(w, n, 2),
            "warm1": Warm1, "incremental": Incremental}


def decide(rec, w, n):
    """RA-RAG's inference: sources in descending weight until kappa give a raw answer, N/A padding, weighted vote."""
    by = {a["source"]: a for a in rec["answers"]}
    answered = []
    for s in [i for _, i in sorted(zip(w, range(n)), reverse=True)]:
        if len(answered) == KAPPA:
            break
        raw = by[s]["raw"]
        if raw is not None and str(raw).strip().lower() != IDK:
            answered.append(s)
    tally = {}
    for s in answered:
        key = by[s]["cluster"] or by[s]["raw"]
        tally[key] = tally.get(key, 0.0) + float(w[s])
    if len(answered) < KAPPA:
        tally.setdefault("N/A", 0.0)
    return max(tally, key=tally.get)


def stored(results_dir):
    """run name -> (setup label, the aged analysis' stored rows), for the runs in the grid"""
    out = {}
    for f in glob.glob(os.path.join(results_dir, "*", "seed*", "*", "aged.json.gz")):
        with gzip.open(f) as fh:
            d = json.load(fh)
        m = d["meta"]
        out[m["run"]] = (f"{m['dataset']} s{m['seed']} {m['setup']}", d)
    return out


def replay(path, makers, outcome):
    with gzip.open(path) as fh:
        d = json.load(fh)
    n = len(d["sources"])
    rows = lambda recs: [[{a["source"]: a for a in r["answers"]}[s]["filtered"] for s in range(n)] for r in recs]
    warm_recs = [r for r in d["records"] if r["phase"] == "warm"]
    test_recs = [r for r in d["records"] if r["phase"] == "test"]
    warm, test = rows(warm_recs), rows(test_recs)
    out = {"run": d["run"], "targets": [r["is_target"] for r in test_recs]}
    for name, make in makers.items():
        est = make(warm, n)
        finals, codes, ms = [], [], []
        for rec, row in zip(test_recs, test):
            final = decide(rec, n * est.acc - 1, n)
            finals.append(final)
            codes.append(CODE[outcome(final, rec["gold"], rec["payload"], rec["is_target"], "fair")])
            t = time.perf_counter()
            est.learn(row)
            ms.append(1000 * (time.perf_counter() - t))
        out[name] = {"finals": finals, "codes": "".join(codes), "ms": statistics.median(ms[-50:])}
    return out


def rates(codes, targets):
    tgt = [c for c, t in zip(codes, targets) if t]
    return codes.count("c") / len(codes), (tgt.count("a") / len(tgt) if tgt else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", help="directory of exported runs (*.json.gz)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    sys.path.insert(0, thesis_dir())
    from src.sl.stream import outcome  # noqa: E402  (the eval's outcome, fair matching)

    makers, grid = estimators(), stored(args.results)
    files = sorted(glob.glob(os.path.join(args.exports, "*.json.gz")))
    if not files:
        sys.exit(f"no exported runs in {args.exports}")
    cheap = [k for k in makers if k != "full"]
    summary = {k: {"same": [], "dacc": [], "dasr": [], "ms": []} for k in makers}
    print("accuracy / attack success (%), share of answers equal to the full re-estimation's, ms per question (last 50)")
    for f in files:
        r = replay(f, makers, outcome)
        label, check = grid.get(r["run"], ("(not in the grid)", None))
        acc0, asr0 = rates(r["full"]["codes"], r["targets"])
        line = [f"{r['run']}  {label}", f"  full        {100 * acc0:5.1f} / {'-' if asr0 is None else f'{100 * asr0:5.1f}'}"
                f"  {r['full']['ms']:6.2f} ms"]
        if check:
            s = check["rows"]["rarag_online"]["fair"]
            line[1] += f"   stored row: same outcome on {100 * sum(a == b for a, b in zip(r['full']['codes'], s)) / len(s):.1f}%"
        summary["full"]["ms"].append(r["full"]["ms"])
        for k in cheap:
            acc, asr = rates(r[k]["codes"], r["targets"])
            same = sum(a == b for a, b in zip(r[k]["finals"], r["full"]["finals"])) / len(r["full"]["finals"])
            summary[k]["same"].append(same)
            summary[k]["dacc"].append(acc - acc0)
            summary[k]["ms"].append(r[k]["ms"])
            if asr0 is not None:
                summary[k]["dasr"].append(asr - asr0)
            line.append(f"  {k:<11} {100 * acc:5.1f} / {'-' if asr is None else f'{100 * asr:5.1f}'}  {r[k]['ms']:6.2f} ms"
                        f"   same answer {100 * same:5.1f}%")
        print("\n".join(line), flush=True)
    rng = lambda xs: f"{100 * min(xs):+.1f} to {100 * max(xs):+.1f}" if xs else "-"
    print(f"\n{len(files)} runs: against the full re-estimation ({statistics.median(summary['full']['ms']):.2f} ms per question)")
    print(f"{'':<12} {'same answer':>14}  {'accuracy (pt)':>14}  {'attack (pt)':>14}  {'ms':>6}")
    for k in cheap:
        s = summary[k]
        print(f"{k:<12} {100 * min(s['same']):6.1f}-{100 * max(s['same']):5.1f}%  {rng(s['dacc']):>14}  {rng(s['dasr']):>14}"
              f"  {statistics.median(s['ms']):6.2f}")


if __name__ == "__main__":
    main()
