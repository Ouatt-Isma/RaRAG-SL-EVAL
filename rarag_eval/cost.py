"""What keeping the trust up to date costs per query, the trust function alone (no LLM, no AlignScore, no
clustering: those are the same for every method that hears every source).

RA-RAG online (webapp/backend/sl_minimal.RaragOnline) has no incremental form: its estimate is the fixed point
of iterative weighted majority voting over the whole record, so before every query it re-runs the estimator from
equal weights on every answer seen so far (with a window, on the last N queries). SL's Step 4 reads only the
current query's answers and adds to each source's record. This times both on answers shaped like the confirm grid
(9 sources, coverage 0.6, reliability 0.9, 3 wrong answers per question), at the grid's record size (200 warm-up
+ 600 test queries) and at larger ones, on one CPU core.

  python -m rarag_eval.cost            (needs the thesis repository, like make run)
"""
import argparse
import statistics
import sys
import time

import numpy as np

from .thesis import thesis_dir

N_SOURCES, COVERAGE, RELIABILITY, N_WRONG = 9, 0.6, 0.9, 3
IDK = "i don't know"


def answers(rng, q):
    """One query's filtered answers, one per source (RA-RAG's estimator compares them as strings)."""
    row = []
    for _ in range(N_SOURCES):
        if rng.random() > COVERAGE:
            row.append(IDK)
        elif rng.random() < RELIABILITY:
            row.append(f"gold {q}")
        else:
            row.append(f"wrong {q} {rng.randint(N_WRONG)}")
    return row


def sl_answers(row):
    return [{"source": s, "cluster": None if a == IDK else a, "g": 0.0 if a == IDK else 0.9} for s, a in enumerate(row)]


def median_seconds(f, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[800, 8000, 80000], help="record sizes (queries)")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, thesis_dir())
    from src.rarag_trace import iterative_wmv_trace  # noqa: E402  (what RaragOnline.learn calls, same settings)
    from src.sl import minimal as mm  # noqa: E402

    rng = np.random.RandomState(args.seed)
    rows = [answers(rng, q) for q in range(max(args.sizes) + 1)]
    model = mm.TrustModel(N_SOURCES, mm.Settings(feedback="agreement", lam_pos=0.97, lam_neg=0.97))
    for r in rows[:200]:
        model.step(sl_answers(r))
    sl = median_seconds(lambda: model.step(sl_answers(rows[-1])), 2000)

    print(f"one query's trust update, {N_SOURCES} sources (median, ms)")
    print(f"{'record':>8}  {'RA-RAG online':>13}  {'iterations':>10}  {'RA-RAG window':>13}  {'SL':>8}  {'online / SL':>11}")
    for size in args.sizes:
        record = rows[:size]
        iters = len(iterative_wmv_trace(record, N_SOURCES, 100, 1e-20)[2])
        repeat = max(3, min(200, 400_000 // size))
        online = median_seconds(lambda: iterative_wmv_trace(record, N_SOURCES, 100, 1e-20), repeat)
        window = median_seconds(lambda: iterative_wmv_trace(record[-args.window:], N_SOURCES, 100, 1e-20), 200)
        print(f"{size:>8}  {1000 * online:>13.2f}  {iters:>10}  {1000 * window:>13.2f}  {1000 * sl:>8.3f}  {online / sl:>11.0f}")


if __name__ == "__main__":
    main()
