"""How honest each method's confidence is, from the saved results, no new run. Two decision rules on the same answers,
both with trust that keeps learning and forgets:

  SL     its own rule (answer, unknown, conflict) on SL's trust with aging: confidence c = P(x*) of the top
         candidate, whose outcome is saved for every question, answered or not ("answer_aged", row "sl_online")
  RA-RAG its weighted vote on RA-RAG's estimate over a window of 100 questions: the confidence is the winner's
         share of the vote, the only one it gives ("aged", row "rarag_window")

Per run: the mean confidence and the accuracy it should match, the calibration error (ECE, 10 bins), the area
under the risk-coverage curve (AURC: answers taken from the most confident down, mean error rate; lower ranks the
errors last), how often the confidence reaches 0.9 and how often such answers are right, the share of questions SL
answers (coverage) and RA-RAG's accuracy on its own most confident answers at that coverage."""
import gzip
import json
import os
import statistics

import numpy as np

from . import store

SETUPS = ["unreliable_k3", "mix5", "attackers_k1", "attackers_k2", "attackers_k3", "attackers_k4", "switch"]


def _load(dataset, seed, setup, analysis):
    p = store.path(dataset, seed, setup, analysis)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= 1))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(out)


def aurc(conf, ok):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    err = 1 - np.asarray(ok, float)[order]
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


def _measures(conf, ok, n_questions):
    hi = [o for c, o in zip(conf, ok) if c >= 0.9]
    return {"mean_conf": float(np.mean(conf)), "accuracy": float(np.mean(ok)), "ece": ece(conf, ok), "aurc": aurc(conf, ok),
            "hi_share": len(hi) / n_questions, "hi_accuracy": float(np.mean(hi)) if hi else None}


def unit(dataset, seed, setup):
    """The measures of one run, or None when its analyses are not saved."""
    own, vote = _load(dataset, seed, setup, "answer_aged"), _load(dataset, seed, setup, "aged")
    if not own or not vote:
        return None
    row = own["rows"]["sl_online"]
    pairs = [(c, t == "c") for c, t in zip(row["conf"], row["top"]) if c is not None and t in "cwa"]
    sl = _measures([c for c, _ in pairs], [o for _, o in pairs], len(row["top"]))
    decided = row["decision"]
    answered = [d for d in decided if d in "cwa"]
    sl.update({"coverage": len(answered) / len(decided), "unknown": decided.count("u") / len(decided),
               "conflict": decided.count("k") / len(decided),
               "answered_accuracy": answered.count("c") / len(answered) if answered else None})
    rr = vote["rows"]["rarag_window"]
    pairs = [(c, o == "c") for c, o in zip(rr["conf"], rr["fair"]) if c is not None]
    ra = _measures([c for c, _ in pairs], [o for _, o in pairs], len(rr["fair"]))
    k = int(round(sl["coverage"] * len(pairs)))   # RA-RAG answering only its most confident share, at SL's coverage
    top = np.argsort(-np.asarray([c for c, _ in pairs], float), kind="stable")[:k]
    ra["same_coverage_accuracy"] = float(np.mean([pairs[i][1] for i in top])) if k else None
    return {"sl": sl, "rarag": ra}


def _answers(dataset, seed, setup):
    """(confidence, right) of every question each side answers (SL: its top candidate, answered or not), and the
    number of test questions; None when the analyses are not saved."""
    own, vote = _load(dataset, seed, setup, "answer_aged"), _load(dataset, seed, setup, "aged")
    if not own or not vote:
        return None
    row, rr = own["rows"]["sl_online"], vote["rows"]["rarag_window"]
    sl = [(c, t == "c") for c, t in zip(row["conf"], row["top"]) if c is not None and t in "cwa"]
    ra = [(c, o == "c") for c, o in zip(rr["conf"], rr["fair"]) if c is not None and o != "x"]
    return {"sl": sl, "rarag": ra}, len(row["top"])


def reliability(datasets, seeds, setup, bins=10, min_share=0.01):
    """The reliability diagram, pooled over runs: per bin of confidence holding at least min_share of the
    answers, (mean confidence, accuracy, share of the questions)."""
    runs = [a for d in datasets for s in seeds if (a := _answers(d, s, setup))]
    n = sum(k for _, k in runs)
    out = {}
    for side in ("sl", "rarag"):
        conf = np.array([c for a, _ in runs for c, _ in a[side]], float)
        ok = np.array([o for a, _ in runs for _, o in a[side]], float)
        idx = np.minimum((conf * bins).astype(int), bins - 1)
        out[side] = [(float(conf[m].mean()), float(ok[m].mean()), m.sum() / n)
                     for b in range(bins) if (m := idx == b).sum() >= min_share * len(conf)]
    return out


def _risk(conf, ok, n, coverages):
    """Share wrong among the answers taken from the most confident down, at each coverage (share of the n
    questions); equal confidences are taken together, pro rata. None past the last answer."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    levels = np.unique(conf)[::-1]
    taken = np.concatenate([[0], np.cumsum([(conf == v).sum() for v in levels])]) / n
    wrong = np.concatenate([[0], np.cumsum([(1 - ok[conf == v]).sum() for v in levels])]) / n
    return [float(np.interp(g, taken, wrong) / g) if g <= taken[-1] + 1e-9 else None for g in coverages], taken[-1]


def risk_coverage(datasets, seeds, setup, step=0.02):
    """The risk-coverage curve of each side, mean over runs: [(coverage, risk)] up to the coverage every run
    reaches, then the mean of the runs' last points."""
    runs = [a for d in datasets for s in seeds if (a := _answers(d, s, setup))]
    grid = np.arange(step, 1 + 1e-9, step)
    out = {}
    for side in ("sl", "rarag"):
        curves = [_risk([c for c, _ in a[side]], [o for _, o in a[side]], n, grid) for a, n in runs]
        ends = [(last, _risk([c for c, _ in a[side]], [o for _, o in a[side]], n, [last])[0][0])
                for (a, n), (_, last) in zip(runs, curves)]
        pts = [(float(g), statistics.mean(r[i] for r, _ in curves)) for i, g in enumerate(grid)
               if all(r[i] is not None for r, _ in curves)]
        end = (statistics.mean(e[0] for e in ends), statistics.mean(e[1] for e in ends))
        out[side] = pts + ([end] if end[0] > pts[-1][0] + 1e-9 else [])
    return out


def summary(datasets, seeds, setups=SETUPS):
    """{setup: {"sl": {measure: mean over runs}, "rarag": {...}, "n": runs}}."""
    out = {}
    for setup in setups:
        units = [u for d in datasets for s in seeds if (u := unit(d, s, setup))]
        if not units:
            continue
        mean = lambda side, k: (statistics.mean(v) if (v := [u[side][k] for u in units if u[side].get(k) is not None])
                                else None)
        out[setup] = {side: {k: mean(side, k) for k in units[0][side]} for side in ("sl", "rarag")}
        out[setup]["n"] = len(units)
        # in how many runs SL ranks its errors last better than RA-RAG
        out[setup]["aurc_sl_better"] = sum(u["sl"]["aurc"] < u["rarag"]["aurc"] for u in units)
    return out
