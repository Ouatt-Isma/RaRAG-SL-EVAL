"""How much the results depend on the model's settings: ageing (lambda), and the two thresholds of the decision rule
(u_max, delta), one at a time around the paper's values. Replays exported runs (no LLM): SL's rule on SL's trust, and
SL's trust inside RA-RAG's decision rule, as the attackers figure does.

The exports hold 3 and 4 attackers, 5 attackers (not in the paper), and unreliable or all-reliable sources; no late
switch and no honest mix, so this does not test lambda against a patient attacker. The baseline's decisions are
checked against the stored results where the run is in the grid.

  python -m rarag_eval.sensitivity EXPORTS_DIR      (make sensitivity; needs the thesis repository, like make run)
"""
import argparse
import collections
import glob
import gzip
import json
import os
import statistics
import sys

import numpy as np

from .calibration import ece, naurc
from .iterations import CODE, ROOT, decide
from .thesis import thesis_dir

GAMMA = 0.1
BASELINE = {"lam": 0.97, "u_max": 0.3, "margin": 0.1}
SWEEPS = {"lam": [0.9, 0.95, 0.97, 0.99, 0.995, 1.0], "u_max": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
          "margin": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]}
DECIDED = {"unknown": "u", "conflict": "k"}
MEASURES = ("coverage", "accuracy", "attack", "ece", "naurc", "vote_accuracy", "vote_attack")


def setup_of(d):
    """The setup of an exported run, from its sources."""
    kinds = collections.Counter(s["kind"] for s in d["sources"])
    a = kinds["adversary"]
    k = sum(s["name"].startswith("Unreliable") for s in d["sources"])
    return f"{a} attackers" if a else f"{k} unreliable" if k else "all reliable"


def replay(d, lam, u_max, margin, mm, outcome):
    """One run at one setting: the code of each test question's decision, and the measures."""
    n = len(d["sources"])
    st = mm.Settings(W=2, a=0.5, L=10, u_max=u_max, margin=margin, threshold=GAMMA, fusion="cumulative",
                     feedback="agreement", lam_pos=lam, lam_neg=lam)
    answers = lambda rec: [{"source": a["source"], "cluster": a["cluster"], "g": a["g"] if a["g"] is not None else 0.0}
                           for a in rec["answers"]]
    model = mm.warmup_fixed_point([answers(r) for r in d["records"] if r["phase"] == "warm"], n, st)["model"]
    code = lambda x, r: CODE[outcome(x, r["gold"], r["payload"], r["is_target"], "fair")]
    decisions, tops, votes = [], [], []
    test = [r for r in d["records"] if r["phase"] == "test"]
    for rec in test:
        votes.append(code(decide(rec, [n * model.opinion(s).P - 1 for s in range(n)], n), rec))
        out = model.step(answers(rec))["decision"]
        top = code(out["top"], rec) if out["top"] else None
        tops.append((out["c"], top))
        decisions.append(top if out["kind"] == "answer" else DECIDED[out["kind"]])
    targets = [r["is_target"] for r in test]
    given = [x for x in decisions if x in "cwa"]
    scored = [(c, t == "c") for c, t in tops if c is not None]
    conf, ok = [c for c, _ in scored], [float(o) for _, o in scored]
    attacks = [x for x, t in zip(decisions, targets) if t]
    voted = [x for x, t in zip(votes, targets) if t]
    return "".join(decisions), {
        "coverage": len(given) / len(decisions), "accuracy": given.count("c") / len(given) if given else None,
        "attack": attacks.count("a") / len(attacks) if attacks else None, "ece": ece(conf, ok) if conf else None,
        "naurc": naurc(conf, ok) if conf else None, "vote_accuracy": votes.count("c") / len(votes),
        "vote_attack": voted.count("a") / len(voted) if voted else None}


def stored(results_dir):
    """run name -> its stored decisions under SL's rule with ageing, for the runs in the grid."""
    out = {}
    for f in glob.glob(os.path.join(results_dir, "*", "seed*", "*", "answer_aged.json.gz")):
        with gzip.open(f) as fh:
            d = json.load(fh)
        out[d["meta"]["run"]] = (d["meta"]["setup"], d["rows"]["sl_online"]["decision"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", help="directory of exported runs (*.json.gz)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    sys.path.insert(0, thesis_dir())
    from src.sl import minimal as mm  # noqa: E402
    from src.sl.stream import outcome  # noqa: E402  (the eval's outcome, fair matching)

    grid = stored(args.results)
    table = collections.defaultdict(lambda: collections.defaultdict(list))   # (setup, knob, value) -> measure -> per run
    same = []
    for path in sorted(glob.glob(os.path.join(args.exports, "*.json.gz"))):
        with gzip.open(path) as fh:
            d = json.load(fh)
        setup = setup_of(d)
        base, check = replay(d, BASELINE["lam"], BASELINE["u_max"], BASELINE["margin"], mm, outcome)
        if d["run"] in grid:
            s = grid[d["run"]][1]
            same.append((d["run"], grid[d["run"]][0], sum(a == b for a, b in zip(base, s)) / len(s)))
        for knob, values in SWEEPS.items():
            for v in values:
                _, m = replay(d, **{**BASELINE, knob: v}, mm=mm, outcome=outcome)
                for k in MEASURES:
                    table[(setup, knob, v)][k].append(m[k])
        print(d["run"], setup, flush=True)
    print("\nthe baseline's decisions against the stored ones:")
    for run, setup, s in same:
        print(f"  {run}  {setup}: same on {100 * s:.1f}%")
    mean = lambda xs: statistics.mean(v) if (v := [x for x in xs if x is not None]) else None
    out = {}
    for setup in sorted({k[0] for k in table}):
        print(f"\n{setup} ({len(table[(setup, 'lam', 0.97)]['coverage'])} runs)")
        print(f"  {'':16}" + "".join(f"{m:>14}" for m in MEASURES))
        for knob, values in SWEEPS.items():
            for v in values:
                row = {m: mean(table[(setup, knob, v)][m]) for m in MEASURES}
                out[f"{setup}|{knob}|{v}"] = row
                mark = " *" if v == BASELINE[knob] else ""
                print(f"  {knob:7}{v:<7}{mark:2}" + "".join(f"{'-' if x is None else f'{100 * x:.1f}':>14}" for x in row.values()))
    target = os.path.join(args.results, "sensitivity.json")
    with open(target, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten: {target}")


if __name__ == "__main__":
    main()
