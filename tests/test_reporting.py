"""The reporting path on synthetic saved results: collecting, the paired statistics, the claims, the LaTeX
tables and the figures. No thesis code, no LLM."""
import os
import random

import pytest

from rarag_eval import claims, collect, stats, store, tables


def swap_record(rng, n=60, setup="attackers_k2", accs=None):
    accs = accs or {"rarag": 0.6, "rarag_online": 0.8, "rarag_window": 0.82, "sl_offline": 0.62, "sl_online": 0.85, "equal": 0.55}
    # attacked questions are the same in every analysis of a setup (the scenario picks them)
    qs = [{"idx": 200 + i, "target": i % 2 == 0, "domain": ["film_tv", "music"][i % 3 == 0]} for i in range(n)]
    rows = {}
    for name, acc in accs.items():
        codes = "".join("c" if rng.random() < acc else ("a" if q["target"] and rng.random() < 0.5 else "w") for q in qs)
        rows[name] = {"fair": codes, "strict": codes, "conf": [round(rng.random(), 3) for _ in qs]}
    return {"questions": qs, "rows": rows, "extra": {},
            "sources": [{"name": "S1", "rarag_estimate": 0.97, "oracle_warm": 0.58}, {"name": "S2", "rarag_estimate": 0.4, "oracle_warm": 0.45}],
            "meta": {"kind": "swap", "thesis_commit": "abc", "model": "m", "run": "r", "seconds": 1}}


def compare_record(rng, n=60):
    qs = [{"idx": 200 + i, "target": i % 2 == 0, "domain": None} for i in range(n)]
    dec = "".join(rng.choice("cccccwauk") for _ in qs)
    rows = {"sl_online": {"decision": dec, "top": "".join(rng.choice("cwa") for _ in qs),
                          "pair": "".join(("1" if rng.random() < 0.8 else "0") if d == "k" else "." for d in dec),
                          "conf": [round(rng.random(), 3) for _ in qs]},
            "sl_offline": {"decision": dec.replace("a", "w"), "top": "c" * n, "pair": "." * n, "conf": [0.5] * n}}
    return {"questions": qs, "rows": rows, "extra": {}, "meta": {"kind": "compare", "thesis_commit": "abc", "model": "m", "run": "r", "seconds": 1}}


@pytest.fixture
def results(tmp_path, monkeypatch):
    root = tmp_path / "results"
    rng = random.Random(1)
    for d in ["NQ_paper", "TQA_paper"]:
        for seed in [0, 1]:
            for setup in ["attackers_k2", "attackers_k3", "attackers_k4"]:
                for analysis in ["plain", "aged", "window"]:
                    store.save(swap_record(rng, setup=setup), d + ".json", seed, setup, analysis, root=str(root))
                store.save(compare_record(rng), d + ".json", seed, setup, "answer_aged", root=str(root))
            store.save(swap_record(rng, setup="experts"), d + ".json", seed, "experts", "plain", root=str(root))
    monkeypatch.setattr(store, "RESULTS", str(root))
    monkeypatch.setattr(tables, "TABLES", str(tmp_path / "tables"))
    return str(root)


def test_paired_statistics():
    a, b = "c" * 50 + "w" * 50, "c" * 70 + "w" * 30
    assert stats.mcnemar(a, b) < 0.001 and stats.mcnemar(a, a) == 1.0
    lo, hi = stats.paired_bootstrap(a, b)
    assert lo > 0 and hi <= 0.3
    agg = stats.aggregate([0.02, 0.03, -0.01], +1)
    assert agg["n"] == 3 and agg["holds"] == 2 and agg["lo"] < agg["mean"] < agg["hi"]
    assert stats.aggregate([0.005, -0.004], 0)["holds"] == 2


def test_same_coverage_accuracy_ranks_by_confidence():
    codes, conf = "ccww", [0.9, 0.8, 0.2, 0.1]
    assert collect.same_coverage_accuracy(codes, conf, 0.5) == 1.0
    assert collect.same_coverage_accuracy(codes, conf, 1.0) == 0.5
    # ties count by their mean: half the questions at one shared confidence, half of them right
    assert collect.same_coverage_accuracy("cwcw", [0.5] * 4, 0.5) == 0.5


def test_collect_claims_tables_and_figures(results, tmp_path, monkeypatch):
    from rarag_eval import plots
    monkeypatch.setattr(plots, "FIGURES", str(tmp_path / "figures"))
    _, index, rows = collect.collect(results)
    assert {r["kind"] for r in rows} == {"swap", "compare"}
    ans = [r for r in rows if r["analysis"] == "answer_aged" and r["row"] == "sl_online"]
    assert all(0 <= r["coverage"] <= 1 and r["rarag_same_coverage"] is not None for r in ans)
    unit = claims.unit_results(index)
    by = {r["claim"] for r in unit}
    assert {"2", "3", "9a", "9b", "15"} <= by          # what these synthetic runs can confirm
    assert all(r["claim"] != "8" for r in unit)          # no warm-up draws saved: not claimed
    two = next(r for r in unit if r["claim"] == "2")
    assert abs(two["diff"] - 0.39) < 1e-9                 # the largest overestimate: 0.97 against 0.58
    summary = claims.summary(unit, ["NQ_paper", "TQA_paper"])
    tables.claims_table(summary, ["NQ_paper", "TQA_paper"])
    tables.robustness_tables(rows, ["NQ_paper", "TQA_paper"], ["attackers_k2", "attackers_k3", "attackers_k4"], "accuracy",
                             "attackers_accuracy", "cap")
    tables.answering_table(rows, ["attackers_k2", "attackers_k3"])
    text = open(os.path.join(tables.TABLES, "claims.tex")).read()
    assert r"\begin{table*}" in text and "TriviaQA" in text and "Holds" in text
    rob = open(os.path.join(tables.TABLES, "attackers_accuracy.tex")).read()
    assert r"\scriptsize$\pm$" in rob                     # two seeds: mean and spread
    plots.claims_forest(summary, ["NQ_paper", "TQA_paper"])
    plots.by_k(rows, ["NQ_paper", "TQA_paper"], ["attackers_k2", "attackers_k3", "attackers_k4"], [2, 3, 4], "accuracy", "acc", "accuracy")
    for f in ["claims_forest.pdf", "acc.pdf"]:
        assert os.path.getsize(os.path.join(plots.FIGURES, f)) > 1000


def test_latex_escaping():
    assert tables.esc("a_b & 50%") == r"a\_b \& 50\%"
