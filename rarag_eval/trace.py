"""How each trust function's estimate of a source moves over the test, against the source's true reliability. The
saved results keep one outcome per test question; this replays the reliability-only analysis (the swap) on a run's
cached answers, without the LLM, and keeps what they drop: every source's trust before each test question, and
whether its answer was right. Not in the paper (thesis FINDINGS F-60): our trust estimates reliability through
agreement, which tracks it only when most answering sources are right.

  python -m rarag_eval.trace configs/confirm.yaml --runs-dir DIR [--embeddings FILE] [--only setup=attackers_k3]
  python -m rarag_eval.trace configs/confirm.yaml --figure FILE.tex [--panels attackers_k3 attackers_k4]

The first writes results/traces/<dataset>__s<seed>__<setup>.json.gz for every unit of SETUPS whose answers DIR holds
(on the server, the thesis's own vector_cache; on a laptop, a copy of a server backup's: the lab rewrites
its caches as it goes). The second draws saved
traces of one run (pgfplots), one panel per setup.
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
SETUPS = ("unreliable_k3", "mix5", "attackers_k1", "attackers_k2", "attackers_k3", "attackers_k4", "switch")
TITLES = {"unreliable_k3": "3 unreliable of 9", "mix5": "honest mix", "attackers_k1": "1 attacker of 9",
          "attackers_k2": "2 attackers of 9", "attackers_k3": "3 attackers of 9", "attackers_k4": "4 attackers of 9",
          "switch": "late switch, 3 of 9"}
RUN = ("NQ_paper", 0)   # a trajectory is one run's, not a mean
LINES = [("sl_aged", "solid", "SL, ageing"), ("rarag_window", "dashed", "RA-RAG, window"),
         ("rarag_frozen", "dotted", "RA-RAG, frozen")]
GAP = 0.1   # pairs whose true reliabilities differ by less are not counted in the order
GROUPS = {"reliable": "green!50!black", "unreliable": "orange!90!black", "attacker": "red!80!black"}


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


def save(th, sc, dataset, seed, setup_id, cfg):
    """Replay a built scenario and save its trace."""
    body = {**trace(th, sc), "meta": {"dataset": store.dataset_name(dataset), "seed": seed, "setup": setup_id,
                                      "run": sc.run_name, "n_est": cfg["n_est"], "n_test": cfg["n_test"],
                                      "thesis_commit": th.commit}}
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(path(dataset, seed, setup_id), "wt") as f:
        json.dump(body, f)


def group(source):
    kind, name = source.get("kind"), source["name"].lower()
    return "attacker" if kind == "adversary" or name.startswith("attacker") else \
        "unreliable" if name.startswith("unreliable") else "reliable"


def true_reliability(t, sources):
    """The sources' true reliability over the test: the share of right answers among their kept answers."""
    kept = [c for row in t["correct"] for i in sources if (c := row[i]) is not None]
    return sum(kept) / len(kept)


def series(t, line):
    """A trust function's P per source at each of the 601 steps (RA-RAG frozen: its warm-up estimate throughout)."""
    return [t["rarag_frozen"]] * (len(t["correct"]) + 1) if line == "rarag_frozen" else t[line]


def error(t, line):
    """Per test question, the mean over sources of |trust - true reliability|, the trust before the question."""
    true = [true_reliability(t, [i]) for i in range(len(t["sources"]))]
    s = series(t, line)
    return [sum(abs(s[q][i] - r) for i, r in enumerate(true)) / len(true) for q in range(len(t["correct"]))]


def order(t, line, gap=GAP):
    """The share of source pairs whose true reliabilities differ by at least gap that the trust orders the same way
    (a tie counts half), mean over the test questions; None when no pair differs that much."""
    true = [true_reliability(t, [i]) for i in range(len(t["sources"]))]
    pairs = [(i, j) for i in range(len(true)) for j in range(i + 1, len(true)) if abs(true[i] - true[j]) >= gap]
    if not pairs:
        return None
    s = series(t, line)
    right = lambda q, i, j: 1.0 if (s[q][i] - s[q][j]) * (true[i] - true[j]) > 0 else 0.5 if s[q][i] == s[q][j] else 0.0
    n = len(t["correct"])
    return sum(sum(right(q, i, j) for i, j in pairs) / len(pairs) for q in range(n)) / n


def figure(traces, every=10, width="4.9cm", height="4.1cm"):
    """One plot per setup, side by side: each kind of source's mean trust before every test question, for each of
    LINES, against its true reliability over the test (light band), with each trust function's mean absolute error (the mean
    over the test and the sources of |trust - true reliability|) written in the plot."""
    present = {group(s) for t in traces.values() for s in t["sources"]}
    common = (rf"width={width}, height={height}, xmin=0, xmax=600, ymin=0, ymax=1, xtick={{0,200,400,600}},"
              r" ytick={0,0.5,1}, tick label style={font=\scriptsize}, label style={font=\scriptsize},"
              r" title style={font=\scriptsize}, ymajorgrids, grid style={gray!20}, no markers, xlabel={test query}")
    out = [f"% Generated by rarag-sl-eval (python -m rarag_eval.trace --figure): one run ({RUN[0]}, seed {RUN[1]}).",
           r"\begin{tikzpicture}"]
    prev = None
    for k, (setup, t) in enumerate(traces.items()):
        groups = {}
        for i, s in enumerate(t["sources"]):
            groups.setdefault(group(s), []).append(i)
        place = rf"at={{({prev}.east)}}, anchor=west, xshift=0.55cm, yticklabels={{}}, " if prev else r"ylabel={trust}, "
        legend = (r"legend to name=trustlegend, legend columns=3, legend cell align=left, legend style={font=\scriptsize,"
                  r" draw=none, /tikz/every even column/.append style={column sep=5pt}}, ") if k == 0 else ""
        title = "" if len(traces) == 1 else rf", title={{({chr(97 + k)}) {TITLES.get(setup, setup)}}}"
        out.append(rf"\begin{{axis}}[name=p{k}, {place}{legend}{common}{title}]")
        for g, color in GROUPS.items():
            if g not in groups:
                continue
            for line, dash, _ in LINES:
                s = series(t, line)
                pts = " ".join(f"({q},{sum(s[q][i] for i in groups[g]) / len(groups[g]):.3f})"
                               for q in range(0, len(s), every))
                out.append(rf"\addplot[{color}, {dash}, thick, forget plot] coordinates {{{pts}}};")
            level = true_reliability(t, groups[g])
            out.append(rf"\addplot[{color}, line width=3pt, opacity=0.25, forget plot] coordinates"
                       rf" {{(0,{level:.3f}) (600,{level:.3f})}};")
        errors = r"\\".join(rf"{label} {sum(e) / len(e):.2f}" for line, _, label in LINES if (e := error(t, line)))
        out.append(rf"\node[anchor=south west, font=\scriptsize, align=left, inner sep=1.5pt, fill=white, fill opacity=0.8,"
                   rf" text opacity=1] at (axis cs:10,0.01) {{mean absolute error (MAE):\\{errors}}};")
        if k == 0:
            for g, color in GROUPS.items():
                if g in present:
                    out.append(rf"\addlegendimage{{{color}, thick}}\addlegendentry{{{g}}}")
            out.append(r"\addlegendimage{black, line width=3pt, opacity=0.25}\addlegendentry{true reliability}")
            for _, dash, label in LINES:
                out.append(rf"\addlegendimage{{black, {dash}, thick}}\addlegendentry{{{label}}}")
        out.append(r"\end{axis}")
        prev = f"p{k}"
    out.append(rf"\node[anchor=north] at ($(p0.outer south)!0.5!({prev}.outer south)$) {{\pgfplotslegendfromname{{trustlegend}}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--runs-dir", help="the lab's runs directory holding the cached answers (default: the thesis's)")
    ap.add_argument("--embeddings", help="a question-embedding cache to use (e.g. a server backup's)")
    ap.add_argument("--nli", default=os.path.join(ROOT, "cache", "nli.pkl"), help="the entailment decisions kept between runs")
    ap.add_argument("--only", nargs="*", default=[], help="filters: dataset=NQ_paper seed=0 setup=attackers_k3")
    ap.add_argument("--figure", help="draw saved traces of one run into this .tex file instead of replaying")
    ap.add_argument("--panels", nargs="*", default=["attackers_k3", "attackers_k4"], help="the setups --figure draws")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.figure:
        traces = {s: json.load(gzip.open(path(f"{RUN[0]}.json", RUN[1], s))) for s in args.panels}
        with open(args.figure, "w") as f:
            f.write(figure(traces))
        for s, t in traces.items():   # the numbers a reader checks against the picture
            for line, _, _ in LINES:
                e = error(t, line)
                print(s, line, "error", round(sum(e) / len(e), 3), "order", round(order(t, line), 3))
        return
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
        save(th, sc, dataset, seed, setup["id"], cfg)
        print(f"{name}: {time.perf_counter() - t0:.0f} s -> {path(dataset, seed, setup['id'])}")
    th.save_nli()


if __name__ == "__main__":
    main()
