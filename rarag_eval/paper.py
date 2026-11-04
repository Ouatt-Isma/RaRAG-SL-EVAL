"""The exhibits of the paper's evaluation, written from the saved results into the paper's own folders, so every
number in them comes from here:

  tables/tab-results.tex      the main results: our model against RA-RAG, each a paired comparison on the same
                              answers: both means over every run (3 datasets x 3 seeds), the mean difference with
                              its 95% t interval, and the runs in which ours is ahead
  tables/tab-decision.tex     SL's own decision rule against RA-RAG's vote: coverage, accuracy, how far the confidence
                              sits from the accuracy, and how well it ranks the errors last (rarag_eval/calibration.py)
  figures/fig-attackers.tex   accuracy and attack success against the number of attackers among 9 sources (pgfplots)
  figures/fig-calibration.tex the two confidences under attack: reliability diagram and risk-coverage curve (pgfplots)
  figures/fig-trust.tex       each trust function's estimate over one run's test, against the share of right answers
                              (from results/traces/, python -m rarag_eval.trace)

  python -m rarag_eval.paper configs/confirm.yaml --out ../6adde414c2e32fc022487a80
"""
import argparse
import gzip
import json
import os
import statistics

from . import calibration, claims, collect, stats, store, trace
from .claims import Claim
from .run import load_config

FROZEN = ("plain", "rarag", "RA-RAG frozen (paper)")
AGED = ("aged", "sl_online", "SL with ageing")
# The main results, groups in reading order of the evaluation. A row is (label, comparison): a claim (its id in
# claims.CLAIMS, or one of its own), or ("cal", setup, measure, direction), a measure of calibration.unit with
# SL's rule against RA-RAG's vote.
MAIN = [
    (r"A patient adversary: SL with ageing vs.\ RA-RAG's frozen trust", [
        (r"Attacks that succeed, $a{=}2$", "3"),
        (r"Attacks that succeed, $a{=}3$", Claim("m1", "", ["attackers_k3"], "attack", FROZEN, AGED, -1)),
        (r"Accuracy, $a{=}3$", Claim("m2", "", ["attackers_k3"], "acc", FROZEN, AGED, +1)),
        (r"Accuracy, late switch", Claim("m3", "", ["switch"], "acc", FROZEN, AGED, +1)),
    ]),
    (r"Scarce or stale evidence: the opinion vs.\ the rate", [
        (r"Accuracy, 5 warm-up queries", "8"),
        (r"Attacks that succeed, stale trust", "15"),
    ]),
    (r"Honest confidence, $a{=}4$: SL's rule vs.\ RA-RAG's vote", [
        (r"Calibration error (ECE)", ("cal", "attackers_k4", "ece", -1)),
        (r"AURC", ("cal", "attackers_k4", "aurc", -1)),
        (r"Right when $c\ge0.9$", ("cal", "attackers_k4", "hi_accuracy", +1)),
    ]),
    (r"Domain experts: SL per domain vs.\ RA-RAG's one trust", [
        (r"Accuracy, LLM label", Claim("m4", "", ["experts"], "acc", FROZEN, ("dom_domain_wfull", "sl_online", ""), +1)),
        (r"Accuracy, nearest centroid",
         Claim("m5", "", ["experts"], "acc", FROZEN, ("dom_centroid_wfull", "sl_online", ""), +1)),
    ]),
]
CALIBRATION_SETUP = "attackers_k4"   # the figure's setup: the attackers are close to a majority
DATASETS = ["NQ_paper", "TQA_paper", "HotpotQA_paper"]
SEEDS = [0, 1, 2]
SETUP_SHORT = {"unreliable_k3": "3 unreliable", "mix5": "honest mix", "attackers_k1": "1 attacker",
               "attackers_k2": "2 attackers", "attackers_k3": "3 attackers", "attackers_k4": "4 attackers",
               "switch": "late switch"}
LINES = [  # the attackers figure: (analysis, row, legend, style)
    ("plain", "rarag", "RA-RAG, frozen", "red!75!black, dashed, mark=o"),
    ("plain", "rarag_online", "RA-RAG, online", "red!75!black, dotted, mark=o"),
    ("aged", "rarag_window", "RA-RAG, window", "red!75!black, solid, mark=o"),
    ("plain", "sl_offline", "SL, frozen", "blue!70!black, dashed, mark=square"),
    ("aged", "sl_online", "SL, ageing", "blue!70!black, solid, mark=square"),
    ("plain", "equal", "equal weights", "gray, dashdotted, mark=triangle"),
]


def num(x, digits=1, sign=True):
    return "--" if x is None else (f"${x:+.{digits}f}$" if sign else f"{x:.{digits}f}").replace("-", "-")


def compare(comparison, ix):
    """One main result over every run: {"base", "ours"} means and the aggregate of the paired differences."""
    if isinstance(comparison, tuple):
        _, setup, measure, direction = comparison
        units = [u for d in DATASETS for s in SEEDS if (u := calibration.unit(d, s, setup))
                 and u["sl"][measure] is not None and u["rarag"][measure] is not None]
        base, ours = [u["rarag"][measure] for u in units], [u["sl"][measure] for u in units]
        diffs = [o - b for b, o in zip(base, ours)]
    else:
        claim = next(c for c in claims.CLAIMS if c.id == comparison) if isinstance(comparison, str) else comparison
        direction = claim.direction
        res = [r for d in DATASETS for s in SEEDS for setup in claim.setups
               if (r := claims.evaluate(claim, ix, (d, s), setup))]
        base, ours, diffs = [r["base"] for r in res], [r["method"] for r in res], [r["diff"] for r in res]
    if not diffs:
        return None
    return {"base": statistics.mean(base), "ours": statistics.mean(ours), **stats.aggregate(diffs, direction)}


def results_table(ix):
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Result & RA-RAG & Ours & $\Delta$ [95\% CI] & Runs \\", r"\midrule"]
    for group, items in MAIN:
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
        for label, comparison in items:
            a = compare(comparison, ix)
            if not a:
                continue
            ci = (rf"\,{{\scriptsize[${100 * a['lo']:+.1f}$, ${100 * a['hi']:+.1f}$]}}" if a["lo"] is not None else "")
            lines.append(rf"\quad {label} & {100 * a['base']:.1f} & \textbf{{{100 * a['ours']:.1f}}}"
                         rf" & ${100 * a['mean']:+.1f}${ci} & {a['holds']}/{a['n']} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def decision_table(cal):
    pct = lambda x: "--" if x is None else f"{100 * x:.1f}"
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{$\bar c-$accuracy} & \multicolumn{2}{c}{AURC} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"Setup & Cov. & SL & Vote & SL & Vote & SL & Vote \\", r"\midrule"]
    for setup, label in SETUP_SHORT.items():
        if setup not in cal:
            continue
        sl, ra = cal[setup]["sl"], cal[setup]["rarag"]
        gap = lambda m: None if m["mean_conf"] is None else m["mean_conf"] - m["accuracy"]
        lines.append(f"{label} & {pct(sl['coverage'])} & {pct(sl['answered_accuracy'])} & {pct(ra['same_coverage_accuracy'])}"
                     f" & {num(None if gap(sl) is None else 100 * gap(sl))} & {num(None if gap(ra) is None else 100 * gap(ra))}"
                     f" & {pct(sl['aurc'])} & {pct(ra['aurc'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def attackers_figure(rows):
    """Two plots side by side (accuracy, attack success) against the number of attackers, one line per trust
    function, mean over every run; the paper scales the picture to the column width."""
    def mean(setup, analysis, row, metric):
        xs = [r[metric] for r in rows if r["setup"] == setup and r["analysis"] == analysis and r["row"] == row
              and r.get(metric) is not None]
        return 100 * statistics.mean(xs) if xs else None
    common = (r"width=4.9cm, height=4.1cm, xtick={1,2,3,4}, xlabel={attackers of 9}, tick label style={font=\scriptsize},"
              r" label style={font=\scriptsize}, ymajorgrids, grid style={gray!20}, mark size=1.4pt")
    out = ["% Generated by rarag-sl-eval (python -m rarag_eval.paper): mean over 3 datasets x 3 seeds.",
           r"\begin{tikzpicture}"]
    for name, metric, ylabel, place in (("acc", "accuracy", r"accuracy (\%)", ""),
                                        ("att", "attack", r"attacks that succeed (\%)", r"at={(acc.east)}, anchor=west, xshift=1.1cm, ")):
        legend = (r"legend to name=attlegend, legend columns=3, legend style={font=\scriptsize, draw=none,"
                  r" /tikz/every even column/.append style={column sep=5pt}}, ") if name == "att" else ""
        out.append(rf"\begin{{axis}}[name={name}, {place}{legend}{common}, ylabel={{{ylabel}}}]")
        for analysis, row, _, style in LINES:
            pts = " ".join(f"({k},{v:.1f})" for k in (1, 2, 3, 4)
                           if (v := mean(f"attackers_k{k}", analysis, row, metric)) is not None)
            out.append(rf"\addplot[{style}, thick] coordinates {{{pts}}};")
        if name == "att":
            out.append(r"\legend{" + ",".join("{" + label + "}" for _, _, label, _ in LINES) + "}")   # labels hold commas
        out.append(r"\end{axis}")
    out.append(r"\node[anchor=north] at ($(acc.outer south)!0.5!(att.outer south)$) {\pgfplotslegendfromname{attlegend}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def calibration_figure(rel, curves, rule):
    """Two plots side by side, one setup under attack: (a) accuracy per bin of stated confidence against the
    diagonal, (b) the share wrong when only the most confident answers are given, against the share answered,
    with the point where SL's rule stops (rule: its coverage and risk). Full text width, at its own size."""
    common = (r"width=8cm, height=5.2cm, tick label style={font=\scriptsize}, label style={font=\footnotesize},"
              r" title style={font=\footnotesize}, grid=major, grid style={gray!20}, mark size=1.6pt")
    style = {"rarag": "red!75!black, mark=o", "sl": "blue!70!black, mark=square"}
    lo = int(10 * min(v for side in rel.values() for x, y, _ in side for v in (x, y))) / 10
    top_c, top_acc, top_share = max(rel["rarag"])          # the vote's most confident bin
    out = ["% Generated by rarag-sl-eval (python -m rarag_eval.paper): pooled over 3 datasets x 3 seeds.",
           r"\begin{tikzpicture}",
           rf"\begin{{axis}}[name=rel, {common}, xmin={lo}, xmax=1, ymin={lo}, ymax=1, title={{(a) Reliability}},"
           r" xlabel={stated confidence $c$}, ylabel={accuracy in bin}, legend to name=callegend, legend columns=3,"
           r" legend style={font=\scriptsize, draw=none, /tikz/every even column/.append style={column sep=5pt}}]"]
    for side in ("rarag", "sl"):
        pts = " ".join(f"({c:.3f},{a:.3f})" for c, a, _ in rel[side])
        out.append(rf"\addplot[{style[side]}, thick] coordinates {{{pts}}};")
    out.append(rf"\addplot[gray, dashed, domain={lo}:1, samples=2] {{x}};")
    out.append(r"\legend{{RA-RAG, vote share},{SL, $P(\hat y)$},{perfect calibration}}")
    # the vote's top bin, from the empty corner above the diagonal
    out.append(rf"\node[anchor=south west, font=\scriptsize, align=left] (silent) at (axis cs:{lo + 0.02:.2f},{top_acc + 0.02:.3f})"
               rf" {{silent errors: the vote says {top_c:.2f}\\on {100 * top_share:.0f}\% of queries, right {100 * top_acc:.0f}\%}};")
    out.append(rf"\draw[->, gray] (silent.south east) -- (axis cs:{top_c - 0.015:.3f},{top_acc:.3f});")
    out.append(r"\end{axis}")
    out.append(rf"\begin{{axis}}[name=rc, at={{(rel.east)}}, anchor=west, xshift=1.6cm, {common}, xmin=0, xmax=1, ymin=0,"
               r" title={(b) Risk vs.\ coverage}, xlabel={coverage (share of queries answered)}, ylabel={risk (share wrong)}]")
    for side in ("rarag", "sl"):
        pts = " ".join(f"({c:.3f},{r:.3f})" for c, r in curves[side])
        out.append(rf"\addplot[{style[side].split(',')[0]}, thick, no markers] coordinates {{{pts}}};")
    out.append(rf"\node[anchor=south west, font=\scriptsize, align=left] at (axis cs:0.02,{curves['rarag'][0][1] + 0.005:.3f})"
               rf" {{{100 * top_share:.0f}\% tied at {top_c:.2f}:\\cannot be ranked}};")
    out.append(rf"\addplot[blue!70!black, only marks, mark=*, mark size=2pt] coordinates {{({rule[0]:.3f},{rule[1]:.3f})}};")
    out.append(rf"\node[anchor=north west, font=\scriptsize] at (axis cs:{rule[0] + 0.01:.3f},{rule[1] - 0.03:.3f}) {{SL's rule}};")
    out.append(r"\end{axis}")
    out.append(r"\node[anchor=north] at ($(rel.outer south)!0.5!(rc.outer south)$) {\pgfplotslegendfromname{callegend}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


TRUST_RUN = ("NQ_paper", 0)   # the trust figure's run: a trajectory is one run's, not a mean
TRUST_PANELS = [   # (setup, side, title): one panel per setup and model family
    ("unreliable_k3", "sl", "(a) SL, 3 unreliable"), ("unreliable_k3", "rarag", "(b) RA-RAG, 3 unreliable"),
    ("attackers_k3", "sl", "(c) SL, 3 attackers"), ("attackers_k3", "rarag", "(d) RA-RAG, 3 attackers"),
]
TRUST_LINES = {"sl": [("sl_aged", "solid"), ("sl_plain", "dashed")],
               "rarag": [("rarag_window", "solid"), ("rarag_online", "dashed")]}
GROUPS = {"reliable": "green!50!black", "unreliable": "orange!90!black", "attacker": "red!80!black"}


def trust_group(source):
    kind, name = source.get("kind"), source["name"].lower()
    return "attacker" if kind == "adversary" or name.startswith("attacker") else \
        "unreliable" if name.startswith("unreliable") else "reliable"


def trust_figure(traces, every=10, window=50):
    """Four plots in a row: each group's mean trust before every test question, for the trust function that forgets
    (solid) and the one that does not (dashed), against the group's share of right answers among its kept answers
    over the last `window` questions (thin). Full text width."""
    common = (r"width=4.9cm, height=3.9cm, xmin=0, xmax=600, ymin=0, ymax=1, xtick={0,200,400,600},"
              r" ytick={0,0.5,1}, tick label style={font=\scriptsize}, label style={font=\scriptsize},"
              r" title style={font=\scriptsize}, ymajorgrids, grid style={gray!20}, no markers")
    out = [f"% Generated by rarag-sl-eval (python -m rarag_eval.paper): one run ({TRUST_RUN[0]}, seed {TRUST_RUN[1]}).",
           r"\begin{tikzpicture}"]
    prev = None
    for k, (setup, side, title) in enumerate(TRUST_PANELS):
        t = traces[setup]
        groups = {}
        for i, s in enumerate(t["sources"]):
            groups.setdefault(trust_group(s), []).append(i)
        place = rf"at={{({prev}.east)}}, anchor=west, xshift=0.55cm, yticklabels={{}}, " if prev else r"ylabel={trust}, "
        legend = (r"legend to name=trustlegend, legend columns=6, legend style={font=\scriptsize, draw=none,"
                  r" /tikz/every even column/.append style={column sep=5pt}}, ") if k == 0 else ""
        out.append(rf"\begin{{axis}}[name=p{k}, {place}{legend}{common}, title={{{title}}}, xlabel={{test query}}]")
        for g, color in GROUPS.items():
            if g not in groups:
                continue
            for line, dash in TRUST_LINES[side]:
                pts = " ".join(f"({q},{sum(t[line][q][i] for i in groups[g]) / len(groups[g]):.3f})"
                               for q in range(0, len(t[line]), every))
                out.append(rf"\addplot[{color}, {dash}, thick, forget plot] coordinates {{{pts}}};")
            acc = []
            for q in range(window, len(t["correct"]) + 1, every):
                kept = [c for row in t["correct"][q - window:q] for i in groups[g] if (c := row[i]) is not None]
                acc.append(f"({q},{sum(kept) / len(kept):.3f})" if kept else "")
            out.append(rf"\addplot[{color}, opacity=0.45, forget plot] coordinates {{{' '.join(acc)}}};")
        if k == 0:
            for g, color in GROUPS.items():
                out.append(rf"\addlegendimage{{{color}, thick}}\addlegendentry{{{g}}}")
            out.append(r"\addlegendimage{black, solid, thick}\addlegendentry{ageing (SL), window (RA-RAG)}"
                       r"\addlegendimage{black, dashed, thick}\addlegendentry{no forgetting (online)}"
                       r"\addlegendimage{black, opacity=0.45}\addlegendentry{share of right answers}")
        out.append(r"\end{axis}")
        prev = f"p{k}"
    out.append(r"\node[anchor=north] at ($(p1.outer south east)!0.5!(p2.outer south west)$) {\pgfplotslegendfromname{trustlegend}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", required=True, help="the paper's folder (figures/ and tables/ are written in it)")
    args = ap.parse_args()
    load_config(args.config)
    _, index, rows = collect.collect()
    cal = calibration.summary(DATASETS, SEEDS)
    for sub in ("figures", "tables"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    head = "% Generated by rarag-sl-eval (python -m rarag_eval.paper) from the saved results; do not edit by hand.\n"
    with open(os.path.join(args.out, "tables", "tab-results.tex"), "w") as f:
        f.write(head + results_table(index))
    with open(os.path.join(args.out, "tables", "tab-decision.tex"), "w") as f:
        f.write(head + decision_table(cal))
    with open(os.path.join(args.out, "figures", "fig-attackers.tex"), "w") as f:
        f.write(attackers_figure(rows))
    rel = calibration.reliability(DATASETS, SEEDS, CALIBRATION_SETUP)
    curves = calibration.risk_coverage(DATASETS, SEEDS, CALIBRATION_SETUP)
    sl = cal[CALIBRATION_SETUP]["sl"]
    with open(os.path.join(args.out, "figures", "fig-calibration.tex"), "w") as f:
        f.write(calibration_figure(rel, curves, (sl["coverage"], 1 - sl["answered_accuracy"])))
    traces = {setup: os.path.join(trace.OUT, f"{TRUST_RUN[0]}__s{TRUST_RUN[1]}__{setup}.json.gz")
              for setup in dict.fromkeys(s for s, _, _ in TRUST_PANELS)}
    if all(os.path.exists(f) for f in traces.values()):   # python -m rarag_eval.trace, from a backup's answers
        with open(os.path.join(args.out, "figures", "fig-trust.tex"), "w") as f:
            f.write(trust_figure({s: json.load(gzip.open(f)) for s, f in traces.items()}))
    else:
        print("fig-trust.tex not written: no traces (python -m rarag_eval.trace)")
    for setup, v in cal.items():
        print(setup, {k: round(100 * x, 1) if isinstance(x, float) else x for k, x in v["sl"].items()},
              "| vote", {k: round(100 * x, 1) if isinstance(x, float) else x for k, x in v["rarag"].items()},
              "| AURC SL better in", v["aurc_sl_better"], "of", v["n"])
    for side in ("sl", "rarag"):   # the numbers the text quotes from the figure
        print(CALIBRATION_SETUP, side, "bins (c, accuracy, share):", [tuple(round(x, 3) for x in b) for b in rel[side]])
        print(CALIBRATION_SETUP, side, "risk at coverage 0.1/0.5/end:",
              [(round(c, 2), round(r, 3)) for c, r in curves[side] if abs(c - 0.1) < 1e-6 or abs(c - 0.5) < 1e-6]
              + [tuple(round(x, 3) for x in curves[side][-1])])
    print("written: tables/tab-results.tex, tables/tab-decision.tex, figures/fig-attackers.tex, figures/fig-calibration.tex")


if __name__ == "__main__":
    main()
