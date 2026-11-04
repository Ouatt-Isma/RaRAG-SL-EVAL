"""The exhibits of the paper's evaluation, written from the saved results into the paper's own folders, so every
number in them comes from here:

  tables/tab-decision.tex     SL's own decision rule against RA-RAG's vote with a window and frozen: coverage, accuracy,
                              calibration error and how well it ranks the errors last (rarag_eval/calibration.py)
  figures/fig-attackers.tex   accuracy and attack success against the number of attackers among 9 sources (pgfplots)
  figures/fig-calibration.tex the two confidences under attack: reliability diagram and risk-coverage curve (pgfplots)
  figures/fig-trust.tex       the trust in each kind of source with 4 attackers, one run (from results/traces/,
                              python -m rarag_eval.trace)

It also prints the main results the text quotes: our model against RA-RAG, each a paired comparison on the same
answers: both means over every run (3 datasets x 3 seeds), the mean difference with its 95% t interval, and the runs
in which ours is ahead.

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


def main_results(ix):
    """The main results the text quotes, one line each: RA-RAG's and our mean, the mean paired difference with its
    95% interval, and the runs in which ours is ahead."""
    lines = []
    for group, items in MAIN:
        lines.append(group.replace("\\ ", " "))
        for label, comparison in items:
            if a := compare(comparison, ix):
                ci = f" [{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}]" if a["lo"] is not None else ""
                lines.append(f"  {label}: RA-RAG {100 * a['base']:.1f}, ours {100 * a['ours']:.1f},"
                             f" diff {100 * a['mean']:+.1f}{ci}, {a['holds']}/{a['n']} runs")
    return "\n".join(lines)


SIDES = [("sl", "SL"), ("rarag", "Win."), ("frozen", "Froz.")]


def decision_table(cal):
    """SL's rule against RA-RAG's vote with a window and with its trust frozen, per setup; the best value of each
    measure in bold."""
    pct = lambda x: "--" if x is None else f"{100 * x:.1f}"
    measures = [("Accuracy", lambda m, side: m[side]["answered_accuracy" if side == "sl" else "same_coverage_accuracy"], max, pct),
                ("ECE", lambda m, side: m[side]["ece"], min, pct),
                ("AURC", lambda m, side: m[side]["aurc"], min, pct)]
    n = 2 + 3 * len(measures)
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for name, *_ in measures)
    rules = "".join(rf"\cmidrule({'lr' if k < len(measures) - 1 else 'l'}){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(measures)))
    lines = [rf"\begin{{tabular}}{{@{{}}l{'r' * (n - 1)}@{{}}}}", r"\toprule", rf" & & {head} \\", rules,
             "Setup & Cov. & " + " & ".join(" & ".join(label for _, label in SIDES) for _ in measures) + r" \\", r"\midrule"]
    for setup, label in SETUP_SHORT.items():
        if setup not in cal:
            continue
        m = cal[setup]
        cells = [label, pct(m["sl"]["coverage"])]
        for _, get, best, fmt in measures:
            vals = [get(m, side) for side, _ in SIDES]
            shown = [fmt(v) for v in vals]
            known = [v for v in vals if v is not None]
            top = fmt(best(known)) if known else None
            bold = lambda x: rf"$\mathbf{{{x[1:-1]}}}$" if x.startswith("$") else rf"\textbf{{{x}}}"
            cells += [bold(x) if x == top else x for x in shown]
        lines.append(" & ".join(cells) + r" \\")
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
    print(main_results(index))
    with open(os.path.join(args.out, "tables", "tab-decision.tex"), "w") as f:
        f.write(head + decision_table(cal))
    with open(os.path.join(args.out, "figures", "fig-attackers.tex"), "w") as f:
        f.write(attackers_figure(rows))
    rel = calibration.reliability(DATASETS, SEEDS, CALIBRATION_SETUP)
    curves = calibration.risk_coverage(DATASETS, SEEDS, CALIBRATION_SETUP)
    sl = cal[CALIBRATION_SETUP]["sl"]
    with open(os.path.join(args.out, "figures", "fig-calibration.tex"), "w") as f:
        f.write(calibration_figure(rel, curves, (sl["coverage"], 1 - sl["answered_accuracy"])))
    traced = trace.path(f"{trace.RUN[0]}.json", trace.RUN[1], CALIBRATION_SETUP)
    if os.path.exists(traced):   # the trust behind the calibration figure's setup, one run
        with open(os.path.join(args.out, "figures", "fig-trust.tex"), "w") as f:
            f.write(trace.figure({CALIBRATION_SETUP: json.load(gzip.open(traced))}, width="7.4cm", height="4.6cm"))
    else:
        print(f"fig-trust.tex not written: no trace of {CALIBRATION_SETUP} (python -m rarag_eval.trace)")
    for setup, v in cal.items():
        print(setup, {k: round(100 * x, 1) if isinstance(x, float) else x for k, x in v["sl"].items()},
              "| vote", {k: round(100 * x, 1) if isinstance(x, float) else x for k, x in v["rarag"].items()},
              "| AURC SL better in", v["aurc_sl_better"], "of", v["n"])
    for side in ("sl", "rarag"):   # the numbers the text quotes from the figure
        print(CALIBRATION_SETUP, side, "bins (c, accuracy, share):", [tuple(round(x, 3) for x in b) for b in rel[side]])
        print(CALIBRATION_SETUP, side, "risk at coverage 0.1/0.5/end:",
              [(round(c, 2), round(r, 3)) for c, r in curves[side] if abs(c - 0.1) < 1e-6 or abs(c - 0.5) < 1e-6]
              + [tuple(round(x, 3) for x in curves[side][-1])])
    print("written: tables/tab-decision.tex, figures/fig-attackers.tex, figures/fig-calibration.tex, figures/fig-trust.tex")


if __name__ == "__main__":
    main()
