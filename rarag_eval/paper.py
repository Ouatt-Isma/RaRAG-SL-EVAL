"""The exhibits of the paper's evaluation, written from the saved results into the paper's own folders, so every
number in them comes from here:

  tables/tab-decision.tex     SL's own decision rule against RA-RAG's vote with a window and frozen: coverage, accuracy,
                              calibration error and how well it ranks the errors last (rarag_eval/calibration.py)
  tables/tab-paced.tex        attacks that succeed by how often the attackers betray
  figures/fig-attackers.tex   accuracy and attack success against the number of attackers among 9 sources (pgfplots)
  figures/fig-calibration.tex the two confidences under attack: reliability diagram and risk-coverage curve (pgfplots)
  figures/fig-warmup.tex      accuracy of the frozen trust functions against the size of the warm-up (pgfplots)
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
WINDOW = ("aged", "rarag_window", "RA-RAG with a window")
RULE = ("answer_aged", "sl_online", "SL's own rule")
EVERY = ("answer_aged", "rarag_all", "RA-RAG frozen, every source")
EQUAL = ("plain", "equal", "equal weights")
ONE_SCORE = ("dom_global_wfull", "sl_online", "SL, one score")
PACES = [("adaptive_r10", "10\\%"), ("adaptive_r20", "20\\%"), ("adaptive_r30", "30\\%"), ("burst_r20", "bursts")]


def paced(what, base, method, direction):
    """One comparison at every pace of betrayal, as rows of MAIN."""
    return [(f"{'Accuracy' if what == 'acc' else 'Attacks that succeed'}, {label}",
             Claim("", "", [setup], what, base, method, direction)) for setup, label in PACES]


def domain(base, method, direction=+1):
    return Claim("", "", ["experts"], "acc", base, method, direction)


# The main results, groups in reading order of the evaluation. A row is (label, comparison): a claim (its id in
# claims.CLAIMS, or one of its own), or ("cal", setup, measure, direction[, side]), a measure of calibration.unit with
# SL's rule against RA-RAG's vote (side, a vote of calibration.unit, by default the one with a window); measure may be
# a pair (SL's, the vote's).
MAIN = [
    (r"A patient adversary: SL with ageing vs.\ RA-RAG's frozen trust", [
        (r"Attacks that succeed, $a{=}2$", "3"),
        (r"Attacks that succeed, $a{=}3$", Claim("m1", "", ["attackers_k3"], "attack", FROZEN, AGED, -1)),
        (r"Accuracy, $a{=}3$", Claim("m2", "", ["attackers_k3"], "acc", FROZEN, AGED, +1)),
        (r"Accuracy, late switch", Claim("m3", "", ["switch"], "acc", FROZEN, AGED, +1)),
    ]),
    (r"Scarce evidence: the opinion vs.\ the rate", [
        (r"Accuracy, 5 warm-up queries", "8"),
    ]),
    (r"Honest confidence, $a{=}4$: SL's rule vs.\ RA-RAG's vote", [
        (r"Calibration error (ECE)", ("cal", "attackers_k4", "ece", -1)),
        (r"NAURC", ("cal", "attackers_k4", "naurc", -1)),
        (r"Right when $c\ge0.9$", ("cal", "attackers_k4", "hi_accuracy", +1)),
        (r"Attacks that succeed", "16"),
    ]),
    (r"Domain experts: SL per domain vs.\ RA-RAG's one trust", [
        (r"Accuracy, LLM label", Claim("m4", "", ["experts"], "acc", FROZEN, ("dom_domain_wfull", "sl_online", ""), +1)),
        (r"Accuracy, nearest centroid",
         Claim("m5", "", ["experts"], "acc", FROZEN, ("dom_centroid_wfull", "sl_online", ""), +1)),
    ]),
    (r"Paced betrayal: SL's rule vs.\ SL's ageing in RA-RAG's vote", paced("attack", AGED, RULE, -1)),
    (r"Paced betrayal: SL's rule vs.\ RA-RAG's vote with a window", paced("attack", WINDOW, RULE, -1)),
    (r"Paced betrayal: SL's rule vs.\ RA-RAG's frozen vote over every source", paced("attack", EVERY, RULE, -1)),
    (r"Paced betrayal: the same window, SL vs.\ RA-RAG", paced("attack", WINDOW, ("window", "sl_online", ""), -1)
     + paced("acc", WINDOW, ("window", "sl_online", ""), +1)),
    (r"Paced betrayal: SL's ageing vs.\ equal weights", paced("acc", EQUAL, AGED, +1)),
    (r"Paced betrayal: confidence, SL's rule vs.\ RA-RAG's vote with a window",
     [(f"{name}, {label}", ("cal", setup, measure, sign)) for setup, label in PACES
      for name, measure, sign in (("ECE", "ece", -1), (r"Right when $c\ge0.9$", "hi_accuracy", +1))]),
    (r"Paced betrayal: confidence, SL's rule vs.\ RA-RAG's frozen vote over every source",
     [(f"{name}, {label}", ("cal", setup, measure, sign, "every")) for setup, label in [*PACES, ("attackers_k4", "4 of 9")]
      for name, measure, sign in (("ECE", "ece", -1), (r"Right when $c\ge0.9$", "hi_accuracy", +1),
                                  ("Accuracy at our coverage", ("answered_accuracy", "same_coverage_accuracy"), +1))]),
    (r"Fair domain test: independent labels (B) on the experts", [
        (r"Soft B vs.\ RA-RAG frozen", domain(FROZEN, ("dom_soft_b_wfull", "sl_online", ""))),
        (r"Soft B vs.\ one score", domain(ONE_SCORE, ("dom_soft_b_wfull", "sl_online", ""))),
        (r"Labels B vs.\ one score", domain(ONE_SCORE, ("dom_domain_b_wfull", "sl_online", ""))),
        (r"Labels A vs.\ labels B", domain(("dom_domain_b_wfull", "sl_online", ""), ("dom_domain_wfull", "sl_online", ""))),
        (r"Soft A vs.\ soft B", domain(("dom_soft_b_wfull", "sl_online", ""), ("dom_soft_wfull", "sl_online", ""))),
        (r"Labels B vs.\ RA-RAG per domain", domain(("dom_domain_b_wfull", "rarag_topic_online", ""), ("dom_domain_b_wfull", "sl_online", ""))),
        (r"Soft B vs.\ RA-RAG per domain", domain(("dom_soft_b_wfull", "rarag_topic_online", ""), ("dom_soft_b_wfull", "sl_online", ""))),
    ]),
]
PACED_ROWS = [("plain", "rarag", "RA-RAG, frozen"), ("aged", "rarag_window", "RA-RAG, window"),
              ("answer_aged", "rarag_all", "RA-RAG, frozen, all 9"), ("aged", "sl_online", "SL, ageing"),
              ("plain", "equal", "Equal weights"), ("answer_aged", "sl_online", "Our rule")]
CALIBRATION_SETUP = "attackers_k4"   # the figure's setup: the attackers are close to a majority
DATASETS = ["NQ_paper", "TQA_paper", "HotpotQA_paper"]
SEEDS = [0, 1, 2]
SETUP_SHORT = {"unreliable_k3": "3 unreliable", "mix5": "honest mix", "attackers_k1": "1 attacker",
               "attackers_k2": "2 attackers", "attackers_k3": "3 attackers", "attackers_k4": "4 attackers",
               "switch": "late switch"}
LINES = [  # the attackers figure: (analysis, row, legend, style). Colour: the model; line: its memory. Lines that
    # coincide stay apart: alternating dashes, a thin line on a thick one, nested marks. The legend reads across
    ("plain", "rarag", "RA-RAG, frozen", "red!75!black, dash pattern=on 3pt off 3pt, mark=o, mark size=2.4pt"),
    ("aged", "rarag_window", "RA-RAG, window", "red!75!black, line width=1.3pt, mark=o, mark size=2.4pt"),
    ("plain", "equal", "equal weights", "gray, dashdotted, mark=triangle, mark size=2pt"),
    ("plain", "sl_offline", "SL, frozen", "blue!70!black, dash pattern=on 3pt off 3pt, dash phase=3pt, mark=square, mark size=1.4pt"),
    ("aged", "sl_online", "SL, ageing", "blue!70!black, line width=0.6pt, mark=square, mark size=1.4pt"),
    ("plain", "oracle", "oracle", "black, dotted, mark=x, mark size=1.5pt"),
]
WARMUPS = (5, 10, 20, 200)   # the warm-up figure's sizes: the honest mix's short warm-ups, then the full one


def num(x, digits=1, sign=True):
    return "--" if x is None else (f"${x:+.{digits}f}$" if sign else f"{x:.{digits}f}").replace("-", "-")


def compare(comparison, ix):
    """One main result over every run: {"base", "ours"} means and the aggregate of the paired differences."""
    if isinstance(comparison, tuple):
        _, setup, measure, direction, *side = comparison
        side = side[0] if side else "rarag"
        mine, theirs = measure if isinstance(measure, tuple) else (measure, measure)
        units = [u for d in DATASETS for s in SEEDS if (u := calibration.unit(d, s, setup))
                 and u["sl"][mine] is not None and u[side][theirs] is not None]
        base, ours = [u[side][theirs] for u in units], [u["sl"][mine] for u in units]
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


def paced_counts(rows, ix):
    """Per pace of betrayal, the numbers the text quotes: the attacks that land per run against SL's ageing in RA-RAG's
    vote, and the share that would succeed if SL's rule answered with its top candidate instead of abstaining."""
    lines = ["Paced betrayal: attacks landed per run, and the attacks that succeed if SL's rule never abstained"]
    for setup, label in [*PACES, ("attackers_k3", "50\\%"), ("attackers_k4", "4 of 9")]:
        landed = [r["attack"] * r["n_target"] for r in rows if r["setup"] == setup and r["analysis"] == "aged"
                  and r["row"] == "sl_online" and r.get("attack") is not None]
        forced = []
        for d in DATASETS:
            for s in SEEDS:
                rec = ix[(d, s, setup, "answer_aged")]
                top = [t for t, q in zip(rec["rows"]["sl_online"]["top"], rec["questions"]) if q["target"]]
                forced.append(top.count("a") / len(top))
        lines.append(f"  {label}: {statistics.mean(landed):.1f} attacks per run, {100 * statistics.mean(forced):.1f}% without abstaining")
    return "\n".join(lines)


SIDES = [("sl", "SL"), ("rarag", "Win."), ("frozen", "Froz.")]


def decision_table(cal):
    """SL's rule against RA-RAG's vote with a window and with its trust frozen, per setup; the best value of each
    measure in bold."""
    pct = lambda x: "--" if x is None else f"{100 * x:.1f}"
    measures = [("Coverage", lambda m, side: m[side]["coverage"], max, pct),
                ("Accuracy", lambda m, side: m[side]["answered_accuracy" if side == "sl" else "same_coverage_accuracy"], max, pct),
                ("ECE", lambda m, side: m[side]["ece"], min, pct),
                ("NAURC", lambda m, side: m[side]["naurc"], min, pct)]
    n = 1 + 3 * len(measures)
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for name, *_ in measures)
    rules = "".join(rf"\cmidrule({'lr' if k < len(measures) - 1 else 'l'}){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(measures)))
    lines = [rf"\begin{{tabular}}{{@{{}}l{'r' * (n - 1)}@{{}}}}", r"\toprule", rf" & {head} \\", rules,
             "Setup & " + " & ".join(" & ".join(label for _, label in SIDES) for _ in measures) + r" \\", r"\midrule"]
    for setup, label in SETUP_SHORT.items():
        if setup not in cal:
            continue
        m = cal[setup]
        cells = [label]
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


def paced_table(rows):
    """Attacks that succeed by how often the attackers betray, the lowest of each column in bold, and the coverage
    of our rule, which alone abstains."""
    setups = [setup for setup, _ in PACES] + ["attackers_k3", "attackers_k4"]

    def mean(setup, analysis, row, metric):
        xs = [r[metric] for r in rows if r["setup"] == setup and r["analysis"] == analysis and r["row"] == row
              and r.get(metric) is not None]
        return 100 * statistics.mean(xs)
    table = [[mean(s, analysis, row, "attack") for s in setups] for analysis, row, _ in PACED_ROWS]
    lows = [min(col) for col in zip(*table)]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{3 of 9, betraying on} & 4 of 9 \\", r"\cmidrule(lr){2-6}\cmidrule(l){7-7}",
             r"Trust & 10\% & 20\% & 30\% & bursts & 50\% & 50\% \\", r"\midrule"]
    for (_, _, label), vals in zip(PACED_ROWS, table):
        cells = [rf"\textbf{{{v:.1f}}}" if v == low else f"{v:.1f}" for v, low in zip(vals, lows)]
        lines.append(" & ".join([label, *cells]) + r" \\")
    cover = [f"{mean(s, 'answer_aged', 'sl_online', 'coverage'):.1f}" for s in setups]
    lines += [r"\midrule", " & ".join(["Our rule, coverage", *cover]) + r" \\", r"\bottomrule", r"\end{tabular}"]
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
            out.append(rf"\addplot[thick, {style}, mark options={{solid, thin}}] coordinates {{{pts}}};")
        if name == "att":
            out.append(r"\legend{" + ",".join("{" + label + "}" for _, _, label, _ in LINES) + "}")   # labels hold commas
        out.append(r"\end{axis}")
    out.append(r"\node[anchor=north] at ($(acc.outer south)!0.5!(att.outer south)$) {\pgfplotslegendfromname{attlegend}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def warmup_figure(rows):
    """Accuracy of each frozen trust function against the size of the warm-up, on the honest mix: a short warm-up is
    a random subset of the full one, mean over its draws and every run. Same lines as the attackers figure."""
    def mean(size, row):
        xs = [r["accuracy"] for r in rows if r["setup"] == "mix5" and r["row"] == row and r.get("accuracy") is not None
              and (r["analysis"] == "plain" if size == WARMUPS[-1] else r["analysis"].startswith(f"warm{size}_d"))]
        return 100 * statistics.mean(xs)
    lines = [(row, label, style) for analysis, row, label, style in LINES if analysis == "plain"]
    out = ["% Generated by rarag-sl-eval (python -m rarag_eval.paper): mean over 3 datasets x 3 seeds and the draws of each warm-up.",
           r"\begin{tikzpicture}",
           r"\begin{axis}[name=warm, width=7.4cm, height=3.2cm, xmode=log, log ticks with fixed point, xtick={"
           + ",".join(str(w) for w in WARMUPS) + r"}, minor xtick={}, xlabel={warm-up queries}, ylabel={accuracy (\%)},"
           r" tick label style={font=\scriptsize}, label style={font=\scriptsize}, ymajorgrids, grid style={gray!20},"
           r" mark size=1.4pt, legend to name=warmlegend, legend columns=4, legend style={font=\scriptsize, draw=none,"
           r" /tikz/every even column/.append style={column sep=5pt}}]"]
    for row, _, style in lines:
        pts = " ".join(f"({w},{mean(w, row):.1f})" for w in WARMUPS)
        out.append(rf"\addplot[thick, {style}, mark options={{solid, thin}}] coordinates {{{pts}}};")
    out.append(r"\legend{" + ",".join("{" + label + "}" for _, label, _ in lines) + "}")
    out += [r"\end{axis}", r"\node[anchor=north] at (warm.outer south) {\pgfplotslegendfromname{warmlegend}};", r"\end{tikzpicture}"]
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
    print(paced_counts(rows, index))
    with open(os.path.join(args.out, "tables", "tab-decision.tex"), "w") as f:
        f.write(head + decision_table(cal))
    with open(os.path.join(args.out, "tables", "tab-paced.tex"), "w") as f:
        f.write(head + paced_table(rows))
    with open(os.path.join(args.out, "figures", "fig-attackers.tex"), "w") as f:
        f.write(attackers_figure(rows))
    rel = calibration.reliability(DATASETS, SEEDS, CALIBRATION_SETUP)
    curves = calibration.risk_coverage(DATASETS, SEEDS, CALIBRATION_SETUP)
    sl = cal[CALIBRATION_SETUP]["sl"]
    with open(os.path.join(args.out, "figures", "fig-calibration.tex"), "w") as f:
        f.write(calibration_figure(rel, curves, (sl["coverage"], 1 - sl["answered_accuracy"])))
    with open(os.path.join(args.out, "figures", "fig-warmup.tex"), "w") as f:
        f.write(warmup_figure(rows))
    traced = trace.path(f"{trace.RUN[0]}.json", trace.RUN[1], CALIBRATION_SETUP)
    if os.path.exists(traced):   # the trust behind the calibration figure's setup, one run
        with open(os.path.join(args.out, "figures", "fig-trust.tex"), "w") as f:
            f.write(trace.figure({CALIBRATION_SETUP: json.load(gzip.open(traced))}, width="7.4cm", height="4.6cm"))
    else:
        print(f"fig-trust.tex not written: no trace of {CALIBRATION_SETUP} (python -m rarag_eval.trace)")
    for setup, v in cal.items():
        print(setup, {k: round(100 * x, 1) if isinstance(x, float) else x for k, x in v["sl"].items()},
              "| vote", {k: round(100 * x, 1) if isinstance(x, float) else x for k, x in v["rarag"].items()},
              "| NAURC SL better in", v["naurc_sl_better"], "of", v["n"])
    for side in ("sl", "rarag"):   # the numbers the text quotes from the figure
        print(CALIBRATION_SETUP, side, "bins (c, accuracy, share):", [tuple(round(x, 3) for x in b) for b in rel[side]])
        print(CALIBRATION_SETUP, side, "risk at coverage 0.1/0.5/end:",
              [(round(c, 2), round(r, 3)) for c, r in curves[side] if abs(c - 0.1) < 1e-6 or abs(c - 0.5) < 1e-6]
              + [tuple(round(x, 3) for x in curves[side][-1])])
    print("written: tables/tab-decision.tex, tables/tab-paced.tex, figures/fig-attackers.tex, figures/fig-calibration.tex, figures/fig-warmup.tex, figures/fig-trust.tex")


if __name__ == "__main__":
    main()
