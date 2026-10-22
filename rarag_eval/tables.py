"""LaTeX tables (booktabs) for the paper, written to paper/tables/*.tex, each a whole table environment
with a caption and a label. Accuracies and rates in percent; differences in percentage points. With
several seeds a cell shows the mean and, in a smaller size, the spread (standard deviation) over seeds."""
import math
import os
import statistics

from .thesis import ROOT

TABLES = os.path.join(ROOT, "paper", "tables")
DATASET_LABEL = {"NQ_paper": "NQ", "TQA_paper": "TriviaQA", "HotpotQA_paper": "HotpotQA", "LIAR_RAW": "LIAR-RAW", "RAWFC": "RAWFC"}
ROW_LABEL = {
    ("plain", "rarag"): "RA-RAG, frozen (paper)", ("plain", "rarag_online"): "RA-RAG online",
    ("aged", "rarag_window"): "RA-RAG, window of 100", ("plain", "sl_offline"): "SL, frozen",
    ("plain", "sl_online"): "SL online, no forgetting", ("aged", "sl_online"): r"SL, aging $\lambda=0.97$",
    ("window", "sl_online"): "SL, window of 100", ("detector", "sl_online"): r"SL, aging + detector",
    ("plain", "oracle"): "Oracle (warm-up truth)", ("plain", "equal"): "Equal weights",
}
ROBUST_ROWS = [("plain", "rarag"), ("plain", "rarag_online"), ("aged", "rarag_window"), ("plain", "sl_offline"),
               ("plain", "sl_online"), ("aged", "sl_online"), ("window", "sl_online"), ("plain", "equal")]
SETUP_LABEL = {"attackers_k1": "1 of 9", "attackers_k2": "2 of 9", "attackers_k3": "3 of 9", "attackers_k4": "4 of 9",
               "switch": "switch", "unreliable_k3": "3 of 9", "unreliable_k7": "7 of 9", "mix5": "5 honest",
               "experts": "experts"}
SETUP_LONG = {"unreliable_k3": "3 unreliable of 9", "unreliable_k7": "7 unreliable of 9", "mix5": "5 honest sources",
              "attackers_k1": "1 attacker of 9", "attackers_k2": "2 attackers of 9", "attackers_k3": "3 attackers of 9",
              "attackers_k4": "4 attackers of 9", "switch": "3 attackers, late switch", "experts": "domain experts"}
PLACEMENT_LABEL = {"global": "No domains", "domain": "LLM's labels", "centroid": "Nearest centroid", "soft": "Soft weights"}


def esc(text):
    return (str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("_", r"\_"))


def dlabel(d):
    return DATASET_LABEL.get(d, esc(d))


def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}"


def pts(x, digits=1):
    """A signed difference in points, with a real minus sign."""
    return "--" if x is None else f"${100 * x:+.{digits}f}$"


def cell(values, digits=1):
    """Mean of a metric over seeds, with the spread as a small subscript when there are several."""
    xs = [v for v in values if v is not None]
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{100 * xs[0]:.{digits}f}"
    return f"{100 * statistics.mean(xs):.{digits}f}\\,{{\\scriptsize$\\pm${100 * statistics.stdev(xs):.{digits}f}}}"


def wrap(tabular, caption, label, small=True, wide=False):
    env = "table*" if wide else "table"
    # no placement option: a paper sets its own; the results document keeps each table where it is input
    return (f"\\begin{{{env}}}\n\\centering\n" + ("\\small\n" if small else "") + tabular
            + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


def write(name, text):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, name + ".tex"), "w") as f:
        f.write(text)


def values(rows, **where):
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def claims_table(summary, datasets):
    head = " & ".join(dlabel(d) for d in datasets)
    lines = [r"\begin{tabular}{@{}l p{6.2cm} " + "r" * len(datasets) + r" r c@{}}", r"\toprule",
             rf"\# & Result (method vs baseline) & {head} & All, 95\% CI & Holds \\", r"\midrule"]
    for s in summary:
        c = s["claim"]
        unit = "" if c.what != "overrate" else " (score)"
        per = " & ".join(pts(s["per_dataset"][d]["mean"]) for d in datasets)
        a = s["all"]
        ci = "" if a["lo"] is None else f" [{pts(a['lo'])}, {pts(a['hi'])}]"
        holds = f"{a['holds']}/{a['n']}" if a["n"] else "--"
        what = {"acc": "accuracy", "attack": "attacks that succeed", "overrate": "estimate minus truth"}[c.what]
        text = f"{esc(c.title)}: {esc(c.method[2])} vs {esc(c.baseline[2])}, {what}{unit}"
        lines.append(f"{esc(c.id)} & {text} & {per} & {pts(a['mean'])}{ci} & {holds} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    caption = (r"The first evaluation series' results across datasets and seeds. Each cell is the method's "
               r"difference from the baseline in percentage points (accuracy or attacks that succeed), averaged over "
               r"seeds; ``All'' averages every dataset and seed, with a 95\% $t$ interval when there are several; "
               r"``Holds'' counts the runs where the difference goes the expected way (for a tie: within one point).")
    write("claims", wrap("\n".join(lines), caption, "tab:claims", wide=True))


def robustness_tables(rows, datasets, setups, metric, name, caption):
    cols = [s for s in setups if values(rows, setup=s)]
    body = []
    for d in datasets:
        if not any(values(rows, dataset=d, setup=s) for s in cols):
            continue
        body.append(rf"\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}{{\textit{{{dlabel(d)}}}}} \\")
        for analysis, row in ROBUST_ROWS:
            cells = [cell([r.get(metric) for r in values(rows, dataset=d, setup=s, analysis=analysis, row=row)]) for s in cols]
            if all(c == "--" for c in cells):
                continue
            body.append(f"\\quad {ROW_LABEL[(analysis, row)]} & " + " & ".join(cells) + r" \\")
    if not cols or not body:
        body = [r"\multicolumn{2}{l}{No results yet.} \\"]
        cols = cols or ["--"]
    head = " & ".join(SETUP_LABEL.get(s, esc(s)) for s in cols)
    tab = "\n".join([r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule", rf"Trust & {head} \\", r"\midrule",
                     *body, r"\bottomrule", r"\end{tabular}", ""])
    write(name, wrap(tab, caption, f"tab:{name}"))


def domains_table(rows, datasets):
    body = []
    for p, label in PLACEMENT_LABEL.items():
        cells = []
        for d in datasets:
            sl = cell([r["accuracy"] for r in values(rows, dataset=d, setup="experts", analysis=f"dom_{p}_wfull", row="sl_online")])
            ra = cell([r["accuracy"] for r in values(rows, dataset=d, setup="experts", analysis=f"dom_{p}_wfull",
                                                       row="rarag_topic_online" if p != "global" else "rarag_online")])
            cells += [sl, ra]
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{dlabel(d)}}}" for d in datasets)
    sub = " & ".join("SL & RA-RAG" for _ in datasets)
    tab = "\n".join([r"\begin{tabular}{@{}l" + "rr" * len(datasets) + "@{}}", r"\toprule", rf"Domain of a question & {head} \\",
                     rf" & {sub} \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""])
    write("domains", wrap(tab, r"Domain experts (right 90\% of the time in their domain, 30\% elsewhere): accuracy with "
                               r"trust learned per domain online, by how each question is placed among the domains, full "
                               r"warm-up. RA-RAG's estimator runs per domain on each question's top domain.", "tab:domains"))


def answering_table(rows, setups):
    cols = ["coverage", "answered_accuracy", "rarag_same_coverage", "conflict_gold", "unknown_would_be_wrong", "attack"]
    body = []
    for s in setups:
        sel = values(rows, setup=s, analysis="answer_aged", row="sl_online")
        if not sel:
            continue
        cells = [cell([r.get(c) for r in sel]) for c in cols]
        twin = values(rows, setup=s, analysis="aged", row="sl_online")
        cells.append(cell([r["attack"] for r in twin]) if any(r["attack"] is not None for r in twin) else "--")
        body.append(f"{SETUP_LONG.get(s, esc(s))} & " + " & ".join(cells) + r" \\")
    if not body:
        body = [r"\multicolumn{8}{l}{No results yet.} \\"]
    tab = "\n".join([r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
                     r"Setup & Answers & Right & RA-RAG, same cov. & Conflict right & Unknown wrong & \multicolumn{2}{c}{Attacks succeed} \\",
                     r" & & & & & & SL & RA-RAG \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""])
    write("answering", wrap(tab, r"SL's own answering (answer, \emph{unknown} or \emph{conflict}) with its online trust and "
                                 r"aging: share of questions answered, accuracy on those, RA-RAG's accuracy on its most "
                                 r"confident questions at the same coverage and trust, conflicts whose two answers include "
                                 r"the right one, \emph{unknown} questions whose top answer would have been wrong, and attacks "
                                 r"that succeed with SL's answering and with RA-RAG's on the same trust. Means over all "
                                 r"datasets and seeds.", "tab:answering", wide=True))


def warmup_table(rows, datasets, sizes):
    body = []
    for d in datasets:
        cells = []
        for size in sizes:
            sl = [r["accuracy"] for r in values(rows, dataset=d, setup="mix5", row="sl_offline") if r["analysis"].startswith(f"warm{size}_d")]
            ra = [r["accuracy"] for r in values(rows, dataset=d, setup="mix5", row="rarag") if r["analysis"].startswith(f"warm{size}_d")]
            cells.append(pts(statistics.mean(sl) - statistics.mean(ra)) if sl and ra else "--")
        body.append(f"{dlabel(d)} & " + " & ".join(cells) + r" \\")
    head = " & ".join(str(s) for s in sizes)
    tab = "\n".join([r"\begin{tabular}{@{}l" + "r" * len(sizes) + "@{}}", r"\toprule", rf"Dataset & {head} \\", r"\midrule",
                     *body, r"\bottomrule", r"\end{tabular}", ""])
    write("warmup", wrap(tab, r"Few warm-up questions (five honest sources, reliabilities 0.53--0.91): SL's frozen trust minus "
                              r"RA-RAG's frozen estimate, accuracy in points, by number of warm-up questions; mean over random "
                              r"warm-up draws and seeds.", "tab:warmup"))


def status_table(index, cfg):
    datasets = [os.path.splitext(os.path.basename(d))[0] for d in cfg["datasets"]]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cfg["seeds"]) + "@{}}", r"\toprule",
             "Dataset & " + " & ".join(f"seed {s}" for s in cfg["seeds"]) + r" \\", r"\midrule"]
    from . import analyses
    for d in datasets:
        cells = []
        for s in cfg["seeds"]:
            want = sum(len(analyses.plan(st, cfg["n_est"])) for st in cfg["setups"])
            have = sum(1 for (dd, ss, _, _) in index if dd == d and ss == s)
            cells.append(f"{have}/{want}")
        lines.append(f"{dlabel(d)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("status", wrap("\n".join(lines), "Analyses saved per dataset and seed, out of those planned.", "tab:status"))


def nan_free(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
