"""Figures for the paper, as PDF in paper/figures/: vector, fonts embedded, sized for a two-column page."""
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .tables import DATASET_LABEL, PLACEMENT_LABEL, ROW_LABEL, values  # noqa: E402
from .thesis import ROOT  # noqa: E402

FIGURES = os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
# RA-RAG rows orange, SL rows blue (told apart by lightness and line style too), references grey
STYLE = {("plain", "rarag"): ("#c2410c", "--", "o"), ("plain", "rarag_online"): ("#c2410c", ":", "o"),
         ("aged", "rarag_window"): ("#c2410c", "-", "o"), ("plain", "sl_offline"): ("#1d4ed8", "--", "s"),
         ("plain", "sl_online"): ("#1d4ed8", ":", "s"), ("aged", "sl_online"): ("#1d4ed8", "-", "s"),
         ("window", "sl_online"): ("#60a5fa", "-", "s"), ("plain", "equal"): ("#6b7280", "-.", "^")}


def plain(label):
    """A table label (LaTeX) as plot text."""
    return label.replace(r"$\lambda=0.97$", "λ 0.97").replace("$", "")


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def empty(name, text="No results yet"):
    fig, ax = plt.subplots(figsize=(3.3, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#6b7280")
    save(fig, name)


def mean_spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    return statistics.mean(xs), min(xs), max(xs)


def by_k(rows, datasets, setups, ks, metric, name, ylabel):
    """Accuracy (or attacks that succeed) against the number of bad sources, one panel per dataset; the line
    is the mean over seeds, the band the range."""
    present = [d for d in datasets if any(values(rows, dataset=d, setup=s) for s in setups)]
    if not present:
        return empty(name)
    fig, axes = plt.subplots(1, len(present), figsize=(2.4 * len(present) + 0.8, 2.2), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], present):
        for key, (color, ls, marker) in STYLE.items():
            xs, ms, los, his = [], [], [], []
            for s, k in zip(setups, ks):
                m, lo, hi = mean_spread([r[metric] for r in values(rows, dataset=d, setup=s, analysis=key[0], row=key[1])])
                if m is not None:
                    xs.append(k), ms.append(100 * m), los.append(100 * lo), his.append(100 * hi)
            if xs:
                ax.plot(xs, ms, color=color, ls=ls, marker=marker, ms=3, lw=1.3, label=plain(ROW_LABEL[key]))
                if any(lo != hi for lo, hi in zip(los, his)):
                    ax.fill_between(xs, los, his, color=color, alpha=0.12, lw=0)
        ax.set_title(DATASET_LABEL.get(d, d))
        ax.set_xlabel("bad sources among 9")
        ax.set_xticks(ks)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, name)


def claims_forest(summary, datasets):
    """Each result's difference (method minus baseline, points) per dataset and over everything, with its
    95% interval; the expected direction is marked by the side of zero the claim sits on."""
    items = [s for s in summary if s["all"]["n"]]
    if not items:
        return empty("claims_forest")
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(items) + 0.9))
    markers = ["o", "s", "^", "D", "v"]
    for i, s in enumerate(items):
        y = len(items) - 1 - i
        a = s["all"]
        if a["lo"] is not None:
            ax.plot([100 * a["lo"], 100 * a["hi"]], [y, y], color="#111827", lw=1.6)
        ax.plot([100 * a["mean"]], [y], "|", color="#111827", ms=10, mew=2)
        for j, d in enumerate(datasets):
            m = s["per_dataset"][d]["mean"]
            if m is not None:
                ax.plot([100 * m], [y + 0.18], markers[j % len(markers)], ms=3.5, color=["#1d4ed8", "#c2410c", "#047857", "#7c3aed", "#6b7280"][j % 5],
                        label=DATASET_LABEL.get(d, d) if i == 0 else None)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{s['claim'].id}  {s['claim'].title}" for s in reversed(items)])
    ax.set_xlabel("method minus baseline (percentage points; claim 2: estimate minus truth, ×100)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", color="#f3f4f6", lw=0.6)
    save(fig, "claims_forest")


def domains(rows, datasets):
    present = [d for d in datasets if values(rows, dataset=d, setup="experts")]
    if not present:
        return empty("domains")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    width = 0.8 / len(PLACEMENT_LABEL)
    colors = ["#9ca3af", "#1d4ed8", "#93c5fd", "#60a5fa"]
    for j, (p, label) in enumerate(PLACEMENT_LABEL.items()):
        ms, errs = [], []
        for d in present:
            m, lo, hi = mean_spread([r["accuracy"] for r in values(rows, dataset=d, setup="experts", analysis=f"dom_{p}_wfull", row="sl_online")])
            ms.append(100 * m if m is not None else 0)
            errs.append([100 * (m - lo) if m is not None else 0, 100 * (hi - m) if m is not None else 0])
        xs = [i + (j - (len(PLACEMENT_LABEL) - 1) / 2) * width for i in range(len(present))]
        ax.bar(xs, ms, width * 0.92, color=colors[j], label=label, yerr=list(zip(*errs)), capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in present])
    ax.set_ylabel("accuracy (%), SL learning online")
    lo = min(v for v in [ax.get_ylim()[0]] + [50])
    ax.set_ylim(lo, None)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, "domains")


def warmup(rows, datasets, sizes):
    present = [d for d in datasets if values(rows, dataset=d, setup="mix5")]
    if not present:
        return empty("warmup")
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for j, d in enumerate(present):
        ys = []
        for size in sizes:
            sl = [r["accuracy"] for r in values(rows, dataset=d, setup="mix5", row="sl_offline") if r["analysis"].startswith(f"warm{size}_d")]
            ra = [r["accuracy"] for r in values(rows, dataset=d, setup="mix5", row="rarag") if r["analysis"].startswith(f"warm{size}_d")]
            ys.append(100 * (statistics.mean(sl) - statistics.mean(ra)) if sl and ra else None)
        pts = [(s, y) for s, y in zip(sizes, ys) if y is not None]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3, lw=1.3, label=DATASET_LABEL.get(d, d))
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.set_xlabel("warm-up questions")
    ax.set_ylabel("SL minus RA-RAG, frozen (points)")
    ax.legend(frameon=False)
    save(fig, "warmup")
