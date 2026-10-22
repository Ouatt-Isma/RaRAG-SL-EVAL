"""Rebuild everything the paper uses from the saved results: results/rows.csv and results/claims.csv,
the LaTeX tables (paper/tables), the figures (paper/figures) and the results document (report/results.pdf).

  python -m rarag_eval.report configs/confirm.yaml          # tables, figures and the PDF
  python -m rarag_eval.report configs/confirm.yaml --no-pdf # without LaTeX
"""
import argparse
import csv
import os
import shutil
import subprocess
import time

from . import claims, collect, plots, tables
from .run import load_config
from .thesis import ROOT

REPORT = os.path.join(ROOT, "report")


def write_generated(cfg, index, rows, summary):
    """report/generated/*.tex: the facts the document states in its text (date, how much has run)."""
    out = os.path.join(REPORT, "generated")
    os.makedirs(out, exist_ok=True)
    units = sorted({(d, s) for d, s, _, _ in index})
    commits = sorted({rec["meta"].get("thesis_commit", "?") for rec in index.values()})
    tags = sorted({rec["meta"].get("llm_tag") or "tag not recorded" for rec in index.values()})
    if len(tags) > 1:
        print(f"WARNING: these results come from different LLMs ({', '.join(tags)}): keep each in its own results folder")
    with open(os.path.join(out, "facts.tex"), "w") as f:
        f.write(f"\\newcommand{{\\builtat}}{{{time.strftime('%Y-%m-%d %H:%M')}}}\n")
        f.write(f"\\newcommand{{\\nanalyses}}{{{len(index)}}}\n")
        f.write(f"\\newcommand{{\\nunits}}{{{len(units)}}}\n")
        f.write(f"\\newcommand{{\\datasets}}{{{', '.join(tables.dlabel(d) for d in sorted({u[0] for u in units})) or 'none yet'}}}\n")
        f.write(f"\\newcommand{{\\seeds}}{{{', '.join(str(s) for s in sorted({u[1] for u in units})) or 'none yet'}}}\n")
        f.write(f"\\newcommand{{\\thesiscommit}}{{{tables.esc(', '.join(commits)) or '--'}}}\n")
        f.write(f"\\newcommand{{\\model}}{{{tables.esc(cfg['model'] + ' (' + ', '.join(tags) + ')' if tags else cfg['model'])}}}\n")


def build(cfg, pdf=True):
    results, index, rows = collect.collect()
    collect.write_csv(rows, os.path.join(ROOT, "results", "rows.csv"))
    datasets = [os.path.splitext(os.path.basename(d))[0] for d in cfg["datasets"]]
    unit_rows = claims.unit_results(index)
    with open(os.path.join(ROOT, "results", "claims.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["claim", "dataset", "seed", "setup", "base", "method", "diff", "p", "lo", "hi", "note"])
        w.writeheader()
        for r in unit_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    summary = claims.summary(unit_rows, datasets)
    attackers = ["attackers_k1", "attackers_k2", "attackers_k3", "attackers_k4"]
    mix5 = next((s for s in cfg["setups"] if s["kind"] == "mix5"), {})
    sizes = (mix5.get("warmup") or {}).get("sizes", [])

    tables.claims_table(summary, datasets)
    tables.robustness_tables(rows, datasets, attackers + ["switch"], "accuracy", "attackers_accuracy",
                             r"Coordinated build-then-betray attackers among 9 sources (and 3 attackers switching halfway "
                             r"through the test): accuracy in percent, mean over seeds with the spread as a subscript.")
    tables.robustness_tables(rows, datasets, attackers + ["switch"], "attack", "attackers_attack",
                             r"The same setups: share of attacked questions answered with the attackers' answer, in percent.")
    tables.robustness_tables(rows, datasets, ["unreliable_k3", "unreliable_k7"], "accuracy", "unreliable",
                             r"Static unreliable sources (right 10\% of the time) among 9: accuracy in percent.")
    tables.domains_table(rows, datasets)
    tables.answering_table(rows, [s["id"] for s in cfg["setups"]])
    tables.warmup_table(rows, datasets, sizes)
    tables.status_table(index, cfg)

    plots.claims_forest(summary, datasets)
    plots.by_k(rows, datasets, attackers, [1, 2, 3, 4], "accuracy", "attackers_accuracy", "accuracy (%)")
    plots.by_k(rows, datasets, attackers, [1, 2, 3, 4], "attack", "attackers_attack", "attacks that succeed (%)")
    plots.domains(rows, datasets)
    plots.warmup(rows, datasets, sizes)
    write_generated(cfg, index, rows, summary)

    if pdf:
        if not shutil.which("latexmk"):
            print("latexmk not found: tables and figures written, no PDF")
            return
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", "results.tex"],
                           cwd=REPORT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("LaTeX failed: see report/build/results.log")
        shutil.copy(os.path.join(REPORT, "build", "results.pdf"), os.path.join(REPORT, "results.pdf"))
        print(f"report/results.pdf: {len(index)} analyses, {len({(d, s) for d, s, _, _ in index})} dataset-seed units")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    build(load_config(args.config), pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
