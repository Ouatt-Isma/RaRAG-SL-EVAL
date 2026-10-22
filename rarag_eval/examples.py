"""Example source documents as plain text, for reading and for a paper's appendix. For a few test questions
of each setup of a run, two files:

  examples/<dataset>/seed<N>/<setup>/question_<idx>.txt
      the question, its gold answer, whether it is attacked; then every source: what it holds for this
      question (documents supporting the gold answer, a wrong answer, the attackers' answer, or nothing), the
      passages it retrieved, the LLM's answer, AlignScore's support and whether the filter kept the answer
  examples/<dataset>/documents_<idx>.txt
      every document the dataset holds for that question: the paraphrased gold passages, each wrong answer's
      documents and the attackers' documents (generated when the dataset was built)

Cached answers only (no LLM call).

  python -m rarag_eval.examples configs/pilot.yaml --runs-dir ../Ra-Rag-Thesis-Adaptation/server_backup/vector_cache
"""
import argparse
import os
import textwrap

from . import setups, store
from .run import load_config, units
from .thesis import ROOT, Thesis

EXAMPLES = os.path.join(ROOT, "examples")
WIDTH = 100
KIND = {"gold": "documents supporting the gold answer", "counterfactual": "documents supporting a wrong answer",
        "adversarial": "the attackers' documents", "none": "no documents for this question"}


def wrap(text, indent="    "):
    return "\n".join(textwrap.fill(p, WIDTH, initial_indent=indent, subsequent_indent=indent)
                     for p in str(text).split("\n") if p.strip()) or indent + "(empty)"


def pick(sc, n):
    """n test questions: attacked ones first (where there are attackers), then ones where the sources hold
    different kinds of documents, then any; deterministic."""
    test = list(range(sc.n_est, sc.n_est + sc.n_test))
    attacked = [i for i in test if i in sc.targets]
    mixed = [i for i in test if i not in sc.targets and len({sc.holdings[s][i]["kind"] for s in range(sc.num_src)}) >= 3]
    out = []
    for group in (attacked[:max(1, n // 2)], mixed, test):
        for i in group:
            if i not in out and len(out) < n:
                out.append(i)
    return out


def documents_file(rec, idx):
    lines = [f"Question {idx}: {rec['question']}", f"Gold answer(s): {' | '.join(rec['answers'])}", "",
             "Documents supporting the gold answer (paraphrases of the gold passage):"]
    for k, t in enumerate(rec.get("paraphrase", []), 1):
        lines += [f"  [{k}]", wrap(t)]
    for j, cf in enumerate(rec.get("counterfactual", []), 1):
        lines += ["", f"Wrong answer {j}: {cf['answers'][0]}"]
        for k, t in enumerate(cf.get("contexts", []), 1):
            lines += [f"  [{k}]", wrap(t)]
    adv = rec.get("adversarial")
    if adv:
        lines += ["", f"Attackers' answer: {adv['answers'][0]}"]
        for k, t in enumerate(adv.get("contexts", []), 1):
            lines += [f"  [{k}]", wrap(t)]
    return "\n".join(lines) + "\n"


def question_file(th, sc, idx, entries, label):
    rec = sc.data[idx]
    attacked = idx in sc.targets
    payload = (rec.get("adversarial") or {}).get("answers", [None])[0]
    lines = [f"Setup: {label}  ({sc.run_name})", f"Question {idx}: {rec['question']}",
             f"Gold answer(s): {' | '.join(rec['answers'])}",
             f"Attacked question: {'yes, the attackers answer ' + repr(payload) if attacked else 'no'}", ""]
    for s, (src, e) in enumerate(zip(sc.sources, entries)):
        hold = sc.holdings[s][idx]
        role = ("attacker: honest through the warm-up, then the attackers' answer on the questions they target"
                if src.get("kind") == "adversary"
                else f"right {src.get('reliability', 0):.0%} of the time"
                + (f" ({', '.join(f'{d} {v:.0%}' for d, v in src['expertise'].items())})" if src.get("expertise") else ""))
        lines += ["=" * WIDTH, f"Source {s + 1}: {src['name']} - {role}",
                  f"Holds for this question: {KIND.get(hold['kind'], hold['kind'])}"
                  + (f" (answer: {hold['answer']})" if hold.get("answer") else "")]
        docs = sc.corpora[s]["q_info"][idx].get("ctxs") or []
        for k, t in enumerate(docs, 1):
            lines += [f"  document [{k}]", wrap(t)]
        lines += ["", f"Retrieved (top {sc.top_k} of everything this source holds):"]
        for k, h in enumerate(e.get("retrieved") or [], 1):
            meta = sc._passage_meta.get(h["text"], {})
            about = meta.get("variant", "unknown")
            other = "" if meta.get("question_idx") in (None, idx) else f", from question {meta['question_idx']}"
            lines += [f"  [{k}] score {h.get('score', 0):.3f}, {about}{' (answer: ' + meta['answer'] + ')' if meta.get('answer') else ''}{other}",
                      wrap(h["text"], "      ")]
        lines += ["", f"LLM answer: {e['raw']!r}", f"AlignScore support g = {e['alignscore']:.3f} -> "
                  + ("kept" if e["filtered"] == e["raw"] else "dropped (the retrieved passages do not support it)"),
                  f"Counts as: {verdict(th, e, rec)}", ""]
    return "\n".join(lines) + "\n"


def abstains(answer):
    return not answer or answer.strip().lower().rstrip(".") in ("i don't know", "unknown")


def verdict(th, e, rec):
    if abstains(e["filtered"]):
        return "no answer"
    if th.fair_match(e["filtered"], rec["answers"]):
        return "right"
    payload = (rec.get("adversarial") or {}).get("answers", [None])[0]
    return "the attackers' answer" if payload and th.fair_match(e["filtered"], [payload]) else "wrong"


def index_line(th, sc, idx, entries, path):
    rec = sc.data[idx]
    counts = {}
    for e in entries:
        v = verdict(th, e, rec)
        counts[v] = counts.get(v, 0) + 1
    tally = ", ".join(f"{n} {v}" for v, n in sorted(counts.items(), key=lambda x: -x[1]))
    return f"  {path}\n      {rec['question']}  (gold: {rec['answers'][0]}{'; attacked' if idx in sc.targets else ''})\n      sources: {tally}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*", default=[], help="filters: dataset=NQ_paper seed=0 setup=attackers_k2")
    ap.add_argument("--questions", type=int, default=3, help="test questions per setup")
    ap.add_argument("--runs-dir")
    ap.add_argument("--embeddings")
    args = ap.parse_args()
    cfg = load_config(args.config)
    only = dict(kv.split("=", 1) for kv in args.only)
    th = Thesis(cfg["model"], runs_dir=args.runs_dir, offline=True, embeddings=args.embeddings)
    written, index = 0, {}
    for dataset, seed, setup in units(cfg, only):
        sc = setups.scenario(th, dataset, seed, setup, cfg)
        if not os.path.exists(os.path.join(sc.run_dir, "corpora.pkl")):
            print(f"{store.dataset_name(dataset)} seed {seed} {setup['id']}: not built, skipped")
            continue
        sc.build()
        bank = th.sl_lab.AnswerBank(sc)
        out = os.path.join(EXAMPLES, store.dataset_name(dataset), f"seed{seed}", setup["id"])
        os.makedirs(out, exist_ok=True)
        for idx in pick(sc, args.questions):
            q = sc.data[idx]["question"]
            try:
                entries = bank.get([(s, q) for s in range(sc.num_src)], {q: th.lab.RES.emb().get([q])[0]})
            except RuntimeError as e:
                print(f"  question {idx}: {e}")
                continue
            with open(os.path.join(out, f"question_{idx}.txt"), "w") as f:
                f.write(question_file(th, sc, idx, entries, f"{store.dataset_name(dataset)}, seed {seed}, {setup['id']}"))
            with open(os.path.join(EXAMPLES, store.dataset_name(dataset), f"documents_{idx}.txt"), "w") as f:
                f.write(documents_file(sc.data[idx], idx))
            index.setdefault(store.dataset_name(dataset), []).append(
                index_line(th, sc, idx, entries, os.path.relpath(os.path.join(out, f"question_{idx}.txt"), EXAMPLES)))
            written += 1
        print(f"{store.dataset_name(dataset)} seed {seed} {setup['id']}: examples in {os.path.relpath(out, ROOT)}")
    with open(os.path.join(EXAMPLES, "INDEX.txt"), "w") as f:
        f.write(INDEX_HEAD)
        for name, lines in index.items():
            f.write(f"\n{name}\n" + "\n".join(lines) + "\n")
    print(f"{written} question files, index in examples/INDEX.txt")


INDEX_HEAD = """Example source documents and answers (generated by `make examples` from the cached LLM answers).

question_<idx>.txt   one test question in one setup: what each source holds for it, the 3 passages it
                     retrieved, the LLM's answer and whether AlignScore kept it
documents_<idx>.txt  every document the dataset holds for that question: the gold passage's paraphrases,
                     the documents for each wrong answer, and the attackers' documents

A source holds documents for only some questions (how often it holds the gold ones sets its reliability);
for the others its retriever brings back passages about other questions and the LLM usually says it does
not know. "Counts as" uses the fair match against the gold answer.
"""


if __name__ == "__main__":
    main()
