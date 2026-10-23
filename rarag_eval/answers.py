"""The LLM answers a run caches in its folder (the thesis lab's vector_cache/<run>/), checked for empty ones. The LLM
never answers with nothing: an empty answer is a failed request (on 2026-10-23, Ollama servers out of GPU memory
answered thousands of requests with no text). The lab reads it as "I don't know", so a setup with empty answers looks
like one where no source knew anything. The LLM's response cache never stores an empty response, so dropping the empty
answers and what was computed from them makes the next run ask only those questions again.

    estimation__<model>.json   the warm-up answers ("raw_answers": per source, per question)
    answers__<model>.json      the test answers ({key: {"raw", ...}})
    everything else but the sources (corpora.pkl, src_*.faiss, manifest.json) is computed from them
"""
import json
import os

KEEP = ("corpora.pkl", "manifest.json")


def _empty(text):
    return text is None or not str(text).strip()


def check(run_dir, model, examples=3):
    """{"warmup": (empty, total), "test": (empty, total), "examples": [...]} for one run's cached answers."""
    out = {"warmup": (0, 0), "test": (0, 0), "examples": []}
    est = os.path.join(run_dir, f"estimation__{model}.json")
    if os.path.exists(est):
        with open(est) as f:
            raw = json.load(f).get("raw_answers") or []
        flat = [(s, q, a) for s, answers in enumerate(raw) for q, a in enumerate(answers)]
        bad = [(s, q, a) for s, q, a in flat if _empty(a)]
        out["warmup"] = (len(bad), len(flat))
        out["examples"] += [f"warm-up, source {s + 1}, question {q}: {a!r}" for s, q, a in bad[:examples]]
    bank = os.path.join(run_dir, f"answers__{model}.json")
    if os.path.exists(bank):
        with open(bank) as f:
            entries = json.load(f)
        bad = [e for e in entries.values() if _empty(e.get("raw"))]
        out["test"] = (len(bad), len(entries))
        out["examples"] += [f"test, source {e.get('source', 0) + 1}: {str(e.get('question'))[:60]!r} -> {e.get('raw')!r}"
                            for e in bad[:examples]]
    return out


def has_empty(report):
    return report["warmup"][0] > 0 or report["test"][0] > 0


def clean(run_dir, model):
    """Drop the empty test answers, the warm-up answers if any is empty, and every file computed from answers; keep
    the sources and the good answers. Returns the names of the files changed or removed."""
    report = check(run_dir, model, examples=0)
    changed = []
    bank = os.path.join(run_dir, f"answers__{model}.json")
    if report["test"][0]:
        with open(bank) as f:
            entries = json.load(f)
        tmp = bank + ".tmp"
        with open(tmp, "w") as f:
            json.dump({k: e for k, e in entries.items() if not _empty(e.get("raw"))}, f)
        os.replace(tmp, bank)
        changed.append(os.path.basename(bank))
    est = f"estimation__{model}.json"
    for name in sorted(os.listdir(run_dir)):
        if name in KEEP or name.startswith("src_") or name == os.path.basename(bank):
            continue
        if name == est and not report["warmup"][0]:
            continue   # good warm-up answers stay
        os.remove(os.path.join(run_dir, name))
        changed.append(name)
    return changed
