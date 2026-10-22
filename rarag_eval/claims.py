"""The first evaluation series' results, each as a comparison that every dataset and seed can confirm or
not. Numbering follows the summary table (First Evaluation Series); a/b are two measures of one result.
A comparison is a baseline row against a method row of the saved analyses, on the same test questions.

what = "acc": share of test questions answered correctly (fair match)
       "attack": share of attacked questions answered with the attackers' answer
direction = +1 the method is expected higher, -1 lower, 0 about equal (within a point)."""
import dataclasses

from . import stats

ATTACKERS = ["attackers_k1", "attackers_k2", "attackers_k3", "attackers_k4"]
LEARNED = [("plain", "rarag"), ("plain", "rarag_online"), ("plain", "sl_online"), ("plain", "sl_offline"),
           ("aged", "rarag_window"), ("aged", "sl_online"), ("window", "sl_online")]


@dataclasses.dataclass
class Claim:
    id: str
    title: str
    setups: list
    what: str
    baseline: tuple      # (analysis, row, label)
    method: tuple
    direction: int
    draws: str = ""      # analyses that are several warm-up draws of one measure: their common prefix
    special: str = ""


CLAIMS = [
    Claim("1", "RA-RAG beats equal weights (the paper's result)", ["unreliable_k3"], "acc",
          ("plain", "equal", "equal weights"), ("plain", "rarag", "RA-RAG (paper)"), +1),
    Claim("2", "RA-RAG's estimate overrates some sources", ["experts"], "overrate",
          ("plain", None, "warm-up truth"), ("plain", "rarag", "RA-RAG's estimate, largest excess"), +1, special="overrate"),
    Claim("3", "Frozen trust is easy to attack (2 attackers of 9)", ["attackers_k2"], "attack",
          ("plain", "rarag", "RA-RAG frozen (paper)"), ("aged", "sl_online", "SL with aging"), -1),
    Claim("4", "Staying online recovers most of it (3 attackers)", ["attackers_k3"], "acc",
          ("plain", "rarag", "RA-RAG frozen"), ("plain", "rarag_online", "RA-RAG online"), +1),
    Claim("5a", "Forgetting is the defense: SL (late switch)", ["switch"], "acc",
          ("plain", "sl_online", "SL online"), ("aged", "sl_online", "SL with aging"), +1),
    Claim("5b", "Forgetting is the defense: RA-RAG (late switch)", ["switch"], "acc",
          ("plain", "rarag_online", "RA-RAG online"), ("aged", "rarag_window", "RA-RAG with a window"), +1),
    Claim("6", "Same memory: SL and RA-RAG tie (1-3 attackers)", ATTACKERS[:3], "acc",
          ("aged", "rarag_window", "RA-RAG with a window"), ("window", "sl_online", "SL with a window"), 0),
    Claim("7", "The switch detector adds little", ["attackers_k3", "switch"], "acc",
          ("aged", "sl_online", "SL with aging"), ("detector", "sl_online", "SL with aging + the detector"), 0),
    Claim("8", "SL is better with few warm-up questions (5 sources, 5 questions)", ["mix5"], "acc",
          ("warm5", "rarag", "RA-RAG frozen"), ("warm5", "sl_offline", "SL frozen"), +1, draws="warm5_d"),
    Claim("9a", "SL holds better near a majority of attackers (4 of 9)", ["attackers_k4"], "acc",
          ("aged", "rarag_window", "RA-RAG with a window"), ("window", "sl_online", "SL with a window"), +1),
    Claim("9b", "... and fewer attacks succeed", ["attackers_k4"], "attack",
          ("aged", "rarag_window", "RA-RAG with a window"), ("window", "sl_online", "SL with a window"), -1),
    Claim("10", "Learning a trust is essential (7 unreliable of 9)", ["unreliable_k7"], "acc",
          ("plain", "equal", "equal weights"), ("*", "best", "best learned trust"), +1, special="best"),
    Claim("11", "One score per source rarely asks the experts", ["experts"], "acc",
          ("plain", "equal", "equal weights"), ("plain", "rarag", "RA-RAG (paper)"), -1),
    Claim("12a", "Trust per domain, online (SL, 15 warm-up questions)", ["experts"], "acc",
          ("dom_global_w15", "sl_online", "SL, one score"), ("dom_domain_w15", "sl_online", "SL per domain"), +1, draws="_d"),
    Claim("12b", "Trust per domain, frozen (SL, 15 warm-up questions)", ["experts"], "acc",
          ("dom_global_w15", "sl_offline", "SL frozen, one score"), ("dom_domain_w15", "sl_offline", "SL frozen per domain"),
          0, draws="_d"),
    Claim("13a", "Without labels, soft weights recover part of the gain", ["experts"], "acc",
          ("dom_global_wfull", "sl_online", "SL, no domains"), ("dom_soft_wfull", "sl_online", "SL, soft weights"), +1),
    Claim("13b", "Soft weights beat the nearest centroid", ["experts"], "acc",
          ("dom_centroid_wfull", "sl_online", "SL, nearest centroid"), ("dom_soft_wfull", "sl_online", "SL, soft weights"), +1),
    Claim("14a", "SL beats RA-RAG per domain (LLM labels)", ["experts"], "acc",
          ("dom_domain_wfull", "rarag_topic_online", "RA-RAG per domain, online"), ("dom_domain_wfull", "sl_online", "SL per domain"), +1),
    Claim("14b", "SL beats RA-RAG per domain (nearest centroid)", ["experts"], "acc",
          ("dom_centroid_wfull", "rarag_topic_online", "RA-RAG per domain, online"), ("dom_centroid_wfull", "sl_online", "SL per domain"), +1),
    Claim("15", "SL's own answering protects out-of-date trust (frozen, 3 attackers)", ["attackers_k3"], "attack",
          ("aged", "sl_offline", "RA-RAG's answering"), ("answer_aged", "sl_offline", "SL's own answering"), -1),
]


def index(results):
    """{(dataset, seed, setup, analysis): record}"""
    return {(d, s, setup, a): rec for d, s, setup, a, rec in results}


def _codes(rec, row):
    r = (rec or {}).get("rows", {}).get(row)
    if r is None:
        return None
    return r.get("fair") or r.get("decision")


def _measure(rec, codes, what):
    if codes is None:
        return None, None
    if what == "acc":
        return codes, "c"
    mask = [q["target"] for q in rec["questions"]]
    return "".join(c for c, m in zip(codes, mask) if m), "a"


def _value(codes, symbol):
    return sum(c == symbol for c in codes) / len(codes) if codes else None


def _analyses(ix, unit, setup, analysis, draws):
    d, s = unit
    if not draws:
        rec = ix.get((d, s, setup, analysis))
        return [rec] if rec else []
    prefix = analysis + draws if draws.startswith("_") else draws
    return [rec for (dd, ss, st, a), rec in sorted(ix.items(), key=lambda kv: kv[0][3])
            if (dd, ss, st) == (d, s, setup) and a.startswith(prefix)]


def _best(ix, unit, setup):
    """The learned row with the highest accuracy on this unit."""
    d, s = unit
    best = None
    for analysis, row in LEARNED:
        rec = ix.get((d, s, setup, analysis))
        codes = _codes(rec, row)
        if codes:
            acc = _value(codes, "c")
            if best is None or acc > best[0]:
                best = (acc, rec, codes, f"{row} ({analysis})")
    return best


def evaluate(claim, ix, unit, setup):
    """The claim's comparison on one dataset, seed and setup, or None when its analyses are missing."""
    if claim.special == "overrate":
        rec = ix.get((*unit, setup, claim.baseline[0]))
        if not rec:
            return None
        gaps = [s["rarag_estimate"] - s["oracle_warm"] for s in rec["sources"]
                if s.get("rarag_estimate") is not None and s.get("oracle_warm") is not None]
        if not gaps:
            return None
        return {"base": 0.0, "method": max(gaps), "diff": max(gaps), "p": None, "lo": None, "hi": None}
    if claim.special == "best":
        rec_b = ix.get((*unit, setup, claim.baseline[0]))
        best = _best(ix, unit, setup)
        if not rec_b or not best:
            return None
        a, sym = _measure(rec_b, _codes(rec_b, claim.baseline[1]), claim.what)
        b, _ = _measure(best[1], best[2], claim.what)
        lo, hi = stats.paired_bootstrap(a, b, sym)
        return {"base": _value(a, sym), "method": _value(b, sym), "diff": _value(b, sym) - _value(a, sym),
                "p": stats.mcnemar(a, b, sym), "lo": lo, "hi": hi, "note": best[3]}
    recs_a = _analyses(ix, unit, setup, claim.baseline[0], claim.draws)
    recs_b = _analyses(ix, unit, setup, claim.method[0], claim.draws)
    if not recs_a or not recs_b or len(recs_a) != len(recs_b):
        return None
    pairs = []
    for ra, rb in zip(recs_a, recs_b):
        a, sym = _measure(ra, _codes(ra, claim.baseline[1]), claim.what)
        b, _ = _measure(rb, _codes(rb, claim.method[1]), claim.what)
        if a is None or b is None or len(a) != len(b) or not a:
            return None
        pairs.append((a, b, sym))
    base = sum(_value(a, s) for a, _, s in pairs) / len(pairs)
    meth = sum(_value(b, s) for _, b, s in pairs) / len(pairs)
    out = {"base": base, "method": meth, "diff": meth - base, "p": None, "lo": None, "hi": None}
    if len(pairs) == 1:
        a, b, sym = pairs[0]
        out["p"] = stats.mcnemar(a, b, sym)
        out["lo"], out["hi"] = stats.paired_bootstrap(a, b, sym)
    return out


def unit_results(ix):
    """Every claim on every dataset, seed and setup that has its analyses."""
    units = sorted({(d, s) for d, s, _, _ in ix})
    out = []
    for claim in CLAIMS:
        for unit in units:
            for setup in claim.setups:
                r = evaluate(claim, ix, unit, setup)
                if r:
                    out.append({"claim": claim.id, "dataset": unit[0], "seed": unit[1], "setup": setup, **r})
    return out


def summary(rows, datasets):
    """Per claim: the aggregate over each dataset's seeds and over everything."""
    out = []
    for claim in CLAIMS:
        mine = [r for r in rows if r["claim"] == claim.id]
        per = {d: stats.aggregate([r["diff"] for r in mine if r["dataset"] == d], claim.direction) for d in datasets}
        out.append({"claim": claim, "per_dataset": per, "all": stats.aggregate([r["diff"] for r in mine], claim.direction),
                    "significant": sum(1 for r in mine if r.get("p") is not None and r["p"] < 0.05)})
    return out
