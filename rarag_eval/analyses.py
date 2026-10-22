"""What is computed on each setup's answers, and the compact record kept of it.

Two kinds of analysis, both the thesis lab's (webapp/backend/sl_minimal.py), on the same cached LLM
answers:
  swap     RA-RAG's own answering (kappa sources, weighted vote), with each reliability in turn: RA-RAG's
           estimate (frozen, online, windowed, per domain), SL's trust (frozen, online), the oracle,
           equal weights. Only the reliability differs between rows.
  compare  SL's own answering (every source, fused evidence; answer, "unknown" or "conflict") on SL's
           trust and on RA-RAG's estimates, next to RA-RAG's answering.
Every row keeps one character per test question (its outcome), so that any two rows can be compared
question by question (paired tests), plus its confidence where a curve needs it."""

# SL's settings in every analysis (the slides' W, base rate and decision thresholds; the agreement rule)
BASE = {"W": 2, "a": 0.5, "L": 10, "u_max": 0.3, "margin": 0.1, "fusion": "cumulative", "feedback": "agreement"}
AGED = {"lam_pos": 0.97, "lam_neg": 0.97, "rarag_window": 100}     # the recommended SL setting + RA-RAG's window
FIXED = {   # analysis id -> (kind, settings on top of BASE)
    "plain": ("swap", {}),
    "aged": ("swap", AGED),
    "window": ("swap", {"sl_window": 100}),
    "detector": ("swap", {**AGED, "dc": True, "dc_response": "reset", "theta": 0.25}),   # aging + the switch detector
    "answer_aged": ("compare", AGED),
}
PLAN = {    # setup kind -> the fixed analyses it gets
    "unreliable": ["plain", "aged", "window", "answer_aged"],
    "attackers": ["plain", "aged", "window", "detector", "answer_aged"],
    "switch": ["plain", "aged", "window", "detector", "answer_aged"],
    "mix5": ["plain", "aged", "window", "answer_aged"],
    "experts": ["plain", "answer_aged"],
}
PLACEMENTS = {  # how a question is placed among domains (F-53, F-56)
    "global": {"scope": False},
    "domain": {"scope": True, "scope_by": "domain"},
    "centroid": {"scope": True, "scope_by": "centroid"},
    "soft": {"scope": True, "scope_by": "soft", "soft_temp": 0.02},
}
CODE = {"correct": "c", "wrong": "w", "attacked": "a", "abstained": "x"}


def plan(setup, n_est):
    """[(analysis id, kind, settings)] for a setup: its fixed analyses, the warm-up study where asked
    (random warm-up subsets, frozen rows), and the domain placements for the experts."""
    out = [(a, *FIXED[a]) for a in PLAN[setup["kind"]]]
    warm = setup.get("warmup")
    if warm:
        for size in warm["sizes"]:
            for draw in range(int(warm["draws"])):
                out.append((f"warm{size}_d{draw}", "swap", {"warmup_size": int(size), "warmup_draw": draw}))
    dom = setup.get("domains")
    if dom:
        sizes = [int(s) for s in dom.get("sizes", [])] + [n_est]
        for placement in dom.get("placements", list(PLACEMENTS)):
            for size in sizes:
                for draw in range(1 if size >= n_est else int(dom.get("draws", 3))):
                    extra = {} if size >= n_est else {"warmup_size": size, "warmup_draw": draw}
                    name = f"dom_{placement}_w{'full' if size >= n_est else size}" + ("" if size >= n_est else f"_d{draw}")
                    out.append((name, "swap", {**PLACEMENTS[placement], **extra}))
    return out


def _code(outcome):
    return CODE.get(outcome, "x")


def _questions(sc, idxs):
    return [{"idx": i, "target": i in sc.targets, "domain": sc.data[i].get("domain")} for i in idxs]


def _sources(sc, res):
    """Each source's spec, RA-RAG's estimate of it, SL's trust after the warm-up and its real share of
    correct warm-up answers (the oracle)."""
    v = res["variants"]
    est = (v.get("rarag") or {}).get("reliability") or [None] * sc.num_src
    sl = (v.get("sl_offline") or {}).get("reliability") or [None] * sc.num_src
    rho = [o.get("rho") for o in (res.get("oracle") or {}).get("warmup") or [{}] * sc.num_src]
    out = []
    for i, s in enumerate(sc.sources):
        out.append({"name": s["name"], "kind": s.get("kind"), "reliability": s.get("reliability"),
                    "expertise": s.get("expertise"), "rarag_estimate": est[i], "sl_warm": sl[i], "oracle_warm": rho[i]})
    return out


def swap(th, sc, settings, kappa):
    """RA-RAG's answering with each reliability: per row, the outcome of every test question (fair and strict
    match) and the winner's vote share."""
    run = th.sl_minimal.MinimalRun(sc, {**BASE, **settings})
    res = run.swap()
    rows, idxs = {}, None
    for name, spec in res["variants"].items():
        rs = spec["by_kappa"][kappa - 1]["rows"]
        idxs = idxs or [r["idx"] for r in rs]
        rows[name] = {"fair": "".join(_code(r["outcome_fair"]) for r in rs),
                      "strict": "".join(_code(r["outcome"]) for r in rs),
                      "conf": [None if r.get("c") is None else round(float(r["c"]), 4) for r in rs]}
    extra = {}
    if settings.get("scope"):   # how often the question's top domain is the LLM's label
        keys = [th.sl_minimal.topic_key(run._scope_topic(i)) for i in idxs]
        extra["label_agreement"] = sum(k == sc.data[i].get("domain") for k, i in zip(keys, idxs)) / len(idxs)
    if run.subset:
        extra["warmup_idx"] = run.subset["idx"]
    return {"questions": _questions(sc, idxs), "sources": _sources(sc, res), "rows": rows, "extra": extra}


def compare(th, sc, settings):
    """SL's own answering next to RA-RAG's: per row, the decision on every test question (c/w/a answered right,
    wrong or with the attackers' answer; u unknown; k conflict; x nobody answered), the top candidate's outcome,
    whether a conflict showed the right answer, and the confidence."""
    run = th.sl_minimal.MinimalRun(sc, {**BASE, **settings})
    res = run.compare()
    rows, idxs = {}, None
    for name, spec in res["variants"].items():
        rs = spec["rows"]
        idxs = idxs or [r["idx"] for r in rs]
        dec, top, pair = [], [], []
        for r in rs:
            kind = r.get("kind")
            dec.append("u" if kind == "unknown" else "k" if kind == "conflict" else _code(r["outcome_fair"]))
            top.append(_code(r["top_outcome_fair"]) if r.get("top") is not None else "x")
            if kind == "conflict":
                gold = sc.data[r["idx"]]["answers"]
                ok = r["top_outcome_fair"] == "correct" or bool(r.get("runner_up") and th.fair_match(r["runner_up"], gold))
                pair.append("1" if ok else "0")
            else:
                pair.append(".")
        rows[name] = {"decision": "".join(dec), "top": "".join(top), "pair": "".join(pair),
                      "conf": [None if r.get("c") is None else round(float(r["c"]), 4) for r in rs]}
    return {"questions": _questions(sc, idxs), "rows": rows, "extra": {}}
