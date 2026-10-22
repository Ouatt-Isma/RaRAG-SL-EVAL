"""The evaluation setups: which sources answer, and how bad ones behave. Each setup becomes one lab
scenario (one set of sources, one round of LLM answers) per dataset and seed. The specs repeat the
thesis runs exactly (the robustness sweep's 9 sources, the defense runs' 5-source mix, the Authority
experts), so a dataset and seed already run there reuses its cached answers."""
import collections

N_SOURCES = 9
# the defense runs' five honest sources (FINDINGS F-46: reliabilities drawn once from Beta(3, 2))
MIX5 = [("Source A", 0.75), ("Source B", 0.91), ("Source C", 0.526), ("Source D", 0.692), ("Source E", 0.618)]
KINDS = ("unreliable", "attackers", "switch", "mix5", "experts")


def _src(kind, rel, i, coverage):
    return {"name": f"{kind} {i + 1}", "reliability": rel, "coverage": coverage}


def expert_domains(records, n_est, n_test, n=5):
    """The n most frequent domains (the LLM's labels) among the test questions, "other" left out."""
    counts = collections.Counter(r.get("domain") for r in records[n_est:n_est + n_test])
    counts.pop(None, None)
    counts.pop("other", None)
    return [d for d, _ in counts.most_common(n)]


def specs(setup, cfg, records=None):
    """(honest, adversaries) for a setup: {"kind", ...} plus the config's coverage, n_est and n_test."""
    kind, cov = setup["kind"], cfg["coverage"]
    if kind == "unreliable":       # k sources right a tenth of the time, the rest 0.9 (the paper's adversary-hammer)
        k = int(setup["k"])
        return ([_src("Unreliable", 0.1, i, cov) for i in range(k)]
                + [_src("Reliable", 0.9, i, cov) for i in range(N_SOURCES - k)], [])
    if kind == "attackers":        # k coordinated build-then-betray attackers, honest through the warm-up
        k = int(setup["k"])
        return ([_src("Reliable", 0.9, i, cov) for i in range(N_SOURCES - k)],
                [{"name": f"Attacker {i + 1}", "est_coverage": cov, "test_coverage": cov} for i in range(k)])
    if kind == "switch":           # 3 attackers honest until halfway through the test, then attacking
        k = int(setup.get("k", 3))
        at = cfg["n_est"] + cfg["n_test"] // 2
        return ([_src("Reliable", 0.9, i, cov) for i in range(N_SOURCES - k)],
                [{"name": f"Attacker {i + 1}", "behaviour": "onoff", "switch_at": at, "est_coverage": cov,
                  "test_coverage": cov} for i in range(k)])
    if kind == "mix5":             # five honest sources between 0.53 and 0.91
        return [{"name": n, "reliability": r, "coverage": cov} for n, r in MIX5], []
    if kind == "experts":          # one expert per frequent domain, plus generalists (F-53)
        if records is None or not any("domain" in r for r in records):
            raise ValueError("the experts setup needs the dataset's domain labels (thesis scripts/label_domains.py)")
        domains = expert_domains(records, cfg["n_est"], cfg["n_test"], int(setup.get("n_experts", 5)))
        honest = [{"name": f"Expert: {d.replace('_', ' ')}", "reliability": float(setup.get("outside", 0.3)),
                   "coverage": cov, "expertise": {d: float(setup.get("expert", 0.9))}} for d in domains]
        honest += [{"name": f"Generalist {i + 1}", "reliability": float(setup.get("generalist", 0.6)), "coverage": cov}
                   for i in range(int(setup.get("generalists", 2)))]
        return honest, []
    raise ValueError(f"unknown setup kind {kind!r} (one of {', '.join(KINDS)})")


def scenario(th, dataset, seed, setup, cfg):
    """The lab scenario of a setup on a dataset with a seed (not built yet)."""
    records = th.lab.load_dataset(dataset)
    honest, advs = specs(setup, cfg, records)
    return th.lab.Scenario([{"file": dataset, "n_est": cfg["n_est"], "n_test": cfg["n_test"]}], honest, advs,
                           cfg["kappa"], top_k=cfg["top_k"], seed=int(seed), target_fraction=cfg["target_fraction"],
                           filtering_threshold=cfg["filtering_threshold"], model=th.model)
