"""The adaptive attackers' setups and the fair domain test's placements, as the confirmation grid defines them."""
from rarag_eval import analyses, claims, setups
from rarag_eval.run import load_config

CFG = load_config("configs/confirm.yaml")


def setup(id_):
    return next(s for s in CFG["setups"] if s["id"] == id_)


def test_thin_attackers_attack_a_share_of_the_test():
    honest, advs = setups.specs(setup("adaptive_r10"), CFG)
    assert len(honest) == 6 and len(advs) == 3 and all("behaviour" not in a for a in advs)   # betray, as attackers_k3
    assert setups.target_fraction(setup("adaptive_r10"), CFG) == 0.10
    assert setups.target_fraction(setup("attackers_k3"), CFG) == CFG["target_fraction"] == 0.5


def test_burst_attackers_attack_in_windows_from_the_start_of_the_test():
    honest, advs = setups.specs(setup("burst_r20"), CFG)
    assert all(a["behaviour"] == "onoff" and a["switch_at"] == CFG["n_est"] and a["periodic"] == [5, 20] for a in advs)
    assert setups.target_fraction(setup("burst_r20"), CFG) == 1.0   # every question may be attacked; windows decide


def test_adaptive_setups_get_the_attackers_analyses():
    names = [a[0] for a in analyses.plan(setup("adaptive_r20"), CFG["n_est"])]
    assert names == analyses.PLAN["attackers"]


def test_the_fair_placements_read_only_the_second_labellers_labels():
    plan = {a[0]: a[2] for a in analyses.plan(setup("experts"), CFG["n_est"])}
    for p in ("domain_b", "centroid_b", "soft_b"):
        assert plan[f"dom_{p}_wfull"]["labels"] == "domain_b" and f"dom_{p}_w15_d2" in plan
    assert "labels" not in plan["dom_domain_wfull"]                  # A: the oracle, as before (names unchanged)


def test_every_claim_compares_analyses_the_grid_runs():
    ids = [c.id for c in claims.CLAIMS]
    assert len(ids) == len(set(ids))
    planned = {s["id"]: {a[0] for a in analyses.plan(s, CFG["n_est"])} for s in CFG["setups"]}
    for c in claims.CLAIMS:
        for s in c.setups:
            assert s in planned, (c.id, s)
            for analysis, *_ in (c.baseline, c.method):
                if analysis == "*":
                    continue
                if c.draws:   # several warm-up draws of one measure: a prefix
                    assert any(a.startswith(analysis) for a in planned[s]), (c.id, analysis)
                else:
                    assert analysis in planned[s], (c.id, analysis)
