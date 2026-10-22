"""Saved analyses -> one flat table of row metrics (results/rows.csv), the input of the tables and plots.

For a swap row (RA-RAG's answering): accuracy and attack success. For a compare row (SL's or RA-RAG's
answering in the SL vs RA-RAG analysis): the three outcomes, accuracy on the answered questions, attack
success, conflicts showing the right answer, and RA-RAG's accuracy at the same coverage on the same trust
(the matching swap row), so that declining can be judged against RA-RAG keeping its most confident ones."""
import csv
import os

from . import store

# compare row -> the swap row of the same trust with RA-RAG's answering (for the same-coverage comparison)
SAME_TRUST = {"sl_online": "sl_online", "sl_offline": "sl_offline", "sl_rarag": "rarag", "sl_rarag_online": "rarag_online",
              "sl_rarag_window": "rarag_window"}
FIELDS = ["dataset", "seed", "setup", "analysis", "kind", "row", "n", "n_target", "accuracy", "attack", "coverage",
          "unknown", "conflict", "answered_accuracy", "conflict_gold", "unknown_would_be_wrong", "rarag_same_coverage",
          "label_agreement"]


def share(codes, symbols):
    return sum(c in symbols for c in codes) / len(codes) if codes else None


def same_coverage_accuracy(codes, conf, coverage):
    """Accuracy of a row answering only its most confident questions until it covers `coverage` of them
    (questions with equal confidence count by their mean)."""
    n = len(codes)
    scored = sorted(((c, k == "c") for k, c in zip(codes, conf) if c is not None and k != "x"), key=lambda x: -x[0])
    groups = []
    for c, ok in scored:
        if groups and groups[-1][0] == c:
            groups[-1][1] += 1
            groups[-1][2] += ok
        else:
            groups.append([c, 1, int(ok)])
    want, took, right = coverage * n, 0.0, 0.0
    for _, cnt, ok in groups:
        take = min(cnt, want - took)
        if take <= 0:
            break
        took += take
        right += ok * take / cnt
    return right / took if took > 0 and took >= want - 1e-9 else None


def rows_of(dataset, seed, setup, analysis, rec, index):
    meta = rec["meta"]
    targets = [q["target"] for q in rec["questions"]]
    n_target = sum(targets)
    out = []
    for name, r in rec["rows"].items():
        base = {"dataset": dataset, "seed": seed, "setup": setup, "analysis": analysis, "kind": meta["kind"], "row": name,
                "n": len(targets), "n_target": n_target, "label_agreement": rec.get("extra", {}).get("label_agreement")}
        if meta["kind"] == "swap":
            codes = r["fair"]
            base.update(accuracy=share(codes, "c"), coverage=1 - share(codes, "x"),
                        attack=share([c for c, t in zip(codes, targets) if t], "a") if n_target else None)
        else:
            dec, top, pair = r["decision"], r["top"], r["pair"]
            answered = [c for c in dec if c in "cwa"]
            unknown_tops = [t for c, t in zip(dec, top) if c == "u" and t != "x"]
            conflicts = [p for p in pair if p != "."]
            cov = len(answered) / len(dec)
            base.update(accuracy=share(dec, "c"), coverage=cov, unknown=share(dec, "u"), conflict=share(dec, "k"),
                        attack=share([c for c, t in zip(dec, targets) if t], "a") if n_target else None,
                        answered_accuracy=share(answered, "c") if answered else None,
                        conflict_gold=share(conflicts, "1") if conflicts else None,
                        unknown_would_be_wrong=share(unknown_tops, "wa") if unknown_tops else None)
            twin = SAME_TRUST.get(name)
            swap_rec = index.get((dataset, seed, setup, analysis.replace("answer_", "")))
            if twin and swap_rec and twin in swap_rec["rows"] and cov > 0:
                s = swap_rec["rows"][twin]
                base["rarag_same_coverage"] = same_coverage_accuracy(s["fair"], s["conf"], cov)
        out.append(base)
    return out


def collect(root=store.RESULTS):
    results = list(store.iter_results(root))
    index = {(d, s, setup, a): rec for d, s, setup, a, rec in results}
    rows = []
    for d, s, setup, a, rec in results:
        rows.extend(rows_of(d, s, setup, a, rec, index))
    return results, index, rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
