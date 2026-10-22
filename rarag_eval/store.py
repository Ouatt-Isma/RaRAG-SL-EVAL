"""Where results live: results/<dataset>/seed<seed>/<setup>/<analysis>.json.gz, one file per analysis,
written whole (a crash never leaves half a file), plus results/runs.jsonl, a log line per file."""
import gzip
import json
import os
import time

from .thesis import ROOT

RESULTS = os.path.join(ROOT, "results")


def dataset_name(file):
    return os.path.splitext(os.path.basename(file))[0]


def path(dataset, seed, setup, analysis, root=RESULTS):
    return os.path.join(root, dataset_name(dataset), f"seed{seed}", setup, f"{analysis}.json.gz")


def exists(dataset, seed, setup, analysis, root=RESULTS):
    return os.path.exists(path(dataset, seed, setup, analysis, root))


def save(record, dataset, seed, setup, analysis, root=RESULTS):
    p = path(dataset, seed, setup, analysis, root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    os.replace(tmp, p)
    with open(os.path.join(root, "runs.jsonl"), "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": os.path.relpath(p, root),
                            **{k: record["meta"].get(k) for k in ("thesis_commit", "model", "run", "seconds")}}) + "\n")
    return p


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def iter_results(root=RESULTS):
    """Every saved analysis: (dataset, seed, setup, analysis, record)."""
    if not os.path.isdir(root):
        return
    for dataset in sorted(os.listdir(root)):
        dpath = os.path.join(root, dataset)
        if not os.path.isdir(dpath):
            continue
        for seed_dir in sorted(os.listdir(dpath)):
            if not seed_dir.startswith("seed"):
                continue
            for setup in sorted(os.listdir(os.path.join(dpath, seed_dir))):
                spath = os.path.join(dpath, seed_dir, setup)
                for f in sorted(os.listdir(spath)):
                    if f.endswith(".json.gz"):
                        yield dataset, int(seed_dir[4:]), setup, f[:-len(".json.gz")], load(os.path.join(spath, f))
