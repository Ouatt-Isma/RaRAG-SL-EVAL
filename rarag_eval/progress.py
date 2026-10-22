"""How far a run is and how long it has left. Work is counted in setups, each weighed by what it costs: a setup not
started yet by its number of sources (its LLM answers take nearly all the time), one with some analyses saved by 1
(its answers are cached). Until a setup finishes, the time left is a first guess (about 3 GPU hours per dataset and
seed on an RTX 4090); after, it is this run's own pace. The launcher keeps the state in logs/progress.json, so
`make status` can tell from another terminal."""
import json
import os
import time

from . import analyses, store
from .run import load_config, units
from .thesis import ROOT

STATE = os.path.join(ROOT, "logs", "progress.json")
# first guess: a dataset and seed (9 setups, 75 sources in all) takes about 3 hours on one 24 GB GPU
SECONDS_PER_SOURCE = 3 * 3600 / 75


def sources(setup):
    return {"mix5": 5, "experts": 7}.get(setup["kind"], 9)


def remaining(cfg, only):
    """{setup key: weight} of the setups with analyses left."""
    left = {}
    for dataset, seed, setup in units(cfg, only):
        todo = analyses.plan(setup, cfg["n_est"])
        saved = sum(store.exists(dataset, seed, setup["id"], a[0]) for a in todo)
        if saved < len(todo):
            left[f"{store.dataset_name(dataset)} seed {seed} {setup['id']}"] = 1 if saved else sources(setup)
    return left


def only_filters(args):
    """The --only key=value filters among rarag_eval.run's options."""
    out, on = {}, False
    for a in args:
        if a.startswith("--"):
            on = a == "--only"
        elif on and "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


def duration(seconds):
    m = int(seconds // 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


class Progress:
    def __init__(self, config, only, workers):
        self.config, self.cfg, self.only, self.workers = config, load_config(config), only, workers
        self.t0, self.pid = time.time(), os.getpid()
        self.start = remaining(self.cfg, only)

    def save(self):
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        with open(STATE, "w") as f:
            json.dump({"config": os.path.abspath(self.config), "t0": self.t0, "pid": self.pid, "workers": self.workers,
                       "only": self.only, "start": self.start}, f)

    @classmethod
    def load(cls):
        """The run the launcher last started (its own config, whatever CONFIG is now)."""
        with open(STATE) as f:
            s = json.load(f)
        p = cls.__new__(cls)
        p.config, p.only, p.workers, p.t0, p.pid, p.start = s["config"], s["only"], s["workers"], s["t0"], s["pid"], s["start"]
        p.cfg = load_config(p.config)
        return p

    def done(self):
        left = remaining(self.cfg, self.only)
        return [k for k in self.start if k not in left], left

    def line(self):
        done, left = self.done()
        elapsed = time.time() - self.t0
        done_w = sum(self.start[k] for k in done)
        left_w = sum(left.values())
        if not left:
            return f"progress: all {len(self.start)} setups done in {duration(elapsed)}"
        if done_w:
            eta, how = left_w * elapsed / done_w, "at this run's pace"
        else:
            eta, how = left_w * SECONDS_PER_SOURCE / self.workers, "first guess, from about 3 GPU hours per dataset and seed"
        end = time.strftime("%a %d %b %H:%M", time.localtime(time.time() + eta))
        return (f"progress: {len(done)} of {len(self.start)} setups done in {duration(elapsed)}; "
                f"about {duration(eta)} left ({how}), finishing around {end}")
