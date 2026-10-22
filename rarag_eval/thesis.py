"""The thesis repository does the work (RA-RAG's pipeline, the SL model, the lab's scenarios); this
package only drives it. This module finds it, imports it with the right environment, and can block
every LLM call so that runs reuse cached answers only."""
import os
import pickle
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def thesis_dir():
    """THESIS_REPO, else the thesis repository next to this one."""
    path = os.environ.get("THESIS_REPO") or os.path.join(os.path.dirname(ROOT), "Ra-Rag-Thesis-Adaptation")
    if not os.path.isdir(os.path.join(path, "webapp", "backend")):
        raise SystemExit(f"thesis repository not found at {path}: set THESIS_REPO")
    return os.path.abspath(path)


def commit(path=None):
    """The thesis repository's commit, with "+dirty" when it has uncommitted changes."""
    path = path or thesis_dir()
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


class Thesis:
    """The thesis modules, loaded once: lab (scenarios, the LLM, embeddings), sl_minimal (the SL model and
    the Reliability only / SL vs RA-RAG analyses), the answer matching."""

    def __init__(self, model, runs_dir=None, offline=False, embeddings=None, nli_memo=None):
        path = thesis_dir()
        os.environ["LAB_RUNS_DIR"] = os.path.abspath(runs_dir or os.path.join(path, "vector_cache"))
        os.environ["WEBAPP_MODEL"] = model
        for p in (path, os.path.join(path, "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from webapp.backend import lab, sl_lab, sl_minimal  # noqa: E402
        from src.matching import fair_match  # noqa: E402
        self.lab, self.sl_lab, self.sl_minimal, self.fair_match = lab, sl_lab, sl_minimal, fair_match
        self.model, self.path, self.commit, self.offline = model, path, commit(path), offline
        self.runs_dir = os.environ["LAB_RUNS_DIR"]
        self._keep_big_results_off_disk()
        if offline:
            self._block_llm()
        if embeddings:
            from src.vector_store import ContrieverEncoder  # noqa: E402
            lab.RES.embeddings = lab.PersistentEmbeddings(ContrieverEncoder, path=embeddings)
        self.nli_path = nli_memo
        self._nli = None

    def _keep_big_results_off_disk(self):
        """The lab saves every analysis whole (about 46 MB each); here only the compact summaries are kept."""
        cls = self.sl_minimal.MinimalRun
        save = cls._save

        def skip(path, data):
            if not os.path.basename(path).startswith(("minimal_swap__", "minimal_warmup__", "minimal_comparison__")):
                save(path, data)
        cls._save = staticmethod(skip)

    def _block_llm(self):
        def refuse(*args, **kwargs):
            raise RuntimeError("offline: this needs LLM answers that are not cached")
        self.lab.Resources.handle = refuse
        self.lab.Scenario.estimate = refuse
        self.sl_lab.AnswerBank._compute = refuse

    # the entailment model's decisions are kept between runs: the first analysis of a setup computes them,
    # every later one reuses them
    def nli(self):
        if self._nli is None:
            self._nli = self.lab.RES.model_cache().entailment()
            if self.nli_path and os.path.exists(self.nli_path):
                with open(self.nli_path, "rb") as f:
                    self._nli._memo.update(pickle.load(f))
        return self._nli

    def save_nli(self):
        if self._nli is None or not self.nli_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.nli_path)), exist_ok=True)
        tmp = self.nli_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._nli._memo, f)
        os.replace(tmp, self.nli_path)
