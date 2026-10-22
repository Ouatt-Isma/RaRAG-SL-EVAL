# rarag-sl-eval

Headless evaluation of trust for multi-source RAG: RA-RAG's reliability estimate against subjective-logic (SL)
trust, under attack and with domain experts, **across datasets and seeds**. It confirms (or not) the results of
the first evaluation series, saves every result, and turns them into what a paper needs:

| Output | Where |
|---|---|
| Saved results, one compressed file per analysis | `results/<dataset>/seed<N>/<setup>/<analysis>.json.gz` |
| Flat tables of every row and every claim | `results/rows.csv`, `results/claims.csv` |
| LaTeX tables (booktabs, `\input` them) | `paper/tables/*.tex` |
| Figures (vector PDF) | `paper/figures/*.pdf` |
| The results document, rebuilt from the above | `report/results.pdf` |
| Example source documents and answers (plain text) | `examples/` (start with `examples/INDEX.txt`) |

No interface: one command runs the grid, another rebuilds the outputs.

## How it works

The thesis repository ([Ra-Rag-Thesis-Adaptation](../Ra-Rag-Thesis-Adaptation), branch `feature/sl-minimal-model`)
does the work: building each source's documents, asking the LLM (Llama 3 8B through Ollama), AlignScore, RA-RAG's
estimator and voting, and the SL model. This repo only drives it, so there is one implementation of every method.
It looks for the thesis repo beside this one (or `THESIS_REPO=…`) and records its commit in every result.

A config (`configs/confirm.yaml`) lists **datasets × seeds × setups**. Each setup is one scenario (one set of sources,
one round of LLM answers) per dataset and seed; its **analyses** all reuse those answers:

| Setup | Sources | Analyses |
|---|---|---|
| `unreliable_k3` | 3 unreliable (right 10%) + 6 reliable (90%): the paper's Table A4 mix | plain, aged, window, SL's own answering |
| `unreliable_k7` | 7 unreliable + 2 reliable | same |
| `mix5` | 5 honest sources, 0.53–0.91 | same + warm-up subsets of 5/10/20 questions × 10 draws |
| `attackers_k1…k4` | k build-then-betray attackers among 9 | plain, aged, window, detector, SL's own answering |
| `switch` | 3 attackers honest until halfway through the test | same |
| `experts` | 5 domain experts (90% in their domain, 30% elsewhere) + 2 generalists | plain, SL's own answering + trust per domain placed 4 ways (none, LLM label, nearest centroid, soft weights) × warm-up 15/60/full |

Analyses (`rarag_eval/analyses.py`): **swap** = RA-RAG's answering (κ = 4) with each reliability in turn (RA-RAG
frozen/online/window, SL frozen/online, oracle, equal weights); **compare** = SL's own answering (answer, unknown,
conflict). *plain* = SL with the agreement rule; *aged* = aging λ 0.97 and RA-RAG's 100-question window; *window* =
SL's 100-question window; *detector* = aging plus the switch detector at θ 0.25. Every row keeps each test question's outcome,
so two rows are compared question by question (McNemar's test, paired bootstrap).

`rarag_eval/claims.py` turns the first series' results (rows 1–15 of the summary) into comparisons; the report gives
each one per dataset, over all runs (mean, 95% interval) and how many runs agree.

## Running

**On a GPU server** (one RTX 4090 is enough; the thesis repo and its environment set up as in its `SETUP.md`):

```bash
# from the laptop: the thesis code (its own script) and this repo
bash ../Ra-Rag-Thesis-Adaptation/scripts/server_backup.sh code <host> <port>
make push HOST=<host> PORT=<port>
# on the server, Ollama running (thesis scripts/setup_ollama.sh)
cd /workspace/rarag-sl-eval && make run CONFIG=configs/confirm.yaml THESIS=/workspace/Ra-Rag-Thesis-Adaptation
# back on the laptop, whenever you want the latest
make pull HOST=<host> PORT=<port> && make report
```

`make run` resumes where it stopped (saved analyses are skipped) and logs to `logs/run.log`. `make dry` lists what is
left. One dataset and seed of the full grid is about 3 GPU hours (about 50,000 LLM answers); the analyses on cached
answers take minutes.

**Several GPUs.** `make run` finds the GPUs (`nvidia-smi`) and starts workers for them: one per 21 GB of a GPU's memory
(one on a 24 GB RTX 4090, two on a 48 GB A6000, three on an 80 GB A100/H100), each with its own Ollama server pinned to
its GPU (ports 11500, 11501, …), the LLM, AlignScore, the entailment model and the retriever all on that GPU. The
workers take the grid's setups one at a time from a shared queue (`results/.claims/`), so they finish together: the
`confirm.yaml` grid (about 27 GPU hours) takes about 14 h on 2 GPUs and 7 h on 4. A model the default Ollama (port
11434) holds on a GPU is unloaded first (the server keeps running); the workers' Ollama servers stop when the run ends.
`make gpus` shows the plan without starting anything; `GPUS=2` uses the first two, `PER_GPU=1` one worker per GPU.
Logs: `logs/run.log` (the launch), `logs/run.w<N>.log` (each worker), `logs/ollama.<port>.log`. With one GPU it is
the plain single run on the Ollama already running.

The LLM is `ollama_model` in the config (`llama3:8b-instruct-fp16`, the tag the earlier server runs used, so cached
answers stay comparable); `OLLAMA_MODEL=…` overrides it, and every result records the tag it ran with.

**On the laptop, from cached answers** (no LLM: setups whose answers a server already produced, e.g. a server backup):

```bash
make setup                                   # once: the reporting environment
make run-offline CONFIG=configs/pilot.yaml RUNS_DIR=../Ra-Rag-Thesis-Adaptation/server_backup/vector_cache \
                 EMB=../Ra-Rag-Thesis-Adaptation/server_backup/llm_cache/embeddings_v2.pkl
make report CONFIG=configs/pilot.yaml
```

`make test` checks the reporting path on synthetic results.

**Example documents.** `make examples` (same `CONFIG`, `RUNS_DIR`, `EMB` as above) writes, for 3 test questions of
every setup (an attacked one first where there are attackers), `examples/<dataset>/seed<N>/<setup>/question_<idx>.txt`:
what each source holds for the question, the passages it retrieved, the LLM's answer and whether AlignScore kept it;
and `examples/<dataset>/documents_<idx>.txt`: every document generated for that question (gold paraphrases, each
wrong answer's documents, the attackers' documents). It reads cached answers only. `--questions N` changes the count.

## Adding a dataset

A dataset is a file in the thesis repo's `dataset/` built by its `scripts/build_dataset.py` (questions, gold answers,
true and false documents per question); the `experts` setup also needs its domain labels
(`scripts/label_domains.py`). Then add it to the config's `datasets`. The fact-checking sets (LIAR-RAW, RAWFC: claims
with a true / half-true / false verdict) need a claims build first: true documents from each claim's explanation,
false ones arguing the other verdicts, generated by the local LLM (about 3–4 GPU hours per dataset).
