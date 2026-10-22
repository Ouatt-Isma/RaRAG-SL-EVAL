# RaRAG-SL-EVAL

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

Three ways, depending on the machine. In all of them `make run` resumes where it stopped (saved analyses are
skipped), `make dry` lists what is left, and `make report` can be run at any time, also during a run.

### 1. On a machine with its own GPU, used directly (a workstation or a laptop)

Linux, or Windows with WSL 2, an NVIDIA driver installed (`nvidia-smi` works). Nothing is copied anywhere: run and
report on the same machine.

Both repositories are private: log in to GitHub on the machine first (`gh auth login`, or a personal access token
when git asks for a password).

```bash
# 1. this repo, then the thesis repo beside it (same parent folder: this repo finds it there)
git clone https://github.com/Ouatt-Isma/RaRAG-SL-EVAL.git
git clone -b feature/sl-minimal-model https://github.com/ATR-oCiTy/Ra-Rag-Thesis-Adaptation.git

# 2. the thesis environment and the LLM (Ollama's installer may ask for your sudo password)
cd Ra-Rag-Thesis-Adaptation
bash scripts/setup_server.sh        # Python 3.11 .venv with the tested packages, AlignScore's checkpoint (1.9 GB)
bash scripts/setup_ollama.sh        # Ollama and Llama 3 8B fp16 (16 GB download), then one test answer
cd ../RaRAG-SL-EVAL

# 3. run (inside tmux or screen, so closing the terminal does not stop it)
make setup                          # once: the reporting environment
make gpus                           # check: the GPUs found and the workers it will start
make run CONFIG=configs/pilot.yaml  # a short check first (NQ, seed 0, 6 setups)
make run                            # the confirmation grid (configs/confirm.yaml)
make report                         # tables, figures and report/results.pdf from what has finished
```

**GPU memory.** The 16-bit model with the lab's models needs about 18.5 GB, so a 24 GB GPU (RTX 3090/4090, A5000,
…). Most laptop GPUs have 8–16 GB: there Ollama keeps part of the model on the CPU and runs several times slower.
`OLLAMA_MODEL=llama3:8b-instruct-q8_0 make run` fits in 16 GB, but it is not the model of the earlier runs: run it
in its own copy of this repo so its results never mix with the fp16 ones (`make report` warns when they do).

**Apple silicon (a Mac).** It runs, with no setup scripts: install [Ollama](https://ollama.com) and the thesis
environment as in its README, `ollama pull llama3:8b-instruct-fp16`, then `make run`. Ollama uses the Mac's GPU,
AlignScore and the entailment model use it too (MPS), the retriever runs on the CPU. The fp16 model needs about 17 GB
of the Mac's shared memory, so at least a 32 GB Mac, and it is far slower than a 4090: use it for a pilot, not the grid.

### 2. On a rented GPU server (vast.ai)

The thesis repo and its environment set up as in its `SETUP.md` ("Rented GPU server"):

```bash
# from the laptop: the thesis code (its own script) and this repo
bash ../Ra-Rag-Thesis-Adaptation/scripts/server_backup.sh code <host> <port>
make push HOST=<host> PORT=<port>
# on the server (in tmux), after the thesis scripts/setup_server.sh and scripts/setup_ollama.sh
cd /workspace/rarag-sl-eval && make run THESIS=/workspace/Ra-Rag-Thesis-Adaptation
# back on the laptop, whenever you want the latest
make pull HOST=<host> PORT=<port> && make report
```

### 3. On a laptop without a usable GPU, from cached answers

No LLM: only setups whose answers a server already produced (e.g. a server backup).

```bash
make setup                                   # once: the reporting environment
make run-offline CONFIG=configs/pilot.yaml RUNS_DIR=../Ra-Rag-Thesis-Adaptation/server_backup/vector_cache \
                 EMB=../Ra-Rag-Thesis-Adaptation/server_backup/llm_cache/embeddings_v2.pkl
make report CONFIG=configs/pilot.yaml
```

### What `make run` does on the GPUs

It finds the GPUs (`nvidia-smi`) and starts workers: one per 21 GB of a GPU's memory (one on a 24 GB RTX 4090, two on
a 48 GB A6000, three on an 80 GB A100/H100), each with its own Ollama server pinned to its GPU (ports 11500, 11501, …;
4 requests at once, the setting measured best on a 24 GB card), the LLM, AlignScore, the entailment model and the
retriever all on that GPU. Even with one GPU it starts its own Ollama, so the settings are the same on every machine
whatever the installed Ollama service is set to; a model that service holds on a GPU is unloaded first (the service
keeps running), and the run's own servers stop when it ends. The workers take the grid's setups one at a time from a
shared queue (`results/.claims/`), so they finish together.

One dataset and seed of the full grid is about 3 GPU hours (about 50,000 LLM answers): the `confirm.yaml` grid is about
27 GPU hours, so about 27 h on one 24 GB GPU, 14 h on 2, 7 h on 4. The analyses on cached answers take minutes.

`make gpus` shows the plan without starting anything; `GPUS=2` uses the first two GPUs, `PER_GPU=1` one worker per GPU.
Logs: `logs/run.log` (the launch, and the run itself with one worker), `logs/run.w<N>.log` (each worker when there are
several), `logs/ollama.<port>.log`.

The LLM is `ollama_model` in the config (`llama3:8b-instruct-fp16`, the tag the earlier server runs used, so cached
answers stay comparable); `OLLAMA_MODEL=…` overrides it, and every result records the tag it ran with.

`make test` checks the reporting path and the GPU planning on synthetic inputs.

### Example documents

`make examples` (same `CONFIG`, `RUNS_DIR`, `EMB` as `make run-offline`) writes, for 3 test questions of every setup (an
attacked one first where there are attackers), `examples/<dataset>/seed<N>/<setup>/question_<idx>.txt`: what each
source holds for the question, the passages it retrieved, the LLM's answer and whether AlignScore kept it; and
`examples/<dataset>/documents_<idx>.txt`: every document generated for that question (gold paraphrases, each wrong
answer's documents, the attackers' documents). It reads cached answers only. `--questions N` changes the count.

## Adding a dataset

A dataset is a file in the thesis repo's `dataset/` built by its `scripts/build_dataset.py` (questions, gold answers,
true and false documents per question); the `experts` setup also needs its domain labels
(`scripts/label_domains.py`). Then add it to the config's `datasets`. The fact-checking sets (LIAR-RAW, RAWFC: claims
with a true / half-true / false verdict) need a claims build first: true documents from each claim's explanation,
false ones arguing the other verdicts, generated by the local LLM (about 3–4 GPU hours per dataset).
