# Run on a GPU server (the thesis repo beside this one, its venv, Ollama running), report anywhere with LaTeX.
CONFIG ?= configs/confirm.yaml
THESIS ?= ../Ra-Rag-Thesis-Adaptation
RUN_PY ?= $(THESIS)/.venv/bin/python
PY ?= .venv/bin/python
HOST ?=
PORT ?= 22
REMOTE ?= /workspace

.PHONY: setup run run-offline dry report report-nopdf examples test push pull clean

setup:                 ## the reporting environment (matplotlib, scipy, pyyaml)
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

run:                   ## the grid, with the LLM (on the GPU server)
	$(RUN_PY) -m rarag_eval.run $(CONFIG)

run-offline:           ## the grid from cached answers only (laptop): RUNS_DIR=<a server backup's vector_cache>
	$(RUN_PY) -m rarag_eval.run $(CONFIG) --offline $(if $(RUNS_DIR),--runs-dir $(RUNS_DIR)) $(if $(EMB),--embeddings $(EMB))

dry:                   ## what the grid still has to run
	$(RUN_PY) -m rarag_eval.run $(CONFIG) --dry-run

report:                ## tables, figures and report/results.pdf from results/
	$(PY) -m rarag_eval.report $(CONFIG)

report-nopdf:
	$(PY) -m rarag_eval.report $(CONFIG) --no-pdf

examples:              ## example source documents and answers as .txt in examples/ (cached answers only)
	$(RUN_PY) -m rarag_eval.examples $(CONFIG) $(if $(RUNS_DIR),--runs-dir $(RUNS_DIR)) $(if $(EMB),--embeddings $(EMB))

test:
	$(PY) -m pytest tests -q

push:                  ## this repo's code to the server (HOST=… PORT=…); the thesis repo goes with its own server_backup.sh
	rsync -az -e "ssh -p $(PORT)" --exclude .venv --exclude .git --exclude cache --exclude logs --exclude report/build \
	  ./ root@$(HOST):$(REMOTE)/rarag-sl-eval/

pull:                  ## the server's results and run log back here
	rsync -az -e "ssh -p $(PORT)" root@$(HOST):$(REMOTE)/rarag-sl-eval/results/ results/
	rsync -az -e "ssh -p $(PORT)" root@$(HOST):$(REMOTE)/rarag-sl-eval/logs/ logs/
	-rsync -az -e "ssh -p $(PORT)" root@$(HOST):$(REMOTE)/rarag-sl-eval/examples/ examples/

clean:
	rm -rf report/build
