# Run on a GPU server (the thesis repo beside this one, its venv, Ollama running), report anywhere with LaTeX.
CONFIG ?= configs/confirm.yaml
THESIS ?= ../Ra-Rag-Thesis-Adaptation
RUN_PY ?= $(THESIS)/.venv/bin/python
# the thesis repo's environment: on a new server, the thesis repo's setup scripts make it (README, step 2)
NEED_THESIS = test -x $(RUN_PY) || { echo "No thesis environment at $(RUN_PY). Set up the thesis repo first, once:"; \
  echo "  cd $(THESIS) && bash scripts/setup_server.sh && bash scripts/setup_ollama.sh"; exit 1; }
PY ?= .venv/bin/python
HOST ?=
PORT ?= 22
REMOTE ?= /workspace
SESSION ?= eval

.PHONY: setup run start attach save gpus status claims claims-check claims-status run-offline dry report report-nopdf examples test push pull clean

setup:                 ## the reporting environment (matplotlib, scipy, pyyaml)
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

run:                   ## the grid, with the LLM, on every GPU of the server (GPUS=n: the first n; PER_GPU=n workers each)
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.launch $(CONFIG) $(if $(GPUS),--gpus $(GPUS)) $(if $(PER_GPU),--per-gpu $(PER_GPU)) $(if $(ONLY),--only $(ONLY))

start:                 ## make run inside tmux: it keeps going when the terminal closes or the connection drops (ONLY="dataset=… seed=… setup=…": part of the grid)
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.launch $(CONFIG) --tmux --session $(SESSION) $(if $(GPUS),--gpus $(GPUS)) $(if $(PER_GPU),--per-gpu $(PER_GPU)) $(if $(ONLY),--only $(ONLY))

attach:                ## watch the run make start started (Ctrl-b then d: leave it running)
	tmux attach -t $(SESSION)

save:                  ## commit the results so far and push them (any time, also during a run); the laptop: git pull && make report
	git add -A -- results examples ':!results/rows.csv' ':!results/claims.csv'
	@if git diff --cached --quiet; then echo "nothing new to save"; else \
	  git commit -q -m "Results: $$(find results -name '*.json.gz' | wc -l | tr -d ' ') analyses ($$(hostname), $$(date '+%Y-%m-%d %H:%M'))" && \
	  git push && echo "saved and pushed: $$(git log -1 --format=%s)"; fi

status:                ## how far make run is and how long it has left (from any terminal)
	@$(NEED_THESIS)
	@$(RUN_PY) -m rarag_eval.launch $(CONFIG) --status

claims:                ## optional: build the fact-checking datasets RAWFC and LIAR-RAW (local LLM, about 3-4 h each), in tmux
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.factcheck $(CONFIG) --tmux

claims-check:          ## optional, first: build 3 RAWFC claims and print them (a few minutes; nothing kept)
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.factcheck $(CONFIG) --check 3

claims-status:         ## how far make claims is
	@grep -E "built,|FAILED|done:|already built" logs/claims.log 2>/dev/null | tail -n 2 || echo "no claims build started here yet"

gpus:                  ## the GPUs found and the workers make run would start
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.launch $(CONFIG) --plan $(if $(GPUS),--gpus $(GPUS)) $(if $(PER_GPU),--per-gpu $(PER_GPU))

run-offline:           ## the grid from cached answers only (laptop): RUNS_DIR=<a server backup's vector_cache>
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.run $(CONFIG) --offline $(if $(RUNS_DIR),--runs-dir $(RUNS_DIR)) $(if $(EMB),--embeddings $(EMB))

dry:                   ## what the grid still has to run
	@$(NEED_THESIS)
	$(RUN_PY) -m rarag_eval.run $(CONFIG) --dry-run $(if $(ONLY),--only $(ONLY))

report:                ## tables, figures and report/results.pdf from results/
	$(PY) -m rarag_eval.report $(CONFIG)

report-nopdf:
	$(PY) -m rarag_eval.report $(CONFIG) --no-pdf

examples:              ## example source documents and answers as .txt in examples/ (cached answers only)
	@$(NEED_THESIS)
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
