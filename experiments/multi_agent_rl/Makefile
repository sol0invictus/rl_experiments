# Common tasks. Python is the repo-root LFM venv; shared scripts live in ../../common (see CLAUDE.md).
COMMON = ../../common
PY    ?= ../../.venv-lfm/bin/python
RUN   ?= org_all
ARGS  ?=
N     ?= 10

.PHONY: test tasks smoke image serve stop-vllm probe show train ablation table figures blog clean

test: tasks smoke                      ## no-GPU checks

tasks:                                 ## shards reassemble, held-out splits are disjoint
	$(PY) org_tasks.py

smoke:                                 ## end-to-end with a scripted fake vLLM + real Docker sandbox
	$(PY) tests/smoke_offline.py

image:                                 ## build the sandbox image (python + pandas, no network)
	docker build -t harness-sandbox:latest $(COMMON)/sandbox_image

serve:                                 ## vLLM policy server on :8000 (background)
	mkdir -p runs && MAX_SEQS=192 nohup $(COMMON)/serve_vllm.sh > runs/vllm.log 2>&1 & echo "vLLM starting, log: runs/vllm.log"

stop-vllm:
	$(COMMON)/stop_vllm.sh

probe:                                 ## base-model org vs solo, N tasks per bucket
	mkdir -p runs && $(PY) probe.py --n $(N) --out runs/probe_base.jsonl $(ARGS)

show:                                  ## print 3 full organisation transcripts
	$(PY) probe.py --show 3 $(ARGS)

train:                                 ## RLCR training run in the background
	mkdir -p runs && nohup $(PY) train.py --run $(RUN) $(ARGS) > runs/$(RUN).log 2>&1 & echo "log: runs/$(RUN).log"

ablation:                              ## same setup, only the lead's tokens are trained
	$(MAKE) train RUN=$(RUN)_lead ARGS="--train-roles lead $(ARGS)"

table:                                 ## markdown results for runs/$(RUN)
	$(PY) analysis/summarize_run.py $(RUN) $(ARGS)

figures:                               ## regenerate blog figures + results JSON
	$(PY) analysis/make_blog_figures.py $(ARGS)

blog:                                  ## re-render blog/multi-agent-rl.html
	NODE_PATH=/home/webpage/node_modules node $(COMMON)/blog/build.mjs blog/multi-agent-rl.md

clean:                                 ## leftover sandbox workspaces and caches (keeps runs)
	rm -rf ../../runs/workspaces __pycache__ */__pycache__ $(COMMON)/__pycache__
