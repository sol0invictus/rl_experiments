# Common tasks. Python is the repo-root LFM venv; shared scripts live in ../../common.
COMMON = ../../common
PY    ?= ../../.venv-lfm/bin/python
RUN   ?= lfm_multi
ARGS  ?=
N     ?= 10

.PHONY: image serve stop-vllm probe show train figures blog clean

image:                                 ## build the sandbox image (python + pandas, no network)
	docker build -t harness-sandbox:latest $(COMMON)/sandbox_image

serve:                                 ## vLLM policy server on :8000 (background)
	mkdir -p runs && MAX_SEQS=128 nohup $(COMMON)/serve_vllm.sh > runs/vllm.log 2>&1 & echo "vLLM starting, log: runs/vllm.log"

stop-vllm:
	$(COMMON)/stop_vllm.sh

probe:                                 ## base model on the 3 harnesses, N tasks per bucket
	mkdir -p runs && $(PY) probe.py --n $(N) --out runs/probe_base.jsonl $(ARGS)

show:                                  ## raw completions of 3 rollouts per harness
	$(PY) probe.py --show 3 $(ARGS)

train:                                 ## GRPO training run in the background
	mkdir -p runs && nohup $(PY) train.py --run $(RUN) $(ARGS) > runs/$(RUN).log 2>&1 & echo "log: runs/$(RUN).log"

figures:                               ## regenerate blog figures + analysis/results_$(RUN).json
	$(PY) analysis/make_blog_figures.py --run $(RUN) --out blog/multi-harness-rl $(ARGS)

blog:                                  ## re-render blog/multi-harness-rl.html
	NODE_PATH=/home/webpage/node_modules node $(COMMON)/blog/build.mjs blog/multi-harness-rl.md

clean:                                 ## leftover sandbox workspaces and caches (keeps runs)
	rm -rf ../../runs/workspaces __pycache__ */__pycache__ $(COMMON)/__pycache__
