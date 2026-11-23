#!/bin/bash
# Policy server for every experiment's train.py / probe.py. Shares the 3090 Ti with the trainer.
# LFM2.5 env at the repo root: .venv-lfm (vLLM 0.32, transformers 5.19, torch 2.13 cu130); VENV=.venv serves Qwen3.
# MAX_SEQS: rl_harness ran with 128, multi_agent_rl with 192 (its Makefile sets it).
MODEL=${MODEL:-LiquidAI/LFM2.5-2.6B}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-.venv-lfm}   # relative to the repo root
# Only for train.py --sync lora (Qwen): LORA_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --max-cpu-loras 4"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_SERVER_DEV_MODE=1          # /collective_rpc reload_weights + /reset_prefix_cache (train.py --sync merge)
export VLLM_USE_FLASHINFER_SAMPLER=0   # its JIT build needs a newer nvcc than the system one
exec "$ROOT/$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --port ${PORT:-8000} \
  --gpu-memory-utilization ${GPU_UTIL:-0.45} --max-model-len ${MAX_LEN:-16384} --max-num-seqs ${MAX_SEQS:-128} \
  ${LORA_ARGS} \
  --return-tokens-as-token-ids --dtype bfloat16 ${EXTRA_ARGS} \
  --generation-config vllm   # ignore LFM generation_config.json (rep penalty 1.1, top_k 50): RL needs plain sampling
