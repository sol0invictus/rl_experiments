# rl_experiments

RL experiments on small agentic LLMs (LiquidAI LFM2.5-2.6B, LoRA, GRPO-style policy gradients). Every
experiment trains on the exact tokens vLLM sampled, recorded by a shared capture proxy.

```
common/
  proxy.py           capture proxy: OpenAI-compatible endpoint between a harness and vLLM (shared)
  sandbox_image/     Dockerfile for the network-less python+pandas sandbox (harness-sandbox:latest)
experiments/
  rl_harness/        single agent, several harnesses (multi-harness LoRA GRPO); blog: blog/multi-harness-rl.md
  multi_agent_rl/    lead + site workers with a collective reward (RLCR); blog: blog/multi-agent-rl.md
```

Each experiment is self-contained apart from `common/`. Its scripts put `../../common` on `sys.path`, so
run them from the experiment directory as before (for example `cd experiments/multi_agent_rl && make test`).
Each experiment's `STATUS.md` holds its current state and next steps.

Setup:

- Build the sandbox image: `docker build -t harness-sandbox:latest common/sandbox_image`.
- Both experiments use the LFM venv at `experiments/rl_harness/.venv-lfm` (vLLM 0.32, transformers 5.19).
  Pinned versions are in `experiments/multi_agent_rl/requirements.txt`.
- Run outputs go to `experiments/*/runs/`. Only `config.json` and the `*.jsonl` logs are tracked, which is
  enough to regenerate the figures. Weights, adapters and logs are not tracked.
