# rl_experiments

RL experiments on small agentic LLMs (LiquidAI LFM2.5-2.6B, LoRA, GRPO-style policy gradients on one GPU).
Every experiment trains on the exact tokens vLLM sampled, recorded by a shared capture proxy.

```
common/                 everything an experiment does not own
  policy.py             LoRA policy: model loading, train_step (IS-capped token-mean PG), vLLM weight sync, resume,
                        shared CLI options (add_args)
  proxy.py              capture proxy: OpenAI-compatible endpoint between a harness and vLLM
  sandbox.py            Docker sandbox (one container, one workspace per rollout; workspaces in runs/workspaces)
  tasks.py              synthetic CSV data-analysis tasks, messy-file variants, grading
  serve_vllm.sh         vLLM policy server (MAX_SEQS, GPU_UTIL, VENV, ... as env vars)
  stop_vllm.sh          stops it by pid (pkill -f would kill the calling shell)
  sandbox_image/        Dockerfile for harness-sandbox:latest (python + pandas, no network)
  blog/                 build.mjs renders any experiment's blog/<slug>.md to HTML; style.css; KaTeX css + fonts
experiments/
  rl_harness/           one agent, three harnesses (shell / pytools / react); blog/multi-harness-rl.md
  multi_agent_rl/       lead + per-site workers, collective reward (RLCR); blog/multi-agent-rl.md
requirements.txt        pinned versions of the shared venv
```

An experiment keeps only what is specific to it: its train loop (`train.py`), rollout/reward code, harnesses or
agents, probe, analysis, blog post and `runs/`. Its scripts put `../../common` on `sys.path`, so run them from
the experiment directory. Each one has a `Makefile` (`make serve`, `make probe`, `make train RUN=...`,
`make figures`, `make blog`) and a `STATUS.md` with its current state and next steps.

## Setup

- Venv: `.venv-lfm` at the repo root (vLLM 0.32, transformers 5.19, torch 2.13 cu130; see `requirements.txt`).
  `VENV=.venv common/serve_vllm.sh` serves Qwen3 from the older `.venv`. Neither venv is tracked.
- Sandbox image: `docker build -t harness-sandbox:latest common/sandbox_image`.
- Blog rendering needs the site's node modules: `NODE_PATH=/home/webpage/node_modules`.
- Run outputs go to `experiments/*/runs/`. Only `config.json` and the `*.jsonl` logs are tracked, which is
  enough to regenerate every figure. Weights, adapters and logs are not tracked.

## Adding an experiment

Create `experiments/<name>/` and import from `common/`. A typical `train.py` builds `policy.Policy(args, run_dir)`,
collects rollouts through `proxy.CaptureProxy`, turns them into `(row, advantage)` samples, and calls
`policy.train_step(samples)` then `policy.sync(session, step, proxy)`. If you change a shared module, keep it
backward compatible, because every experiment uses it.
