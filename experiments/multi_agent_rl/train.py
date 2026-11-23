"""Multi-agent RL with LoRA on one GPU: reinforcement learning from collective rewards (Avalon-M1's RLCR,
scaled down to a 2.6B model and a single 3090 Ti).

Loop per step: sample partitioned-data tasks -> run a GRPO group of whole organisations per task (a lead
plus one worker per site, all the same policy) through the capture proxy -> score each organisation's
submitted answer -> give every sampled token of every agent in the organisation the same group-relative
advantage -> policy-gradient update on the exact captured tokens -> push merged weights to vLLM.
vLLM (separate process, serve_vllm.sh) and the trainer share the GPU and take turns.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import aiohttp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules: policy, proxy, sandbox, tasks
from policy import Policy, add_args
from proxy import CaptureProxy
from org_tasks import TEST_BUCKETS, make_org_split, make_org_test
from rollout import collective_samples, evaluate, run_rollout, summarize
from sandbox import Sandbox

p = argparse.ArgumentParser()
p.add_argument("--run", default="org")
p.add_argument("--mode", default="org", choices=["org", "solo"], help="what is trained: organisations or one agent")
p.add_argument("--eval-modes", default="org,solo")
p.add_argument("--train-roles", default="all", choices=["all", "lead", "worker"],
               help="whose tokens get the collective advantage: all agents (Avalon RLCR) or only the lead "
                    "(Kimi PARL freezes sub-agents) / only the workers")
p.add_argument("--eff-bonus", type=float, default=0.1, help="max reward bonus for a short critical path")
p.add_argument("--steps", type=int, default=60)
p.add_argument("--n-test", type=int, default=30, help="test tasks per bucket")
p.add_argument("--eval-buckets", default=",".join(TEST_BUCKETS), help="subset of seen,tmpl,domain,wide")
args = add_args(p).parse_args()

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run)
os.makedirs(RUN_DIR, exist_ok=True)
random.seed(args.seed)
torch.manual_seed(args.seed)


def log(name, obj):
    with open(os.path.join(RUN_DIR, name), "a") as f:
        f.write(json.dumps(obj) + "\n")


policy = Policy(args, RUN_DIR)   # LoRA model, train_step, vLLM weight sync (common/policy.py)
tok = policy.tok


# --------------------------------------------------------------------------------------- main
async def main():
    train_tasks = make_org_split(args.n_train, 21)
    test_tasks = make_org_test(args.n_test, buckets=args.eval_buckets.split(","))
    eval_modes = args.eval_modes.split(",")
    template_kwargs = {"enable_thinking": False} if "Qwen3" in args.model else {}
    proxy = CaptureProxy(tok, vllm_url=args.vllm, model_name=args.model, temperature=args.temperature,
                         max_tokens=args.max_tokens, max_ctx=args.max_ctx, template_kwargs=template_kwargs)
    await proxy.start()
    sbx = Sandbox(name=f"sbx-{args.run}", max_concurrent=32)
    session = aiohttp.ClientSession()
    log("config.json", vars(args))
    start = args.resume_step + 1
    if args.resume_step:
        await policy.resume(session, proxy)

    async def do_eval(step):
        t0 = time.time()
        res, results = await evaluate(proxy, sbx, test_tasks, eval_modes, args.eval_temperature)
        res.update(step=step, time=time.time() - t0, model=proxy.model_name)
        log("eval.jsonl", res)
        log("eval_rollouts.jsonl", {"step": step, "results": results})
        bk = lambda d: "/".join(f"{d.get('acc_' + b, float('nan')):.2f}" for b in TEST_BUCKETS)
        line = []
        for m in eval_modes:
            d = res[m]
            line.append(f"{m}={d['acc']:.3f} [{bk(d)}] lat={d['latency']:.1f}"
                        + (f" cov={d['coverage']:.2f} fan={d['max_fanout']:.2f}" if m == "org" else ""))
        print(f"[eval step {step}] ({'/'.join(TEST_BUCKETS)}) " + "  ".join(line) + f"  ({res['time']:.0f}s)",
              flush=True)

    if not args.resume_step:
        await do_eval(0)
    cursor = args.resume_step * args.tasks_per_step
    for step in range(start, args.steps + 1):
        t0 = time.time()
        batch = train_tasks[cursor:cursor + args.tasks_per_step]
        cursor += args.tasks_per_step
        jobs = [run_rollout(proxy, sbx, task, args.mode, args.eff_bonus) for task in batch for _ in range(args.group)]
        results = await asyncio.gather(*jobs)
        t_roll = time.time() - t0

        samples, info = collective_samples(results, args.group, args.train_roles, args.adv_std_floor)
        t1 = time.time()
        stats = policy.train_step(samples) if samples else {}
        torch.cuda.empty_cache()
        await policy.sync(session, step, proxy)
        summ = summarize(results)
        m = {"step": step, "reward": sum(r["reward"] for r in results) / len(results),
             **{k: v for k, v in summ.items() if k not in ("statuses", "tokens", "n")},
             **info, "rows": len(samples),
             **{f"gen_tokens_{k}": v for k, v in summ["tokens"].items()},
             **{f"acc_k{k}": sum(r["correct"] for r in results if r["n_sites"] == k)
                / max(1, sum(r["n_sites"] == k for r in results)) for k in sorted({r["n_sites"] for r in results})},
             "t_rollout": t_roll, "t_train": time.time() - t1, **stats}
        log("metrics.jsonl", m)
        print(f"step {step:3d} reward={m['reward']:.3f} acc={m['acc']:.3f} lat={m['latency']:.1f} "
              + (f"cov={m['coverage']:.2f} fan={m['max_fanout']:.2f} asks={m['asks']:.1f} " if args.mode == "org" else "")
              + f"flat={info['flat_groups']}/{len(batch)} err={info['errors']} rows={len(samples)} tok={stats.get('seq_tokens', 0)} "
              f"kl={stats.get('mismatch_kl', 0):.4f} gn={stats.get('grad_norm', 0):.3f} "
              f"roll={t_roll:.0f}s train={m['t_train']:.0f}s mem={stats.get('peak_mem_gb', 0):.1f}G", flush=True)
        for r in results:
            r.pop("calls")
        log("train_rollouts.jsonl", {"step": step, "results": results})
        if step % args.eval_every == 0 or step == args.steps:
            await do_eval(step)
    await session.close()
    await proxy.stop()
    sbx.close()


asyncio.run(main())
