"""Multi-harness agentic RL with LoRA on one GPU.

Loop per step: sample tasks -> run GRPO groups of rollouts inside black-box harnesses (one harness per
group) through the capture proxy -> score -> policy-gradient update on the exact captured tokens ->
hot-load the new LoRA adapter into vLLM. vLLM (separate process, serve_vllm.sh) and the trainer share
the GPU and take turns.
"""
import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules: policy, proxy, sandbox, tasks
from policy import Policy, add_args
from proxy import CaptureProxy
from rollout import evaluate, run_rollout, to_rows
from sandbox import Sandbox
from tasks import make_split, make_test

p = argparse.ArgumentParser()
p.add_argument("--run", default="multi")
p.add_argument("--harnesses", default="shell,pytools,react", help="harnesses used for training")
p.add_argument("--eval-harnesses", default="shell,pytools,react")
p.add_argument("--steps", type=int, default=40)
p.add_argument("--n-test", type=int, default=40, help="test tasks per bucket (seen / tmpl / domain)")
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
    train_tasks = make_split(args.n_train, 11)
    test_tasks = make_test(args.n_test)
    train_h = args.harnesses.split(",")
    eval_h = args.eval_harnesses.split(",")
    template_kwargs = {"enable_thinking": False} if "Qwen3" in args.model else {}
    proxy = CaptureProxy(tok, vllm_url=args.vllm, model_name=args.model, temperature=args.temperature,
                         max_tokens=args.max_tokens, max_ctx=args.max_ctx, template_kwargs=template_kwargs)
    await proxy.start()
    sbx = Sandbox(name=f"sbx-{args.run}")
    session = aiohttp.ClientSession()
    log("config.json", vars(args))
    start = args.resume_step + 1
    if args.resume_step:
        await policy.resume(session, proxy)

    async def do_eval(step):
        t0 = time.time()
        res, results = await evaluate(proxy, sbx, test_tasks, eval_h, args.eval_temperature)
        res.update(step=step, time=time.time() - t0, model=proxy.model_name)
        log("eval.jsonl", res)
        log("eval_rollouts.jsonl", {"step": step, "results": results})
        bk = lambda d: "/".join(f"{d.get('acc_' + b, 0):.2f}" for b in ("seen", "tmpl", "domain"))
        print(f"[eval step {step}] (seen/tmpl/domain) " + "  ".join(f"{h}={res[h]['acc']:.3f} [{bk(res[h])}]" for h in eval_h)
              + f"  all={res['all']['acc']:.3f} [{bk(res['all'])}]  calls={res['all']['tool_calls']:.2f}  ({res['time']:.0f}s)", flush=True)

    if not args.resume_step:
        await do_eval(0)
    cursor = args.resume_step * args.tasks_per_step
    for step in range(start, args.steps + 1):
        t0 = time.time()
        batch = train_tasks[cursor:cursor + args.tasks_per_step]
        cursor += args.tasks_per_step
        jobs = []
        for j, task in enumerate(batch):
            h = train_h[((step - 1) * args.tasks_per_step + j) % len(train_h)]
            jobs += [run_rollout(proxy, sbx, task, h) for _ in range(args.group)]
        results = await asyncio.gather(*jobs)
        t_roll = time.time() - t0

        samples, flat_groups = [], 0
        per_h = defaultdict(list)
        for g in range(len(batch)):
            grp = results[g * args.group:(g + 1) * args.group]
            rs = [r["reward"] for r in grp]
            mu = sum(rs) / len(rs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in rs) / len(rs))
            if sd < 1e-6:
                flat_groups += 1
            for r in grp:
                per_h[r["harness"]].append(r)
                a = (r["reward"] - mu) / (sd + args.adv_std_floor)
                if abs(a) > 1e-6:
                    samples += [(row, a) for row in to_rows(r["calls"])]
        t1 = time.time()
        stats = policy.train_step(samples) if samples else {}
        torch.cuda.empty_cache()
        await policy.sync(session, step, proxy)
        m = {"step": step, "reward": sum(r["reward"] for r in results) / len(results),
             "acc": sum(r["correct"] for r in results) / len(results),
             "tool_calls": sum(r["n_tool_calls"] for r in results) / len(results),
             "flat_groups": flat_groups, "rows": len(samples),
             "calls_per_rollout": sum(len(r["calls"]) for r in results) / len(results),
             "t_rollout": t_roll, "t_train": time.time() - t1,
             **{f"acc_{h}": sum(r["correct"] for r in rs) / len(rs) for h, rs in per_h.items()},
             **{f"calls_{h}": sum(r["n_tool_calls"] for r in rs) / len(rs) for h, rs in per_h.items()},
             **stats}
        log("metrics.jsonl", m)
        print(f"step {step:3d} reward={m['reward']:.3f} acc={m['acc']:.3f} calls={m['tool_calls']:.2f} "
              f"flat={flat_groups}/{len(batch)} rows={len(samples)} tok={stats.get('seq_tokens', 0)} "
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
