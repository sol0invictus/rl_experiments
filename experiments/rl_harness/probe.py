"""Base-model probe: run harnesses on test tasks without training, print per-bucket accuracy and,
with --show, the raw model text of a few rollouts (to debug a harness's format)."""
import sys
from pathlib import Path
import argparse
import asyncio
import json
from collections import Counter

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared modules (repo-root common/)
from proxy import CaptureProxy
from rollout import evaluate, run_rollout
from sandbox import Sandbox
from tasks import make_test

p = argparse.ArgumentParser()
p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
p.add_argument("--harnesses", default="shell,pytools,react")
p.add_argument("--n", type=int, default=10, help="tasks per bucket")
p.add_argument("--out", default="")
p.add_argument("--show", type=int, default=0, help="print raw completions of this many rollouts per harness")
args = p.parse_args()


async def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    proxy = CaptureProxy(tok, model_name=args.model, max_tokens=4096, max_ctx=16384, port=8101)
    await proxy.start()
    sbx = Sandbox(name="sbx-probe")
    tasks = make_test(args.n)
    if args.show:
        for h in args.harnesses.split(","):
            for t in tasks[:args.show]:
                r = await run_rollout(proxy, sbx, t, h)
                print(f"===== {h} {t.template} gold={t.answer} -> {r['answer']!r} {r['status']} correct={r['correct']}")
                for c in r["calls"][:4]:
                    print("--- completion:", tok.decode(c["completion_ids"])[-1500:])
    else:
        res, results = await evaluate(proxy, sbx, tasks, args.harnesses.split(","))
        print(json.dumps(res, indent=1))
        if args.out:
            with open(args.out, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
        for h in args.harnesses.split(","):
            print(h, Counter(r["status"] for r in results if r["harness"] == h))
    await proxy.stop()
    sbx.close()


asyncio.run(main())
