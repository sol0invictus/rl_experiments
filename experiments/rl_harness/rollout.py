"""Run harnesses on tasks through the capture proxy, grade them, and turn captures into training rows."""
import asyncio
import uuid
from collections import defaultdict

from harnesses import HARNESSES, MAX_TURNS
from tasks import grade

EFF_BONUS = 0.1


def reward_fn(correct, n_calls):
    """1 for correct, 0 otherwise, plus up to 0.1 for solving it in fewer tool calls (correct only)."""
    if not correct:
        return 0.0
    return 1.0 + EFF_BONUS * min(1.0, max(0.0, (MAX_TURNS - n_calls) / (MAX_TURNS - 2)))


async def run_rollout(proxy, sbx, task, harness):
    rid = uuid.uuid4().hex
    ws = sbx.new_workspace(task.files)
    try:
        res = await HARNESSES[harness](task, ws, proxy.base_url(rid))
    except Exception as e:   # a crashing harness is a failed rollout, not a crashed run
        res = {"answer": None, "n_tool_calls": 0, "status": f"error:{type(e).__name__}"}
    finally:
        ws.cleanup()
    correct = grade(res["answer"], task)
    return {"task_id": task.task_id, "harness": harness, "difficulty": task.difficulty, "bucket": task.bucket,
            "template": task.template, **res,
            "correct": bool(correct), "reward": reward_fn(correct, res["n_tool_calls"]),
            "calls": proxy.pop(rid)}


def to_rows(calls):
    """Merge consecutive calls whose prompt extends the previous prompt+completion token for token.

    Each row is (ids, loss_mask, old_logprobs). Harnesses that rewrite history (pytools) break the
    extension and yield one row per call; append-only harnesses usually collapse into one row.
    """
    rows = []
    for c in calls:
        p, comp, lp = c["prompt_ids"], c["completion_ids"], c["logprobs"]
        if rows:
            ids = rows[-1]["ids"]
            if len(p) >= len(ids) and p[:len(ids)] == ids:
                ext = p[len(ids):]
                rows[-1]["ids"] += ext + comp
                rows[-1]["mask"] += [0] * len(ext) + [1] * len(comp)
                rows[-1]["old_lp"] += [0.0] * len(ext) + lp
                continue
        rows.append({"ids": list(p) + comp, "mask": [0] * len(p) + [1] * len(comp),
                     "old_lp": [0.0] * len(p) + lp})
    return rows


async def evaluate(proxy, sbx, tasks, harnesses, temperature=0.7):
    old_t = proxy.temperature
    proxy.temperature = temperature
    jobs = [run_rollout(proxy, sbx, t, h) for t in tasks for h in harnesses]
    results = await asyncio.gather(*jobs)
    proxy.temperature = old_t
    out = {}
    by_h = defaultdict(list)
    for r in results:
        by_h[r["harness"]].append(r)
    for h, rs in by_h.items():
        solved = [r for r in rs if r["correct"]]
        out[h] = {"acc": sum(r["correct"] for r in rs) / len(rs),
                  "tool_calls": sum(r["n_tool_calls"] for r in rs) / len(rs),
                  "tool_calls_solved": sum(r["n_tool_calls"] for r in solved) / max(1, len(solved)),
                  "submitted": sum(r["status"] == "submitted" for r in rs) / len(rs),
                  "statuses": dict(_count(r["status"] for r in rs))}
        if any("format_fallbacks" in r for r in rs):   # react: share of actions not in the ```bash format
            out[h]["fallback_rate"] = (sum(r.get("format_fallbacks", 0) for r in rs)
                                       / max(1, sum(r["n_tool_calls"] + (r["status"] == "submitted") for r in rs)))
        for b in sorted({r["bucket"] for r in rs}):
            out[h]["acc_" + b] = _mean(r["correct"] for r in rs if r["bucket"] == b)
    out["all"] = {"acc": sum(r["correct"] for r in results) / len(results),
                  "tool_calls": sum(r["n_tool_calls"] for r in results) / len(results),
                  **{"acc_" + b: _mean(r["correct"] for r in results if r["bucket"] == b)
                     for b in sorted({r["bucket"] for r in results})}}
    for r in results:
        r.pop("calls")
    return out, results


def _mean(it):
    it = list(it)
    return sum(it) / max(1, len(it))


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d
