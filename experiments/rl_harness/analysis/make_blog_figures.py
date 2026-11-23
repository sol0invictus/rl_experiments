"""Figures for the multi-harness RL blog post, built only from a run's logs.

    .venv-lfm/bin/python analysis/make_blog_figures.py --run lfm_multi --out <dir>

Reads runs/<run>/{metrics,eval,eval_rollouts}.jsonl (and rebuilds the test set from tasks.make_test to
recover each test task's difficulty and mess). Writes PNGs to --out and analysis/results_<run>.json with
every number the post quotes, so the text and the figures come from the same place.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))  # shared modules (repo-root common/)
from tasks import make_test   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="lfm_multi")
p.add_argument("--out", default=os.path.join(ROOT, "analysis", "figures"))
p.add_argument("--n-test", type=int, default=40)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
RUN = os.path.join(ROOT, "runs", args.run)

# categorical slots 1-3 of the reference palette (validated all-pairs, light mode), fixed per harness
HARNESSES = ["shell", "pytools", "react"]
COLOR = {"shell": "#2a78d6", "pytools": "#eb6834", "react": "#1baf7a"}
INK, INK2, MUTED, GRID, BASE = "#1a1a1a", "#52514e", "#8a8984", "#e6e5e1", "#b9b8b2"
BUCKETS = [("seen", "Seen templates"), ("tmpl", "Held-out templates"), ("domain", "Held-out domain (web)")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "#fdfdfd",
    "axes.facecolor": "#fdfdfd", "savefig.facecolor": "#fdfdfd",
})


def jl(name):
    with open(os.path.join(RUN, name)) as f:
        return [json.loads(l) for l in f]


metrics = jl("metrics.jsonl")
timeouts = {e["step"]: sum(r["status"].startswith("error") for r in e["results"]) for e in jl("train_rollouts.jsonl")}
evals = {e["step"]: e for e in jl("eval.jsonl")}
ev_roll = {e["step"]: e["results"] for e in jl("eval_rollouts.jsonl")}
STEPS = sorted(evals)
S0, SF = STEPS[0], STEPS[-1]
TASKS = {(t.bucket, t.task_id): t for t in make_test(args.n_test)}
results = {"eval_steps": STEPS, "final_step": SF}


def save(fig, name):
    fig.savefig(os.path.join(args.out, name))
    plt.close(fig)
    print("wrote", name)


def pct_axis(ax, lo=0.0, hi=1.0):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))


def sign_test(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


# ------------------------------------------------------------- fig: held-out accuracy over training
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
panels = [("all", "All test tasks")] + BUCKETS
for ax, (b, title) in zip(axes, panels):
    key = "acc" if b == "all" else "acc_" + b
    ends = []
    for h in HARNESSES:
        ys = [evals[s][h][key] for s in STEPS]
        ax.plot(STEPS, ys, color=COLOR[h], lw=2, marker="o", ms=6, mec="#fdfdfd", mew=1.5, label=h)
        ends.append([ys[-1], ys[-1], h])
    ends.sort()
    for i in range(1, len(ends)):   # push end labels apart so they never overlap
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.045)
    for v, y, h in ends:
        ax.annotate(f"{v:.0%}", (STEPS[-1], v), xytext=(STEPS[-1] + 2.5, y), textcoords="data",
                    va="center", fontsize=9, color=INK2)
    ys = [evals[s]["all"][key] for s in STEPS]
    ax.plot(STEPS, ys, color=INK, lw=1.5, ls="--", label="mean of harnesses")
    ax.set_title(title)
    ax.set_xticks(STEPS)
    ax.set_xlabel("training step")
    ax.set_xlim(STEPS[0] - 3, STEPS[-1] + 12)
    pct_axis(ax, 0.3, 1.02)
axes[0].set_ylabel("held-out accuracy")
axes[0].legend(loc="lower right", fontsize=9)
save(fig, "fig_eval_curves.png")
results["eval"] = {s: {h: {k: evals[s][h][k] for k in ("acc", "acc_seen", "acc_tmpl", "acc_domain", "tool_calls")}
                       for h in HARNESSES + ["all"] if h in evals[s]} for s in STEPS}

# ------------------------------------------------------------- fig: training curves + stability
def rolling(xs, k=5):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


steps = [m["step"] for m in metrics]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
ax = axes[0]
for h in HARNESSES:
    ys = [m.get("acc_" + h, float("nan")) for m in metrics]
    ax.plot(steps, ys, color=COLOR[h], lw=0.8, alpha=0.3)
    ax.plot(steps, rolling(ys), color=COLOR[h], lw=2, label=h)
stall = [s for s, n in sorted(timeouts.items()) if n >= 5]
for s in stall:
    ax.axvline(s, color=MUTED, lw=1, ls=":")
    ax.annotate(f"stall ({timeouts[s]} timeouts)", (s, 1.0), xytext=(3, 0), textcoords="offset points",
                fontsize=8, color=MUTED, va="top")
ax.set_title("Batch accuracy (5-step mean)")
ax.set_xlabel("training step")
pct_axis(ax, 0.3, 1.02)
ax.legend(loc="lower right", fontsize=9)
ax = axes[1]
ax.plot(steps, [1e3 * m["mismatch_kl"] for m in metrics], color=INK2, lw=1.5)
ax.set_title("vLLM vs trainer KL per token (x1e-3)")
ax.set_xlabel("training step")
ax.set_ylim(0, 1e3 * max(m["mismatch_kl"] for m in metrics) * 1.4)
ax = axes[2]
ax.plot(steps, [m["flat_groups"] / 12 for m in metrics], color=INK2, lw=1.5)
ax.set_title("Groups with no reward variance")
ax.set_xlabel("training step")
pct_axis(ax, 0, 0.5)
save(fig, "fig_training.png")
results["training"] = {
    "steps": len(metrics), "stall_steps": {s: timeouts[s] for s in stall},
    "kl_mean": sum(m["mismatch_kl"] for m in metrics) / len(metrics),
    "kl_max": max(m["mismatch_kl"] for m in metrics),
    "flat_mean": sum(m["flat_groups"] for m in metrics) / len(metrics) / 12,
    "t_rollout_median": sorted(m["t_rollout"] for m in metrics)[len(metrics) // 2],
    "t_train_median": sorted(m["t_train"] for m in metrics)[len(metrics) // 2],
    "peak_mem_gb": max(m.get("peak_mem_gb", 0) for m in metrics),
    "trainable_tokens_median": sorted(m.get("train_tokens", 0) for m in metrics)[len(metrics) // 2],
}

# ------------------------------------------------------------- fig: baseline vs final, by task property
def acc_by(step, key):
    d = defaultdict(list)
    for r in ev_roll[step]:
        d[key(r, TASKS[(r["bucket"], r["task_id"])])].append(r["correct"])
    return {k: (sum(v) / len(v), len(v)) for k, v in d.items()}


groups = [
    ("Difficulty", lambda r, t: t.difficulty, ["easy", "medium", "hard"]),
    ("Messy files", lambda r, t: {0: "clean", 1: "1 kind of mess", 2: "2 kinds of mess"}[len(t.mess)],
     ["clean", "1 kind of mess", "2 kinds of mess"]),
    ("Tasks containing each mess kind", None, ["dup", "money", "case", "dates", "split"]),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.0))
results["breakdown"] = {}
for ax, (title, key, cats) in zip(axes, groups):
    if key is None:
        a0 = {k: acc_by(S0, lambda r, t, k=k: k in t.mess).get(True) for k in cats}
        a1 = {k: acc_by(SF, lambda r, t, k=k: k in t.mess).get(True) for k in cats}
    else:
        a0, a1 = acc_by(S0, key), acc_by(SF, key)
    ys = list(range(len(cats)))[::-1]
    for y, c in zip(ys, cats):
        (v0, n), (v1, _) = a0[c], a1[c]
        ax.plot([v0, v1], [y, y], color=BASE, lw=2, zorder=1)
        ax.scatter([v0], [y], s=60, color=BASE, zorder=2, edgecolor="#fdfdfd", lw=1.5)
        ax.scatter([v1], [y], s=60, color=COLOR["shell"], zorder=3, edgecolor="#fdfdfd", lw=1.5)
        ax.annotate(f"{v1:.0%}", (v1, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
        results["breakdown"][f"{title}/{c}"] = {"base": v0, "final": v1, "n": n}
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}\n(n={a0[c][1]})" for c in cats], fontsize=9.5)
    ax.set_title(title)
    ax.set_xlim(0.3, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
axes[0].scatter([], [], s=60, color=BASE, label=f"base (step {S0})")
axes[0].scatter([], [], s=60, color=COLOR["shell"], label=f"trained (step {SF})")
axes[0].legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, -0.12), ncol=2)
fig.subplots_adjust(wspace=0.45)
save(fig, "fig_breakdown.png")

# ------------------------------------------------------------- fig: pytools episode outcomes + tool calls
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
OUT = [("submitted", "called final_answer", COLOR["pytools"]), ("final_text", "replied in text", "#f3ae8f"),
       ("max_turns", "hit the 10-turn limit", BASE)]
for i, s in enumerate(STEPS):
    st = evals[s]["pytools"]["statuses"]
    tot, left = sum(st.values()), 0
    for k, lab, col in OUT:
        w = st.get(k, 0) / tot
        ax.barh(i, w, left=left, color=col, edgecolor="#fdfdfd", lw=2, height=0.62, label=lab if i == 0 else None)
        if w > 0.06:
            ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=9, color=INK)
        left += w
ax.set_yticks(range(len(STEPS)))
ax.set_yticklabels([f"step {s}" for s in STEPS])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(False)
ax.set_title("How pytools episodes end")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
ax = axes[1]
for h in HARNESSES:
    ys = [evals[s][h]["tool_calls"] for s in STEPS]
    ax.plot(STEPS, ys, color=COLOR[h], lw=2, marker="o", ms=6, mec="#fdfdfd", mew=1.5, label=h)
ax.set_title("Tool calls per test episode")
ax.set_xticks(STEPS)
ax.set_xlabel("training step")
ax.set_ylim(0, None)
ax.legend(fontsize=9)
save(fig, "fig_efficiency.png")
results["pytools_status"] = {s: evals[s]["pytools"]["statuses"] for s in STEPS}

# ------------------------------------------------------------- numbers for tables
paired = {}
R = {s: {(r["harness"], r["bucket"], r["task_id"]): r["correct"] for r in ev_roll[s]} for s in (S0, SF)}
for h in HARNESSES + ["all"]:
    keys = [k for k in R[S0] if h == "all" or k[0] == h]
    w = sum(R[SF][k] and not R[S0][k] for k in keys)
    l = sum(R[S0][k] and not R[SF][k] for k in keys)
    paired[h] = {"fixed": w, "broke": l, "p": sign_test(w, l), "n": len(keys)}
results["paired"] = paired
results["react_fallback"] = {s: evals[s]["react"].get("fallback_rate") for s in STEPS}
results["harness_x_difficulty"] = {
    s: {k: v for k, v in acc_by(s, lambda r, t: f"{r['harness']}/{t.difficulty}").items()} for s in (S0, SF)}
tmpl = {s: acc_by(s, lambda r, t: t.template) for s in (S0, SF)}
results["templates"] = {k: {"base": tmpl[S0][k][0], "final": tmpl[SF][k][0], "n": tmpl[S0][k][1]} for k in tmpl[S0]}
with open(os.path.join(ROOT, "analysis", f"results_{args.run}.json"), "w") as f:   # not in --out: Astro rejects data files there
    json.dump(results, f, indent=1, default=str)
print(f"wrote analysis/results_{args.run}.json")
