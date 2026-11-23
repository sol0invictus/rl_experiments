"""Three small agent harnesses. Each one owns its loop and only talks to an OpenAI-compatible URL.

They differ along the axes the article calls out (action format, context structure, control flow):

  shell     Codex-like. Native tool calling with one `shell` tool and a `submit` tool. A reply with no
            tool call ends the episode, and that reply is taken as the answer.
  pytools   OpenCode/Claude-Code-like. Four tools with their own names (list_files, read_file,
            run_python, final_answer). It compacts its history: old tool outputs are elided, so its
            prompts do not extend the previous call's token for token. Nudges once if no tool is called.
  react     Mini-SWE-Agent-like. No tool-calling API at all: the model writes one ```bash block per turn
            and finishes with a line `FINAL ANSWER: ...`. Format errors get a corrective message.

Every harness returns {"answer", "n_tool_calls", "status"}. The trainer never sees inside them.
"""
import sys
from pathlib import Path
import json
import re

from openai import AsyncOpenAI, BadRequestError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # shared proxy.py
from proxy import _parse_pythonic

MAX_TURNS = 10


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def _user_prompt(task):
    files = ", ".join(task.files)
    return (f"The data files {files} are in the current working directory.\n\n"
            f"Question: {task.question}\n\nInspect the files, compute the answer, and report just the final value.")


async def _chat(client, **kw):
    try:
        return (await client.chat.completions.create(model="policy", **kw)).choices[0]
    except BadRequestError:   # context window exceeded
        return None


# ------------------------------------------------------------------------------------------ shell
SHELL_SYS = ("You are a coding agent working in a Linux sandbox with Python 3 and pandas installed. "
             "Use the shell tool to run commands. When you know the answer, call submit with it.")
SHELL_TOOLS = [
    _fn("shell", "Run a bash command in the working directory and return its output.",
        {"command": {"type": "string", "description": "The bash command to run."}}, ["command"]),
    _fn("submit", "Submit the final answer to the question.",
        {"answer": {"type": "string", "description": "The final answer value."}}, ["answer"]),
]


async def shell_harness(task, ws, base_url):
    client = AsyncOpenAI(base_url=base_url, api_key="x", max_retries=0, timeout=600)
    msgs = [{"role": "system", "content": SHELL_SYS}, {"role": "user", "content": _user_prompt(task)}]
    n = 0
    for _ in range(MAX_TURNS):
        ch = await _chat(client, messages=msgs, tools=SHELL_TOOLS)
        if ch is None:
            return {"answer": None, "n_tool_calls": n, "status": "context"}
        m = ch.message
        if not m.tool_calls:
            return {"answer": m.content, "n_tool_calls": n, "status": "final_text"}
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "submit":
                return {"answer": str(args.get("answer", "")), "n_tool_calls": n, "status": "submitted"}
            n += 1
            if tc.function.name == "shell" and "command" in args:
                rc, out = await ws.bash(str(args["command"]))
                res = f"exit code {rc}\n{out}"
            else:
                res = f"Error: unknown tool or bad arguments: {tc.function.name}"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
    return {"answer": None, "n_tool_calls": n, "status": "max_turns"}


# ---------------------------------------------------------------------------------------- pytools
PY_SYS = ("You are a data analysis assistant. Explore the data with list_files and read_file, compute "
          "results with run_python (print what you need to see), and finish by calling final_answer.")
PY_TOOLS = [
    _fn("list_files", "List the files in the working directory.", {}, []),
    _fn("read_file", "Show the first lines of a text file.",
        {"path": {"type": "string"}, "max_lines": {"type": "integer", "description": "Default 10."}}, ["path"]),
    _fn("run_python", "Execute a Python script and return stdout/stderr. pandas is available.",
        {"code": {"type": "string"}}, ["code"]),
    _fn("final_answer", "Report the final answer and end the task.", {"answer": {"type": "string"}}, ["answer"]),
]
KEEP_RECENT = 2   # tool outputs older than the last KEEP_RECENT are replaced by a stub


def _compact(msgs):
    tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    out = [dict(m) for m in msgs]
    for i in tool_idx[:-KEEP_RECENT]:
        out[i]["content"] = "[earlier output elided to save context]"
    return out


async def pytools_harness(task, ws, base_url):
    client = AsyncOpenAI(base_url=base_url, api_key="x", max_retries=0, timeout=600)
    msgs = [{"role": "system", "content": PY_SYS}, {"role": "user", "content": _user_prompt(task)}]
    n, nudged = 0, False
    for _ in range(MAX_TURNS):
        ch = await _chat(client, messages=_compact(msgs), tools=PY_TOOLS)
        if ch is None:
            return {"answer": None, "n_tool_calls": n, "status": "context"}
        m = ch.message
        if not m.tool_calls:
            if nudged:
                return {"answer": m.content, "n_tool_calls": n, "status": "final_text"}
            nudged = True
            msgs += [{"role": "assistant", "content": m.content or ""},
                     {"role": "user", "content": "Use the tools to work on the task, and call final_answer "
                                                 "with the value when you are done."}]
            continue
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            args, name = json.loads(tc.function.arguments), tc.function.name
            if name == "final_answer":
                return {"answer": str(args.get("answer", "")), "n_tool_calls": n, "status": "submitted"}
            n += 1
            if name == "list_files":
                _, res = await ws.argv(["ls", "-la"])
            elif name == "read_file" and "path" in args:
                k = args.get("max_lines", 10)
                k = k if isinstance(k, int) and 0 < k <= 200 else 10
                _, res = await ws.argv(["head", "-n", str(k), str(args["path"])])
            elif name == "run_python" and "code" in args:
                with open(f"{ws.host}/_script.py", "w") as f:
                    f.write(str(args["code"]))
                rc, res = await ws.argv(["python3", "_script.py"])
                res = res or "(no output; use print() to see values)"
            else:
                res = f"Error: unknown tool or bad arguments: {name}"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
    return {"answer": None, "n_tool_calls": n, "status": "max_turns"}


# ------------------------------------------------------------------------------------------ react
REACT_SYS = """You are an agent that solves data questions in a Linux shell (Python 3 and pandas installed).

You have NO tool-calling functions. Tool calls are not executed. Act by writing plain text only.
Every response must contain exactly ONE bash code block with the command to run next:
```bash
<command>
```
You will see its output, then you can run another command. Run one command per response.
When you are certain of the answer, reply with a final line of the form
FINAL ANSWER: <value>
and no code block."""
# a worked example, as mini-SWE-agent-style harnesses ship one, so the text action format is concrete
REACT_DEMO = [
    {"role": "user", "content": "Example task. The data file scores.csv is in the current working directory.\n\n"
                                "Question: How many rows does scores.csv have?"},
    {"role": "assistant", "content": "I will count the data rows.\n```bash\npython3 -c \"import pandas as pd; "
                                     "print(len(pd.read_csv('scores.csv')))\"\n```"},
    {"role": "user", "content": "<returncode>0</returncode>\n<output>\n42\n</output>"},
    {"role": "assistant", "content": "FINAL ANSWER: 42"},
]
TEXT_CALL_RE = re.compile(r"\[\s*\w+\(.*\)\s*\]", re.S)
BASH_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.S)
FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.I)


def _text_call(text):
    """Fallback parser: LFM2.5 writes `[bash(command='ls')]` even with no tools offered (the proxy, like a
    real server, returns it as plain text). Accept a single bash/answer call written that way."""
    m = TEXT_CALL_RE.search(text)
    if not m:
        return None
    try:
        calls = _parse_pythonic(m.group(0))
    except (SyntaxError, ValueError, TypeError):
        return None
    if len(calls) != 1:
        return None
    name, args = calls[0]
    if "command" in args and name in ("bash", "shell", "sh", "run", "execute"):
        return "bash", str(args["command"])
    if "answer" in args:
        return "answer", str(args["answer"])
    return None


async def react_harness(task, ws, base_url):
    client = AsyncOpenAI(base_url=base_url, api_key="x", max_retries=0, timeout=600)
    msgs = [{"role": "system", "content": REACT_SYS}, *REACT_DEMO, {"role": "user", "content": _user_prompt(task)}]
    n = fb = 0   # fb: actions taken through the text-call fallback instead of the ```bash format
    for _ in range(MAX_TURNS):
        ch = await _chat(client, messages=msgs)
        if ch is None:
            return {"answer": None, "n_tool_calls": n, "status": "context", "format_fallbacks": fb}
        text = ch.message.content or ""
        msgs.append({"role": "assistant", "content": text})
        blocks = BASH_RE.findall(text)
        final = FINAL_RE.search(text)
        if final and not blocks:
            return {"answer": final.group(1).strip().splitlines()[0], "n_tool_calls": n, "status": "submitted",
                    "format_fallbacks": fb}
        if not blocks and not final and (tc := _text_call(text)):
            fb += 1
            if tc[0] == "answer":
                return {"answer": tc[1], "n_tool_calls": n, "status": "submitted", "format_fallbacks": fb}
            blocks = [tc[1]]
        if len(blocks) != 1:
            msgs.append({"role": "user", "content": "Format error: tool calls are not available here. Write exactly "
                                                    "one ```bash code block as plain text, or reply with "
                                                    "`FINAL ANSWER: <value>` and no code block."})
            continue
        n += 1
        rc, out = await ws.bash(blocks[0])
        msgs.append({"role": "user", "content": f"<returncode>{rc}</returncode>\n<output>\n{out}</output>"})
    return {"answer": None, "n_tool_calls": n, "status": "max_turns", "format_fallbacks": fb}


HARNESSES = {"shell": shell_harness, "pytools": pytools_harness, "react": react_harness}
