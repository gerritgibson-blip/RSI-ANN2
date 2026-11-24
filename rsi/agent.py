"""Agentic episodes: tool-calling coding agent, self-analysis, task proposal, meta-planning.

All episodes are run in lock-step batches through a `generate(prompts, sampling)`
callable, so vLLM can batch every active episode at each turn.
"""
import json, re, hashlib
from concurrent.futures import ThreadPoolExecutor
from .sandbox import run_code, grade
from .config import MAX_TURNS, MIX_FLOOR

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python", "description": "Execute a Python script in a sandbox and return stdout/stderr. Use for exploring, debugging and checking ideas.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run your candidate solution against the visible example tests. Hidden tests are stricter.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Submit the final, complete solution code (all needed imports and definitions). Ends the episode.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

AGENT_SYSTEM = """You are an autonomous software agent. Solve the task by planning, writing code, testing it with tools, and fixing errors.
Work in steps: think briefly, then make exactly ONE tool call per turn. Test edge cases yourself with run_python; hidden tests are stricter than the visible ones.
You have at most {turns} turns. Finish by calling submit with the complete solution."""

LESSON_BLOCK = "\n\nLessons from your own analysis of a previous failed attempt at this task:\n{lesson}"

ANALYSIS_PROMPT = """You are analysing one of your own failed agent runs in order to improve yourself.

TASK:
{task}

YOUR TRAJECTORY (condensed):
{traj}

RESULT OF HIDDEN TESTS ON YOUR SUBMISSION:
{result}

Diagnose precisely why the attempt failed (wrong algorithm? misread spec? edge case? bad testing process? ran out of turns?).
Then write a concrete, reusable lesson that would make the next attempt succeed.
Answer in exactly this format:
DIAGNOSIS: <what went wrong and why>
LESSON: <actionable instructions for the retry>"""

PROPOSE_PROMPT = """You are designing training tasks for yourself, to fix your own weaknesses as a coding agent.
Recent diagnoses of your failures:
{weak}

Propose ONE new self-contained Python programming task that exercises one of these weaknesses. It must be precisely specified, deterministic, and solvable in under 60 lines.
Return ONLY a JSON object with keys:
 "prompt": task statement including the exact function signature,
 "solution": a correct reference implementation,
 "tests": Python code with at least 5 `assert` statements (including edge cases) that call the function."""

PLAN_PROMPT = """You control your own training loop. Each cycle you: solve training tasks with tools, analyse your failures, retry with lessons, propose new tasks, then fine-tune on verified successes. A candidate is kept only if it beats the current model on held-out tasks graded by hidden tests; otherwise it is reverted.

History of previous cycles (plan -> outcome):
{history}

Current capacity: LoRA rank {rank}, targets {targets}. Allowed ranges: {bounds}
Fields: lr, epochs, n_train_tasks, attempts, temperature, repair_rounds, n_selfgen, mix (weights for solve/repair/analysis/selfgen/plan samples), expand (true to grow LoRA capacity - only useful if improvements have stalled).
Reason briefly about what the history implies, then output the plan for the next cycle as a JSON object inside <plan></plan> tags."""


def parse_tool_call(text):
    m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", text, re.S)
    raw = m.group(1) if m else None
    if raw is None:  # lenient: a bare JSON call object
        j = text.find('{"name"')
        if j < 0:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[j:])
            raw = json.dumps(obj)
        except Exception:
            return None
    try:
        obj = json.loads(raw)
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return obj.get("name"), args
    except Exception:
        return None


def _fmt_exec(r):
    s = ""
    if r["stdout"].strip():
        s += "STDOUT:\n" + r["stdout"][-1500:]
    if r["stderr"].strip():
        s += ("\n" if s else "") + "STDERR:\n" + r["stderr"][-1500:]
    return s or f"(no output, exit code {r['returncode']})"


def _exec_tool(task, name, args):
    code = (args or {}).get("code", "")
    if not isinstance(code, str) or not code.strip():
        return "ERROR: missing 'code' argument."
    if name == "run_python":
        return _fmt_exec(run_code(code, timeout=10))
    if name == "run_tests":
        if not task.get("visible_tests", "").strip():
            return "No visible tests for this task; write your own checks with run_python."
        r = grade(code, task["visible_tests"], timeout=10)
        return "VISIBLE TESTS PASSED." if r["passed"] else "VISIBLE TESTS FAILED.\n" + _fmt_exec(r)
    return f"ERROR: unknown tool {name}."


def run_episodes(generate, tasks, sampling, lessons=None, pool=None):
    """Run one agent episode per task. Returns list of dicts with messages, passed, result."""
    pool = pool or ThreadPoolExecutor(16)
    eps = []
    for i, t in enumerate(tasks):
        sys = AGENT_SYSTEM.format(turns=MAX_TURNS)
        if lessons and lessons[i]:
            sys += LESSON_BLOCK.format(lesson=lessons[i])
        eps.append({"task": t, "messages": [{"role": "system", "content": sys},
                                             {"role": "user", "content": t["prompt"]}],
                    "done": False, "submission": None, "turns": 0, "tool_calls": 0, "bad_calls": 0})
    for turn in range(MAX_TURNS):
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        outs = generate([e["messages"] for e in active], sampling)
        jobs = []
        for e, out in zip(active, outs):
            e["messages"].append({"role": "assistant", "content": out})
            e["turns"] += 1
            call = parse_tool_call(out)
            if call is None:
                e["bad_calls"] += 1
                e["messages"].append({"role": "user", "content": "No valid tool call found. Respond with exactly one <tool_call> using run_python, run_tests or submit."})
                continue
            name, args = call
            e["tool_calls"] += 1
            if name == "submit":
                e["submission"] = (args or {}).get("code", "")
                e["done"] = True
                continue
            jobs.append((e, pool.submit(_exec_tool, e["task"], name, args)))
        for e, fut in jobs:
            res = fut.result()
            if turn == MAX_TURNS - 2:
                res += "\n\n[Only one turn left: call submit now.]"
            e["messages"].append({"role": "tool", "content": res})
    # grade on HIDDEN tests
    futs = [pool.submit(grade, e["submission"] or "", e["task"]["hidden_tests"]) if e["submission"] else None for e in eps]
    for e, f in zip(eps, futs):
        if f is None:
            e["passed"], e["result"] = False, "No submission (ran out of turns or never called submit)."
        else:
            r = f.result()
            e["passed"] = r["passed"]
            e["result"] = "ALL HIDDEN TESTS PASSED" if r["passed"] else ("FAILED.\n" + _fmt_exec(r))
        # drop a trailing non-assistant message so training samples end on the model's action
        while e["messages"] and e["messages"][-1]["role"] != "assistant":
            e["messages"].pop()
    return eps


def condense(ep, limit=6000):
    parts = []
    for m in ep["messages"][2:]:
        tag = {"assistant": "YOU", "tool": "TOOL", "user": "ENV"}[m["role"]]
        parts.append(f"[{tag}] {m['content'][:1500]}")
    s = "\n".join(parts)
    return s if len(s) <= limit else s[:limit // 2] + "\n...\n" + s[-limit // 2:]


def analysis_messages(ep):
    return [{"role": "user", "content": ANALYSIS_PROMPT.format(task=ep["task"]["prompt"], traj=condense(ep), result=ep["result"][:2000])}]


PLAN_FORMAT_REMINDER = """

Your previous answer could not be used. End your answer with the plan as valid JSON inside the tags, for example:
<plan>{"lr": 3e-5, "epochs": 1, "n_train_tasks": 320, "attempts": 4, "temperature": 0.8, "repair_rounds": 2, "n_selfgen": 128, "mix": {"solve": 1.0, "repair": 1.5, "analysis": 1.0, "selfgen": 0.75, "plan": 0.3}, "expand": false}</plan>"""


def parse_lesson(text):
    m = re.search(r"LESSON:\s*(.+)", text, re.S)
    d = re.search(r"DIAGNOSIS:\s*(.+?)(?:LESSON:|$)", text, re.S)
    return (m.group(1).strip()[:1500] if m else None), (d.group(1).strip()[:600] if d else "")


def propose_messages(weak, exemplar=None):
    msg = PROPOSE_PROMPT.format(weak=weak)
    if exemplar:
        msg += ("\n\nCOMPLEXITY TARGET: your task must be at least as demanding as this reference task "
                "(multi-step reasoning, non-trivial algorithm, real edge cases):\n" + exemplar[:1200])
    return [{"role": "user", "content": msg}]


def parse_proposal(text, cycle, idx):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        p, s, t = obj["prompt"], obj["solution"], obj["tests"]
        if not all(isinstance(x, str) and x.strip() for x in (p, s, t)) or t.count("assert") < 5:
            return None
    except Exception:
        return None
    h = hashlib.md5(p.encode()).hexdigest()[:10]
    first = next((l for l in t.splitlines() if "assert" in l), "")
    return {"id": f"self-{cycle}-{idx}-{h}", "source": "selfgen", "prompt": p.strip() + (f"\n\nExample test:\n{first.strip()}" if first else ""),
            "visible_tests": first.strip(), "hidden_tests": t, "ref_solution": s}


def validate_proposal(task):
    """Reference must pass; an empty solution must fail (tests actually test something)."""
    if not grade(task["ref_solution"], task["hidden_tests"])["passed"]:
        return False
    if grade("", task["hidden_tests"])["passed"]:
        return False
    return True


def plan_messages(history, rank, targets, bounds):
    return [{"role": "user", "content": PLAN_PROMPT.format(history=history or "(none yet - this is the first cycle)", rank=rank, targets=targets, bounds=json.dumps(bounds))}]


def parse_plan(text, default, bounds):
    plan = json.loads(json.dumps(default))
    m = re.search(r"<plan>\s*(\{.*?\})\s*</plan>", text, re.S)
    if not m:
        return plan, False
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return plan, False
    for k, (lo, hi) in bounds.items():
        if k in obj:
            try:
                v = type(default[k])(obj[k])
                plan[k] = min(max(v, type(default[k])(lo)), type(default[k])(hi))
            except Exception:
                pass
    if isinstance(obj.get("mix"), dict):
        for k in plan["mix"]:
            try:
                plan["mix"][k] = float(min(max(float(obj["mix"].get(k, plan["mix"][k])), MIX_FLOOR.get(k, 0.0)), 3.0))
            except Exception:
                pass
    plan["expand"] = bool(obj.get("expand", False))
    return plan, True
