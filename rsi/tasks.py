"""Verifiable task pools.

Every task has HIDDEN tests that decide success. The agent only sees the task
statement plus (at most) a small subset of tests. Splits:

  train  : KodCode-V1 sample + MBPP train/val/prompt + model-proposed tasks
  gate   : fixed held-out set used ONLY to accept/revert candidate models
  final  : HumanEval + rest of MBPP test, never used for any decision; only
           reported, to detect the loop overfitting to the gate.
"""
import json, os, random, re
from .config import STATE, SMOKE

PYTEST_SHIM = r'''
import contextlib as _cl, math as _m
class _Raises:
    def __init__(self, exc, match=None): self.exc, self.match = exc, match
    def __enter__(self): return self
    def __exit__(self, et, ev, tb):
        if et is None: raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc): return False
        if self.match is not None:
            import re as _re
            if not _re.search(self.match, str(ev)): raise AssertionError("match failed")
        self.value = ev
        return True
class pytest:
    @staticmethod
    def raises(exc, match=None): return _Raises(exc, match)
    @staticmethod
    def approx(v, rel=1e-6, abs=1e-12):
        class _A:
            def __eq__(s, o): return _m.isclose(o, v, rel_tol=rel, abs_tol=abs)
        return _A()
'''

RUN_TESTS = r'''
_tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
assert _tests, "no tests"
for _t in _tests:
    _t()
'''


def _kodcode_task(row, i):
    test = row.get("test") or ""
    if not test or "fixture" in test or "monkeypatch" in test or "capsys" in test or "tmp_path" in test:
        return None
    # only zero-arg test functions; only pytest.raises / approx
    if re.search(r"def test_\w+\(\s*\w", test):
        return None
    if re.search(r"pytest\.(?!raises|approx)\w+", test):
        return None
    lines = [l for l in test.splitlines()
             if not re.match(r"\s*(from\s+solution\s+import|import\s+solution|import\s+pytest|from\s+pytest)", l)]
    test = "\n".join(lines)
    hidden = PYTEST_SHIM + "\n" + test + "\n" + RUN_TESTS
    # visible: first test function only
    m = re.search(r"(def test_\w+\(\):.*?)(?=\ndef |\Z)", test, re.S)
    visible = (PYTEST_SHIM + "\n" + m.group(1) + "\n" + RUN_TESTS) if m else ""
    ti = row.get("test_info") or []
    sig = ""
    if ti and isinstance(ti, list) and isinstance(ti[0], dict):
        sig = ti[0].get("function_declaration") or ""
    prompt = row["question"].strip()
    if sig:
        prompt += f"\n\nRequired signature: `{sig.strip()}`"
    return {"id": f"kod-{i}", "source": "kodcode", "prompt": prompt,
            "visible_tests": visible, "hidden_tests": hidden,
            "difficulty": row.get("gpt_difficulty", "")}


def _mbpp_task(row):
    tests = row["test_list"]
    setup = row.get("test_setup_code") or ""
    hidden = setup + "\n" + "\n".join(tests) + "\n" + "\n".join(row.get("challenge_test_list") or [])
    return {"id": f"mbpp-{row['task_id']}", "source": "mbpp",
            "prompt": row["text"].strip() + f"\n\nYour code must satisfy tests like:\n{tests[0]}",
            "visible_tests": setup + "\n" + tests[0], "hidden_tests": hidden}


def _humaneval_task(row):
    return {"id": f"he-{row['task_id']}", "source": "humaneval",
            "prompt": "Complete this Python function (return the full function definition):\n\n" + row["prompt"],
            "visible_tests": "", "hidden_tests": row["test"] + f"\n\ncheck({row['entry_point']})\n"}


def build_pools(seed=0, n_kod_train=6000, n_kod_gate=120):
    from datasets import load_dataset
    rng = random.Random(seed)
    pools = {"train": [], "gate": [], "final": []}

    mb = load_dataset("google-research-datasets/mbpp", "full")
    for split in ("train", "validation", "prompt"):
        pools["train"] += [_mbpp_task(r) for r in mb[split]]
    test = [_mbpp_task(r) for r in mb["test"]]
    rng.shuffle(test)
    pools["gate"] += test[:200]
    pools["final"] += test[200:]

    he = load_dataset("openai/openai_humaneval")["test"]
    pools["final"] += [_humaneval_task(r) for r in he]

    try:
        kod = load_dataset("KodCode/KodCode-V1", split="train", streaming=True)
        kod_tasks = []
        for i, row in enumerate(kod):
            t = _kodcode_task(row, i)
            if t:
                kod_tasks.append(t)
            if len(kod_tasks) >= (n_kod_train + n_kod_gate) * 2:
                break
        rng.shuffle(kod_tasks)
        pools["gate"] += kod_tasks[:n_kod_gate]
        pools["train"] += kod_tasks[n_kod_gate:n_kod_gate + n_kod_train]
    except Exception as e:  # dataset unavailable: MBPP-only training pool
        print("KodCode unavailable:", e)
    if SMOKE:
        pools["gate"], pools["final"] = pools["gate"][:12] + pools["gate"][-8:], pools["final"][:6] + pools["final"][-6:]
    return pools


def load_pools():
    path = os.path.join(STATE, "pools.json")
    if not os.path.exists(path):
        pools = build_pools()
        with open(path, "w") as f:
            json.dump(pools, f)
    with open(path) as f:
        pools = json.load(f)
    sg = os.path.join(STATE, "selfgen_tasks.jsonl")
    if os.path.exists(sg):
        with open(sg) as f:
            pools["train"] += [json.loads(l) for l in f if l.strip()]
    if SMOKE:
        pools["gate"], pools["final"] = pools["gate"][:12] + pools["gate"][-8:], pools["final"][:6] + pools["final"][-6:]
        pools["retain"] = pools.get("retain", [])[:6]
    return pools
