"""GPU rollout phase (vLLM). Run as: python -m rsi.rollout <job.json>

modes:
  collect : meta-plan -> solve train tasks -> self-analysis + repair -> propose tasks
            -> write verified training samples
  eval    : score an adapter on gate (and optionally final) tasks.
            per-task score = 1.0 solved first try, 0.5 solved after its own
            self-analysis + retry, 0.0 otherwise  (all judged by hidden tests)
"""
import json, os, random, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from .config import BASE_MODEL, MAX_NEW_TOKENS, MAX_SEQ_LEN, BOUNDS, STATE, ENGINE, SELFGEN_MIN_FRONTIER, N_RECOMMENDED
from . import agent as A
from .tasks import load_pools


class Engine:
    def __init__(self, adapter, max_rank, engine=None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.SP = SamplingParams
        self.llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=max_rank, max_loras=1,
                       max_model_len=MAX_SEQ_LEN, gpu_memory_utilization=0.90,
                       enable_prefix_caching=True, dtype="bfloat16", seed=int(time.time()) % 100000,
                       max_num_seqs=384, **(ENGINE if engine is None else engine))
        self.tok = self.llm.get_tokenizer()
        self.LR = LoRARequest
        self.lora = LoRARequest("champ", 1, adapter) if adapter else None
        self._lid = 1
        self.gen_tokens = 0

    def set_adapter(self, adapter):
        self._lid += 1
        self.lora = self.LR(f"a{self._lid}", self._lid, adapter) if adapter else None

    def generate(self, convs, temperature, tools=True, max_new=MAX_NEW_TOKENS, n=1):
        prompts, sps, keep = [], [], []
        for c in convs:
            p = self.tok.apply_chat_template(c, tools=A.TOOLS if tools else None, add_generation_prompt=True, tokenize=False)
            ids = self.tok(p, add_special_tokens=False).input_ids
            room = MAX_SEQ_LEN - len(ids) - 8
            if room < 64:
                keep.append(False); continue
            keep.append(True)
            prompts.append({"prompt_token_ids": ids})
            sps.append(self.SP(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                               max_tokens=min(max_new, room), n=n,
                               stop=["</tool_call>"] if tools else None, include_stop_str_in_output=True,
                               skip_special_tokens=False))
        outs = self.llm.generate(prompts, sps, lora_request=self.lora, use_tqdm=False) if prompts else []
        res, it = [], iter(outs)
        for k in keep:
            if not k:
                res.append([""] * n if n > 1 else ""); continue
            o = next(it)
            self.gen_tokens += sum(len(x.token_ids) for x in o.outputs)
            texts = [x.text.replace("<|im_end|>", "").replace("<|endoftext|>", "").rstrip() for x in o.outputs]
            res.append(texts if n > 1 else texts[0])
        return res


def agent_gen(engine):
    return lambda convs, temp: engine.generate(convs, temp, tools=True)


def analyse_and_repair(engine, failed_eps, temperature, rounds, n_analyses=1):
    """Self-analysis of failures followed by lesson-conditioned retries.
    Returns list of (failed_ep, analysis_msgs, analysis_text, repair_ep) for every attempt."""
    records = []
    current = failed_eps
    for _ in range(max(rounds, 0)):
        if not current:
            break
        amsgs = [A.analysis_messages(e) for e in current]
        texts = engine.generate(amsgs, temperature, tools=False, max_new=700, n=n_analyses)
        if n_analyses == 1:
            texts = [[t] for t in texts]
        retry_tasks, lessons, meta = [], [], []
        for e, am, ts in zip(current, amsgs, texts):
            for t in ts:
                lesson, diag = A.parse_lesson(t)
                if lesson:
                    retry_tasks.append(e["task"]); lessons.append(lesson); meta.append((e, am, t, diag))
        reps = A.run_episodes(agent_gen(engine), retry_tasks, temperature, lessons=lessons) if retry_tasks else []
        nxt = []
        solved_ids = set()
        for (e, am, t, diag), r in zip(meta, reps):
            records.append({"failed": e, "analysis_msgs": am, "analysis": t, "diagnosis": diag, "repair": r})
            if r["passed"]:
                solved_ids.add(e["task"]["id"])
        for (e, am, t, diag), r in zip(meta, reps):
            if e["task"]["id"] not in solved_ids and r["task"]["id"] not in {x["task"]["id"] for x in nxt}:
                nxt.append(r)
        current = nxt
    return records


def add_pairs(samples, tid, wins, fails, k=2):
    """Preference pairs from the SAME task: a hidden-test pass (chosen) vs a fail (rejected)."""
    wins = sorted(wins, key=lambda e: e["turns"])
    for w, f in list(zip(wins, fails))[:k]:
        if len(f["messages"]) > 2 and len(w["messages"]) > 2:
            samples["pair"].append({"kind": "pair", "task_id": tid, "chosen": w["messages"], "rejected": f["messages"]})


def strip_lesson(messages):
    m = [dict(x) for x in messages]
    m[0]["content"] = m[0]["content"].split("\n\nLessons from your own analysis")[0]
    return m


def mode_eval(engine, job):
    if job.get("adapters"):
        res = {}
        for name, path in job["adapters"].items():
            engine.set_adapter(path)
            print("=== evaluating candidate", name, flush=True)
            res[name] = _eval_one(engine, job)
        return {"candidates": res}
    return _eval_one(engine, job)


def _eval_one(engine, job):
    pools = load_pools()
    out = {}
    for split in job["splits"]:
        tasks = pools[split]
        eps = A.run_episodes(agent_gen(engine), tasks, 0.0)
        scores = {e["task"]["id"]: (1.0 if e["passed"] else 0.0) for e in eps}
        failed = [e for e in eps if not e["passed"]]
        recs = analyse_and_repair(engine, failed, 0.0, rounds=1)
        for r in recs:
            if r["repair"]["passed"]:
                scores[r["failed"]["task"]["id"]] = 0.5
        out[split] = {"scores": scores,
                      "pass1": sum(e["passed"] for e in eps) / len(eps),
                      "repair_rate": (sum(r["repair"]["passed"] for r in recs) / max(len(recs), 1)),
                      "score": sum(scores.values()) / len(scores),
                      "by_source": {s: sum(scores[t["id"]] for t in tasks if t["source"] == s) / max(1, sum(t["source"] == s for t in tasks))
                                    for s in {t["source"] for t in tasks}}}
        print(split, {k: v for k, v in out[split].items() if k != "scores"}, flush=True)
    return out


def mode_collect(engine, job):
    rng = random.Random(job["cycle"] * 7919 + int(time.time()))
    pools = load_pools()
    stats_path = os.path.join(STATE, "task_stats.json")
    tstats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    samples = defaultdict(list)
    metrics = {}

    # 1. meta-plan: the model plans its own next training cycle
    pm = A.plan_messages(job["history"], job["rank"], job["targets"], BOUNDS)
    ptxt = engine.generate([pm], 0.7, tools=False, max_new=900)[0]
    plan, ok = A.parse_plan(ptxt, job["default_plan"], BOUNDS)
    if not ok:   # one retry with an explicit format reminder
        pm = pm[:-1] + [{"role": "user", "content": pm[-1]["content"] + A.PLAN_FORMAT_REMINDER}]
        ptxt = engine.generate([pm], 0.7, tools=False, max_new=900)[0]
        plan, ok = A.parse_plan(ptxt, job["default_plan"], BOUNDS)
    metrics["plan_parsed"] = ok
    plan_sample = {"messages": pm + [{"role": "assistant", "content": ptxt}], "kind": "plan"} if ok else None
    print("PLAN", json.dumps(plan), flush=True)

    # 2. curriculum: prefer tasks at the frontier (unseen or sometimes-solved)
    def weight(t):
        s = tstats.get(t["id"])
        if not s:
            return 1.0
        rate = s["solved"] / max(s["tries"], 1)
        return 0.15 if rate >= 0.99 else (1.5 if 0 < rate < 0.99 else 0.6)
    train = pools["train"]
    chosen = rng.choices(train, weights=[weight(t) for t in train], k=min(plan["n_train_tasks"] * 2, len(train)))
    seen, picked = set(), []
    for t in chosen:
        if t["id"] not in seen:
            seen.add(t["id"]); picked.append(t)
        if len(picked) >= plan["n_train_tasks"]:
            break

    k = plan["attempts"]
    eps = A.run_episodes(agent_gen(engine), [t for t in picked for _ in range(k)], plan["temperature"])
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task"]["id"]].append(e)
    for tid, es in by_task.items():
        s = tstats.setdefault(tid, {"tries": 0, "solved": 0})
        s["tries"] += len(es); s["solved"] += sum(e["passed"] for e in es)
        wins = sorted([e for e in es if e["passed"]], key=lambda e: (e["turns"], len(str(e["messages"]))))
        # HARD-WON ONLY: a task solved in every attempt teaches nothing new -> no sample
        if 0 < len(wins) < len(es):
            for w in wins[:2]:
                samples["solve"].append({"messages": w["messages"], "kind": "solve", "hard": True, "task_id": tid})
            add_pairs(samples, tid, wins, [e for e in es if not e["passed"]])
    metrics["train_pass_rate"] = sum(e["passed"] for e in eps) / max(len(eps), 1)
    metrics["train_tasks_any_solved"] = sum(any(e["passed"] for e in es) for es in by_task.values()) / max(len(by_task), 1)

    # 3. self-analysis + repair on tasks never solved
    failed = [es[0] for es in by_task.values() if not any(e["passed"] for e in es)]
    recs = analyse_and_repair(engine, failed, plan["temperature"], plan["repair_rounds"], n_analyses=2)
    diagnoses = []
    for r in recs:
        if r["diagnosis"]:
            diagnoses.append(r["diagnosis"])
        if r["repair"]["passed"]:
            samples["repair"].append({"messages": strip_lesson(r["repair"]["messages"]), "kind": "repair"})
            samples["pair"].append({"kind": "pair", "task_id": r["failed"]["task"]["id"], "chosen": strip_lesson(r["repair"]["messages"]), "rejected": r["failed"]["messages"]})
            samples["analysis"].append({"messages": r["analysis_msgs"] + [{"role": "assistant", "content": r["analysis"]}], "kind": "analysis"})
    metrics["repair_attempts"] = len(recs)
    metrics["repair_success_rate"] = sum(r["repair"]["passed"] for r in recs) / max(len(recs), 1)
    metrics["unsolved_tasks"] = len(failed)

    # 4. propose new tasks targeting own weaknesses
    n = plan["n_selfgen"]
    if not diagnoses:   # nothing failed: aim proposals at harder ground than the tasks just solved
        diagnoses = ["Tasks were solved; the next tasks should be harder: tricky edge cases, efficiency constraints, multi-step logic. Example solved task: " + t["prompt"][:250] for t in picked[:20]]
    if n:
        weak_lists = ["\n".join("- " + d[:300] for d in rng.sample(diagnoses, min(5, len(diagnoses)))) for _ in range(n)]
        hard_pool = [t for t in pools["train"] if t.get("difficulty") == "hard"]
        pmsgs = [A.propose_messages(w, rng.choice(hard_pool)["prompt"] if hard_pool else None) for w in weak_lists]
        ptexts = engine.generate(pmsgs, 0.9, tools=False, max_new=1200)
        props = [(m, t, A.parse_proposal(t, job["cycle"], i)) for i, (m, t) in enumerate(zip(pmsgs, ptexts))]
        props = [p for p in props if p[2]]
        with ThreadPoolExecutor(16) as ex:
            valid = list(ex.map(lambda p: A.validate_proposal(p[2]), props))
        props = [p for p, v in zip(props, valid) if v]
        known = {t["prompt"] for t in pools["train"]}
        props = [p for p in props if p[2]["prompt"] not in known]
        seps = A.run_episodes(agent_gen(engine), [p[2] for p in props for _ in range(4)], plan["temperature"])
        good = 0
        with open(os.path.join(STATE, "selfgen_tasks.jsonl"), "a") as f:
            for j, (m, t, task) in enumerate(props):
                es = seps[j * 4:(j + 1) * 4]
                solved = sum(e["passed"] for e in es)
                f.write(json.dumps(task) + "\n")
                if 0 < solved < 4:   # valid AND discriminative: genuinely at the frontier
                    good += 1
                    samples["selfgen"].append({"messages": m + [{"role": "assistant", "content": t}], "kind": "selfgen"})
                if 0 < solved < 4:
                    for e in [e for e in es if e["passed"]][:1]:
                        samples["solve"].append({"messages": e["messages"], "kind": "solve", "hard": True, "task_id": task["id"]})
        metrics["selfgen_proposed"] = n
        metrics["selfgen_valid"] = len(props)
        metrics["selfgen_frontier"] = good

    # 5. RECOMMENDED COMPLEX TASKS: if the model failed to produce enough useful
    #    frontier tasks itself, or failed to produce a usable plan, hand it unseen
    #    hard tasks to attempt (only hard-won outcomes become training data).
    frontier = metrics.get("selfgen_frontier", 0)
    if frontier < SELFGEN_MIN_FRONTIER or not ok:
        unseen_hard = [t for t in pools["train"] if t.get("difficulty") == "hard" and t["id"] not in tstats]
        if len(unseen_hard) < N_RECOMMENDED:   # fall back to hard tasks it has not yet mastered
            unseen_hard += [t for t in pools["train"] if t.get("difficulty") == "hard" and t["id"] in tstats
                            and tstats[t["id"]]["solved"] < tstats[t["id"]]["tries"]]
        rec = rng.sample(unseen_hard, min(N_RECOMMENDED, len(unseen_hard)))
        reps = A.run_episodes(agent_gen(engine), [t for t in rec for _ in range(k)], plan["temperature"])
        rb = defaultdict(list)
        for e in reps:
            rb[e["task"]["id"]].append(e)
        hw = 0
        for tid, es in rb.items():
            s = tstats.setdefault(tid, {"tries": 0, "solved": 0})
            s["tries"] += len(es); s["solved"] += sum(e["passed"] for e in es)
            wins = [e for e in es if e["passed"]]
            if 0 < len(wins) < len(es):
                hw += 1
                for w in sorted(wins, key=lambda e: e["turns"])[:2]:
                    samples["solve"].append({"messages": w["messages"], "kind": "solve", "hard": True, "task_id": tid, "recommended": True})
                add_pairs(samples, tid, wins, [e for e in es if not e["passed"]])
        # never-solved recommended tasks get the same self-analysis + repair treatment
        rfailed = [es[0] for es in rb.values() if not any(e["passed"] for e in es)]
        for r in analyse_and_repair(engine, rfailed, plan["temperature"], 1, n_analyses=2):
            if r["repair"]["passed"]:
                samples["repair"].append({"messages": strip_lesson(r["repair"]["messages"]), "kind": "repair"})
                samples["pair"].append({"kind": "pair", "task_id": r["failed"]["task"]["id"], "chosen": strip_lesson(r["repair"]["messages"]), "rejected": r["failed"]["messages"]})
                samples["analysis"].append({"messages": r["analysis_msgs"] + [{"role": "assistant", "content": r["analysis"]}], "kind": "analysis"})
        metrics["recommended_reason"] = "low self-generated frontier" if frontier < SELFGEN_MIN_FRONTIER else "plan unusable"
        metrics["recommended_tasks"] = len(rec)
        metrics["recommended_hard_won"] = hw
        metrics["recommended_pass_rate"] = sum(e["passed"] for e in reps) / max(len(reps), 1)
    metrics["plan_parsed"] = ok

    json.dump(tstats, open(stats_path, "w"))
    return {"plan": plan, "plan_text": ptxt, "plan_sample": plan_sample, "metrics": metrics,
            "samples": {k2: v for k2, v in samples.items()}}


def main():
    job = json.load(open(sys.argv[1]))
    t0 = time.time()
    eng = Engine(job.get("adapter"), job["max_rank"], {} if job.get("engine") == "bf16" else None)
    res = mode_eval(eng, job) if job["mode"] == "eval" else mode_collect(eng, job)
    res["gen_tokens"] = eng.gen_tokens
    res["seconds"] = time.time() - t0
    with open(job["out"] + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(job["out"] + ".tmp", job["out"])
    print("DONE", job["mode"], round(res["seconds"]), "s", eng.gen_tokens, "tokens", flush=True)


if __name__ == "__main__":
    main()
