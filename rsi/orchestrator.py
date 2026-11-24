"""24-hour recursive self-improvement loop. Resumable: all state in STATE/state.json.

Each cycle:
  1. COLLECT  (champion) : model writes its own training plan, solves tasks with
                           tools, analyses its failures, retries with its own
                           lessons, proposes new tasks aimed at its weaknesses.
                           Only outcomes verified by hidden tests become data.
  2. EXPAND   (optional) : grow LoRA capacity if GPU headroom + stall/request.
  3. TRAIN    -> candidate
  4. GATE     : candidate vs champion on held-out tasks (first-try solve AND
                self-analysis-repair), paired test.
  5. ACCEPT or REVERT    : worse/no-better candidates are discarded, champion kept.
At the deadline: final held-out eval (never used for decisions) and export.
"""
import json, math, os, random, shutil, signal, subprocess, sys, time
from .config import (STATE, ROOT, TOTAL_HOURS, EXPORT_RESERVE_H, DEFAULT_PLAN, LORA_RANKS,
                     LORA_TARGETS_STAGES, ACCEPT_Z, ACCEPT_Z_MULTI, ENGINE_SIG, SMOKE, N_RETAIN, RETAIN_MAX_DROP)
from . import expand as X

SP = os.path.join(STATE, "state.json")
LOGS = os.path.join(STATE, "logs"); os.makedirs(LOGS, exist_ok=True)
AD = os.path.join(STATE, "adapters"); os.makedirs(AD, exist_ok=True)
TRAIN_TOKEN_BUDGET = int(os.environ.get("RSI_TRAIN_TOKENS", "900000"))


def log(msg, **kw):
    line = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": msg, **kw}
    print(json.dumps(line), flush=True)
    with open(os.path.join(STATE, "events.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")


def save(st):
    with open(SP + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(SP + ".tmp", SP)


def wait_gpu_free(max_wait=300):
    """Block until no other process holds GPU memory (vLLM's profiler aborts otherwise)."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.split()[0])
        except Exception:
            used = 0
        if used < 1024:
            time.sleep(3)
            return
        time.sleep(3)


def run_phase(module, job, name, timeout, retries=1):
    for attempt in range(retries + 1):
        rc = _run_phase(module, job, name, timeout)
        if rc == 0 or rc == -9:
            return rc
        log("phase failed, retrying" if attempt < retries else "phase failed", phase=name, rc=rc)
    return rc


CHILD = {"pid": None}


def _on_term(signum, frame):   # supervisor stop/restart: take the phase's process group down too
    if CHILD["pid"]:
        try:
            os.killpg(CHILD["pid"], 9)
        except ProcessLookupError:
            pass
    sys.exit(0)


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)
signal.signal(signal.SIGHUP, _on_term)


def _run_phase(module, job, name, timeout):
    wait_gpu_free()
    jp = os.path.join(STATE, f"job_{name}.json")
    json.dump(job, open(jp, "w"))
    lf = open(os.path.join(LOGS, f"{name}.log"), "a")
    env = dict(os.environ, PYTHONPATH=ROOT, VLLM_LOGGING_LEVEL="WARNING", TOKENIZERS_PARALLELISM="false")
    p = subprocess.Popen([sys.executable, "-m", f"rsi.{module}", jp], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    CHILD["pid"] = p.pid
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = -9
    finally:   # kill the whole group (vLLM spawns an EngineCore child)
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
        p.wait()
    lf.close()
    return rc


def paired(cand, champ):
    ids = sorted(set(cand) & set(champ))
    d = [cand[i] - champ[i] for i in ids]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1))
    z = m / (sd / math.sqrt(n)) if sd > 0 else (0.0 if m == 0 else math.copysign(9.9, m))
    return m, z


def history_text(st):
    lines = []
    for h in st["history"][-10:]:
        p = h["plan"]
        lines.append(f"cycle {h['cycle']}: plan lr={p['lr']:.1e} epochs={p['epochs']} tasks={p['n_train_tasks']} attempts={p['attempts']} "
                     f"temp={p['temperature']:.2f} repair_rounds={p['repair_rounds']} selfgen={p['n_selfgen']} mix={json.dumps(p['mix'])} expand={p['expand']}"
                     f" -> train_pass={h['metrics'].get('train_pass_rate', 0):.2f} repair_success={h['metrics'].get('repair_success_rate', 0):.2f}"
                     f" selfgen_frontier={h['metrics'].get('selfgen_frontier', 0)}/{h['metrics'].get('selfgen_proposed', 0)}"
                     f" gate={h.get('gate_score', 0):.3f} (delta {h.get('delta', 0):+.3f}, z={h.get('z', 0):.2f}) retain={h.get('retain_solved', '-')}/{h.get('retain_champion', '-')}"
                     f"{' recommended_hard_tasks=' + str(h['metrics'].get('recommended_hard_won')) + '/' + str(h['metrics'].get('recommended_tasks')) + ' hard-won' if h['metrics'].get('recommended_tasks') else ''}"
                     f"{' candidates: ' + ', '.join(f'{n}(z={r['z']:+.2f})' for n, r in h['candidates'].items()) + ' best=' + h['best'] if h.get('candidates') else ''}"
                     f" {'ACCEPTED' if h.get('accepted') else 'REVERTED'} [cycle took {h.get('minutes', 0):.0f} min]")
    left = (st["deadline"] - EXPORT_RESERVE_H * 3600 - time.time()) / 3600
    lines.append(f"TIME LEFT for training: {left:.1f} h. More tasks/attempts = more data but fewer cycles (each cycle is one chance to improve).")
    return "\n".join(lines)


_TASK_RATE = {}


def hard_won(sample):
    """Keep only samples that carry real learning signal: repairs, analyses, frontier task
    proposals, plans, and solves of tasks solved in SOME but not ALL attempts. Older,
    untagged solve samples are judged by the task's cumulative solve rate."""
    if sample["kind"] == "pair":
        return False
    if sample["kind"] != "solve":
        return True
    if "hard" in sample:
        return bool(sample["hard"])
    if not _TASK_RATE:
        ts = json.load(open(os.path.join(STATE, "task_stats.json")))
        pools = json.load(open(os.path.join(STATE, "pools.json")))
        by_prompt = {t["prompt"]: t["id"] for t in pools["train"]}
        _TASK_RATE["_p"] = by_prompt
        _TASK_RATE["_s"] = {k: v["solved"] / max(v["tries"], 1) for k, v in ts.items()}
    tid = _TASK_RATE["_p"].get(sample["messages"][1]["content"]) if len(sample["messages"]) > 1 else None
    r = _TASK_RATE["_s"].get(tid)
    return r is not None and 0 < r < 0.9


def make_retention_set():
    """Freeze N_RETAIN previously HARD-WON tasks (once failed, now usually solved) into a
    'retain' split, removed from training. Candidates must not lose them."""
    pp = os.path.join(STATE, "pools.json")
    pools = json.load(open(pp))
    if pools.get("retain"):
        return
    ts = json.load(open(os.path.join(STATE, "task_stats.json")))
    cand = [t for t in pools["train"] if t["id"] in ts and ts[t["id"]]["tries"] >= 8
            and 0.75 <= ts[t["id"]]["solved"] / ts[t["id"]]["tries"] < 1.0]
    rng = random.Random(1234)
    hard = [t for t in cand if t.get("difficulty") == "hard"]
    rest = [t for t in cand if t.get("difficulty") != "hard"]
    rng.shuffle(hard); rng.shuffle(rest)
    keep = (hard[:N_RETAIN // 2] + rest)[:N_RETAIN]
    ids = {t["id"] for t in keep}
    pools["retain"] = keep
    pools["train"] = [t for t in pools["train"] if t["id"] not in ids]
    json.dump(pools, open(pp + ".tmp", "w")); os.replace(pp + ".tmp", pp)
    log("retention set created", n=len(keep), hard=sum(t.get("difficulty") == "hard" for t in keep))


def build_dataset(st, cycle, res):
    """New verified samples + replay of earlier accepted-cycle samples, weighted by the plan's mix."""
    path = os.path.join(STATE, f"train_c{cycle}.jsonl")
    if os.path.exists(path):   # resumed cycle: dataset already built
        chosen = [json.loads(l) for l in open(path)]
        kinds = {}
        for s in chosen:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        return path, kinds, -1
    buf = os.path.join(STATE, "buffer.jsonl")
    old = [json.loads(l) for l in open(buf)] if os.path.exists(buf) else []
    new = [s for k, v in res["samples"].items() for s in v]
    if not any(o.get("cycle") == cycle for o in old):
        with open(buf, "a") as f:
            for s in new:
                f.write(json.dumps({**s, "cycle": cycle}) + "\n")
    old = [o for o in old if o.get("cycle") != cycle]
    new = [x for x in new if hard_won(x)]
    old = [x for x in old if hard_won(x)]
    plans = os.path.join(STATE, "plan_samples.jsonl")   # plans whose cycle was ACCEPTED
    good_plans = [json.loads(l) for l in open(plans)] if os.path.exists(plans) else []
    mix = res["plan"]["mix"]
    rng = random.Random(cycle)
    est = lambda s: sum(len(m["content"]) for m in s["messages"]) // 3 + 300
    pool = [(s, 2.0 * mix.get(s["kind"], 1.0)) for s in new] + \
           [(s, 0.5 * mix.get(s["kind"], 1.0)) for s in old] + \
           [(s, 1.0 * mix.get("plan", 0.2)) for s in good_plans]
    pool = [(s, w) for s, w in pool if w > 0]
    chosen, toks = [], 0
    budget = TRAIN_TOKEN_BUDGET / max(res["plan"]["epochs"], 1)
    while pool and toks < budget:
        i = rng.choices(range(len(pool)), weights=[w for _, w in pool])[0]
        s, _ = pool.pop(i)
        chosen.append(s); toks += est(s)
    with open(path, "w") as f:
        for s in chosen:
            f.write(json.dumps({"messages": s["messages"], "kind": s["kind"]}) + "\n")
    kinds = {}
    for s in chosen:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    return path, kinds, toks


def build_pairs(cycle, res, cap=160):
    """Preference pairs: this cycle's pass-vs-fail and fail-vs-repair pairs, topped up from the buffer."""
    path = os.path.join(STATE, f"pairs_c{cycle}.jsonl")
    if not os.path.exists(path):
        new = res["samples"].get("pair", [])
        buf = os.path.join(STATE, "buffer.jsonl")
        old = [x for x in (json.loads(l) for l in open(buf)) if x.get("kind") == "pair" and x.get("cycle") != cycle] if os.path.exists(buf) else []
        rng = random.Random(cycle)
        rng.shuffle(new); rng.shuffle(old)
        chosen = (new + old)[:cap]
        with open(path, "w") as f:
            for p in chosen:
                f.write(json.dumps({"chosen": p["chosen"], "rejected": p["rejected"]}) + "\n")
    return path, sum(1 for _ in open(path))


def build_restart_dataset(cycle, budget=1_200_000):
    """All hard-won experience accumulated so far, for training a fresh adapter from the base model."""
    path = os.path.join(STATE, f"restart_c{cycle}.jsonl")
    if not os.path.exists(path):
        buf = os.path.join(STATE, "buffer.jsonl")
        allx = [x for x in (json.loads(l) for l in open(buf)) if hard_won(x)]
        pl = os.path.join(STATE, "plan_samples.jsonl")
        allx += [json.loads(l) for l in open(pl)] if os.path.exists(pl) else []
        random.Random(cycle).shuffle(allx)
        toks = 0
        with open(path, "w") as f:
            for x in allx:
                t = sum(len(m["content"]) for m in x["messages"]) // 3 + 300
                if toks + t > budget:
                    break
                toks += t
                f.write(json.dumps({"messages": x["messages"], "kind": x["kind"]}) + "\n")
    return path, sum(1 for _ in open(path))


def candidate_specs(st, c, res, data, init, rank, stage):
    """Diverse moves each cycle. Incremental (the model's own plan) + DPO (learn from its own
    pass-vs-fail contrasts) + alternating Bold (large step) / Restart (fresh, larger adapter
    from the base model on all hard-won experience, to leave the champion's basin)."""
    plan = res["plan"]
    specs = {"incremental": {"desc": "SFT from champion with the model's own plan", "rank": rank, "stage": stage,
                             "job": {"data": data, "init_adapter": init, "rank": rank, "stage": stage, "lr": plan["lr"], "epochs": plan["epochs"]}}}
    pairs, npairs = build_pairs(c, res)
    if npairs >= (4 if SMOKE else 24):
        specs["dpo"] = {"desc": f"DPO+NLL on {npairs} own pass-vs-fail / fail-vs-repair pairs", "rank": rank, "stage": stage,
                        "job": {"mode": "dpo", "data": pairs, "init_adapter": init, "rank": rank, "stage": stage, "lr": 2e-5, "epochs": 1}}
    if c % 2 == 0:
        lr = min(3e-4, max(1e-4, 5 * plan["lr"]))
        specs["bold"] = {"desc": f"large step from champion: lr {lr:.1e}, 2 epochs", "rank": rank, "stage": stage,
                         "job": {"data": data, "init_adapter": init, "rank": rank, "stage": stage, "lr": lr, "epochs": 2}}
    else:
        rr, rs = max(32, rank), max(1, stage)
        rdata, n = build_restart_dataset(c, budget=40_000 if SMOKE else 1_200_000)
        specs["restart"] = {"desc": f"fresh rank-{rr} adapter from base on {n} accumulated hard-won samples", "rank": rr, "stage": rs,
                            "job": {"data": rdata, "init_adapter": None, "rank": rr, "stage": rs, "lr": 1e-4, "epochs": 1}}
    return specs


def main():
    if os.path.exists(SP):
        st = json.load(open(SP))
    else:
        now = time.time()
        st = {"start": now, "deadline": now + TOTAL_HOURS * 3600, "cycle": 0, "champion": None,
              "champ_gate": None, "rank": LORA_RANKS[0], "stage": 0, "history": [], "rejects_in_row": 0,
              "last_train": {"peak_vram_frac": 1.0, "gpu_util": 1.0}, "baseline": None, "done": False}
        save(st)
        log("start", deadline=time.ctime(st["deadline"]))
    if st.get("done"):
        log("already finished"); return
    if os.environ.get("RSI_EXTEND_H") and not st.get("extended"):
        st["deadline"] += float(os.environ["RSI_EXTEND_H"]) * 3600
        st["extended"] = True; save(st)
        log("deadline extended", hours=float(os.environ["RSI_EXTEND_H"]), new_deadline=time.ctime(st["deadline"]))
    stop_at = st["deadline"] - EXPORT_RESERVE_H * 3600
    max_rank = LORA_RANKS[-1]
    make_retention_set()

    # pools (cached)
    if not os.path.exists(os.path.join(STATE, "pools.json")):
        subprocess.run([sys.executable, "-c", "from rsi.tasks import load_pools; p=load_pools(); print({k: len(v) for k, v in p.items()})"],
                       cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), check=True)

    # engine changed since scores were recorded -> scores not comparable: re-measure
    if st["champ_gate"] is not None and st.get("engine_sig") != ENGINE_SIG:
        tag = st.get("engine_sig") or "bf16"
        if st["champion"] is None:
            for f in ("eval_baseline.json",):
                if os.path.exists(os.path.join(STATE, f)):
                    os.replace(os.path.join(STATE, f), os.path.join(STATE, f.replace(".json", f"_{tag}.json")))
            st["champ_gate"] = None
        else:
            out = os.path.join(STATE, f"eval_rebase_{ENGINE_SIG}.json")
            if not os.path.exists(out):
                rc = run_phase("rollout", {"mode": "eval", "adapter": st["champion"], "max_rank": LORA_RANKS[-1], "splits": ["gate"], "out": out}, "eval_rebase", 3 * 3600)
                assert rc == 0, "re-baseline failed"
            st["champ_gate"] = json.load(open(out))["gate"]["scores"]
            st["engine_sig"] = ENGINE_SIG
        save(st); log("engine changed: re-baselining", old=tag, new=ENGINE_SIG)

    # baseline: the untouched base model is the first champion
    if st["champ_gate"] is None:
        out = os.path.join(STATE, "eval_baseline.json")
        if not os.path.exists(out):
            have_bf16 = os.path.exists(os.path.join(STATE, "eval_baseline_bf16.json"))
            rc = run_phase("rollout", {"mode": "eval", "adapter": None, "max_rank": max_rank, "splits": ["gate"] if have_bf16 else ["gate", "final"], "out": out}, "eval_baseline", 4 * 3600)
            assert rc == 0, "baseline eval failed"
        r = json.load(open(out))
        st["champ_gate"] = r["gate"]["scores"]
        st["baseline"] = {s: {k: v for k, v in r[s].items() if k != "scores"} for s in r if s in ("gate", "final")}
        st["engine_sig"] = ENGINE_SIG
        save(st); log("baseline", **st["baseline"])

    if st["champ_gate"] is not None and st.get("champ_retain") is None:
        out = os.path.join(STATE, "eval_retain_champion.json")
        if not os.path.exists(out):
            rc = run_phase("rollout", {"mode": "eval", "adapter": st["champion"], "max_rank": max_rank, "splits": ["retain"], "out": out}, "eval_retain", 3600)
            assert rc == 0, "retention eval failed"
        r = json.load(open(out))["retain"]
        st["champ_retain"] = r["scores"]; save(st)
        log("champion retention baseline", pass1=r["pass1"], score=r["score"])

    while time.time() < stop_at:
        if os.path.exists(os.path.join(STATE, "STOP")):   # user asked to end the run early
            log("stop requested: ending training"); break
        mins = sorted(h.get("minutes", 0) for h in st["history"] if h.get("minutes"))
        est = mins[len(mins) // 2] * 60 if mins else 0
        if time.time() + est > stop_at:
            log("not enough time for another cycle", est_minutes=est / 60); break
        c = st["cycle"] + 1
        cyc_t0 = time.time()
        # ---------- 1. collect ----------
        cout = os.path.join(STATE, f"collect_c{c}.json")
        if not os.path.exists(cout):
            job = {"mode": "collect", "adapter": st["champion"], "max_rank": max_rank, "cycle": c,
                   "history": history_text(st), "rank": st["rank"], "targets": LORA_TARGETS_STAGES[st["stage"]],
                   "default_plan": DEFAULT_PLAN, "out": cout}
            rc = run_phase("rollout", job, f"collect_c{c}", 3 * 3600)
            if rc != 0:
                log("collect failed; retrying cycle", cycle=c, rc=rc); time.sleep(30); continue
        res = json.load(open(cout))
        log("collected", cycle=c, metrics=res["metrics"], plan=res["plan"], n={k: len(v) for k, v in res["samples"].items()})
        data, kinds, toks = build_dataset(st, c, res)
        log("dataset", cycle=c, kinds=kinds, est_tokens=toks)

        # ---------- 2. expansion (compute-aware) ----------
        init, new_rank, new_stage = st["champion"], st["rank"], st["stage"]
        st["plan"] = res["plan"]
        do, why = X.should_expand(st, st["last_train"]["peak_vram_frac"], st["last_train"]["gpu_util"])
        if do:
            nr, ns = X.next_capacity(st["rank"], st["stage"])
            extra = X.lora_param_count(nr, LORA_TARGETS_STAGES[ns]) - X.lora_param_count(st["rank"], LORA_TARGETS_STAGES[st["stage"]])
            pred = st["last_train"]["peak_vram_frac"] + extra * 10 / (24 * 2 ** 30)  # fp32 param+grad + 8bit adam
            if pred < 0.92:
                new_rank, new_stage = nr, ns
                if st["champion"]:
                    init = X.expand(st["champion"], os.path.join(AD, f"expanded_c{c}"), nr, LORA_TARGETS_STAGES[ns], seed=c)
                log("expand", cycle=c, reason=why, rank=nr, targets=LORA_TARGETS_STAGES[ns], predicted_peak=pred)
            else:
                log("expand skipped: would not fit", predicted_peak=pred)

        # ---------- 3. train a POPULATION of diverse candidates (escape local optima) ----------
        specs = candidate_specs(st, c, res, data, init, new_rank, new_stage)
        trained = {}
        for name, sp in specs.items():
            cand = os.path.join(AD, f"cand_c{c}_{name}")
            tstats = os.path.join(STATE, f"train_c{c}_{name}_stats.json")
            tj = {**sp["job"], "out_adapter": cand, "out_stats": tstats,
                  "deadline": min(time.time() + 35 * 60, stop_at)}
            if os.path.exists(tstats) and os.path.exists(os.path.join(cand, "adapter_model.safetensors")):
                rc = 0   # resumed
            else:
                rc = run_phase("train", tj, f"train_c{c}_{name}", 2 * 3600)
            if rc != 0 or not os.path.exists(tstats):
                log("candidate train failed", cycle=c, candidate=name, rc=rc)
                shutil.rmtree(cand, ignore_errors=True); continue
            ts = json.load(open(tstats))
            if name == "incremental" or "last_train" not in st:
                st["last_train"] = ts
            trained[name] = {"path": cand, "train": ts, **{k: v for k, v in sp.items() if k != "job"}}
            log("trained", cycle=c, candidate=name, **ts)
        if not trained:
            st["history"].append({"cycle": c, "plan": res["plan"], "metrics": res["metrics"], "accepted": False, "note": "all candidates failed to train"})
            st["cycle"] = c; st["rejects_in_row"] += 1; save(st); continue

        # ---------- 4. gate: all candidates in ONE engine session ----------
        eout = os.path.join(STATE, f"eval_c{c}.json")
        if not os.path.exists(eout):
            rc = run_phase("rollout", {"mode": "eval", "adapters": {n: t["path"] for n, t in trained.items()}, "max_rank": max_rank,
                                       "splits": ["gate", "retain"], "out": eout}, f"eval_c{c}", 4 * 3600)
        else:
            rc = 0
        if rc != 0:
            log("gate eval failed -> revert", cycle=c)
            for t in trained.values():
                shutil.rmtree(t["path"], ignore_errors=True)
            st["history"].append({"cycle": c, "plan": res["plan"], "metrics": res["metrics"], "accepted": False, "note": "eval failed"})
            st["cycle"] = c; st["rejects_in_row"] += 1; save(st); continue
        allev = json.load(open(eout))["candidates"]
        ret_ch = sum(v == 1.0 for v in (st.get("champ_retain") or {}).values())
        p1c = {k: (1.0 if v == 1.0 else 0.0) for k, v in st["champ_gate"].items()}
        results = {}
        for name, evs in allev.items():
            ev = evs["gate"]
            m, z = paired(ev["scores"], st["champ_gate"])
            _, zp1 = paired({k: (1.0 if v == 1.0 else 0.0) for k, v in ev["scores"].items()}, p1c)
            evr = evs.get("retain")
            ret_c = sum(v == 1.0 for v in evr["scores"].values()) if evr else 0
            retain_ok = (evr is None) or (ret_c >= ret_ch - RETAIN_MAX_DROP)
            results[name] = {"gate": ev["score"], "pass1": ev["pass1"], "repair": ev["repair_rate"], "delta": m, "z": z,
                             "z_pass1": zp1, "retain": ret_c, "retain_ok": retain_ok,
                             "eligible": m > 0 and zp1 > -1.0 and retain_ok, "desc": trained[name]["desc"]}
            log("candidate result", cycle=c, candidate=name, **{k: v for k, v in results[name].items() if k != "desc"})
        # best-of-N needs a stricter bar so that testing more candidates doesn't admit noise
        z_need = ACCEPT_Z if len(results) == 1 else ACCEPT_Z_MULTI
        elig = sorted([n for n, r in results.items() if r["eligible"]], key=lambda n: results[n]["z"], reverse=True)
        winner = elig[0] if elig and results[elig[0]]["z"] >= z_need else None
        best = winner or max(results, key=lambda n: results[n]["z"])
        accept = winner is not None
        br, bt = results[best], trained[best]

        # ---------- 5. accept / revert ----------
        h = {"cycle": c, "plan": res["plan"], "metrics": res["metrics"], "gate_score": br["gate"], "gate_pass1": br["pass1"],
             "gate_repair_rate": br["repair"], "delta": br["delta"], "z": br["z"], "z_pass1": br["z_pass1"], "accepted": accept,
             "best": best, "z_needed": z_need, "candidates": results,
             "rank": bt["rank"], "targets": LORA_TARGETS_STAGES[bt["stage"]],
             "retain_solved": br["retain"], "retain_champion": ret_ch, "retain_ok": br["retain_ok"], "train": bt["train"],
             "minutes": (time.time() - cyc_t0) / 60}
        if accept:
            keep = os.path.join(AD, f"champ_c{c}")
            shutil.rmtree(keep, ignore_errors=True); shutil.move(bt["path"], keep)
            old = st["champion"]
            st.update(champion=keep, champ_gate=allev[best]["gate"]["scores"], rank=bt["rank"], stage=bt["stage"], rejects_in_row=0)
            if allev[best].get("retain"):
                st["champ_retain"] = allev[best]["retain"]["scores"]
            if old and os.path.exists(old):
                shutil.rmtree(old, ignore_errors=True)
            if res.get("plan_sample"):
                with open(os.path.join(STATE, "plan_samples.jsonl"), "a") as f:
                    f.write(json.dumps(res["plan_sample"]) + "\n")
        else:
            st["rejects_in_row"] += 1
        for n, t in trained.items():
            if not (accept and n == best):
                shutil.rmtree(t["path"], ignore_errors=True)
        shutil.rmtree(os.path.join(AD, f"expanded_c{c}"), ignore_errors=True)
        st["history"].append(h); st["cycle"] = c; save(st)
        log("ACCEPTED" if accept else "REVERTED", cycle=c, best=best, gate=br["gate"], pass1=br["pass1"], delta=br["delta"],
            z=br["z"], z_needed=z_need, retain=f"{br['retain']}/{ret_ch}",
            all={n: round(r["z"], 2) for n, r in results.items()})

    # ---------- final held-out evaluation + export ----------
    fout = os.path.join(STATE, "eval_final.json")
    if st["champion"] and not os.path.exists(fout):
        run_phase("rollout", {"mode": "eval", "adapter": st["champion"], "max_rank": max_rank, "splits": ["gate", "final"], "out": fout, "engine": "bf16"}, "eval_final", 2 * 3600)
    if os.path.exists(fout):
        r = json.load(open(fout))
        st["final"] = {s: {k: v for k, v in r[s].items() if k != "scores"} for s in r if s in ("gate", "final")}
    bb = os.path.join(STATE, "eval_baseline_bf16.json")
    if os.path.exists(bb):   # report baseline in the same (bf16) precision as the final eval
        r = json.load(open(bb))
        st["baseline"] = {s: {k: v for k, v in r[s].items() if k != "scores"} for s in r if s in ("gate", "final")}
    save(st)
    st.setdefault("ended", time.time()); save(st)
    log("training finished", baseline=st["baseline"], final=st.get("final"))
    rc = subprocess.run([sys.executable, "-m", "rsi.export"], cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT)).returncode
    st["done"] = rc == 0; save(st)
    log("export", rc=rc)


if __name__ == "__main__":
    main()
