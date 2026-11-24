"""Export: write report + champion adapter into the repo and push to GitHub.

Needs GITHUB_TOKEN (repo scope) in the environment / /workspace/.env.
Repo name: RSI_GITHUB_REPO (default "RSI-ANN2"; GitHub does not allow spaces).
The full merged 7B weights (~15 GB) exceed GitHub limits, so the repo carries
the LoRA adapter (plain git, < 100 MB) plus `merge.py` to rebuild the full model.
"""
import json, os, shutil, subprocess, time, urllib.request
from .config import STATE, ROOT, GITHUB_REPO, BASE_MODEL, SMOKE


def sh(*cmd, check=True, **kw):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True, **kw)


def report(st):
    end = st.get("ended") or time.time()
    hrs = (end - st["start"]) / 3600
    L = [f"# RSI-ANN2 — {hrs:.0f}h self-improvement run report", "",
         "Run extended by 24h after the first day; from then on training used only hard-won samples, "
         "complex tasks were recommended when self-generated tasks/plans fell short, and candidates also had to "
         "keep a frozen retention set of 40 previously hard-won tasks (`retain`).", "",
         f"Base model: Qwen2.5-Coder-7B-Instruct. Run: {time.ctime(st['start'])} → {time.ctime(end)} UTC "
         f"(stopped early by the operator after cycle {st['cycle']}; scheduled end was {time.ctime(st['deadline'])}).", "",
         "Analysis and recommendations: see [REVIEW.md](REVIEW.md).", "",
         "All scores come from executing code against hidden tests. `gate` decided accept/revert; `final` "
         "(HumanEval + unused MBPP test) was never used for any decision, so it is the honest measure.", "",
         "Score per task: 1.0 = solved first try, 0.5 = solved after its own failure analysis + retry, 0 = unsolved.", "",
         "| split | metric | baseline | final champion |", "|---|---|---|---|"]
    for s in ("gate", "final"):
        b, f = (st.get("baseline") or {}).get(s, {}), (st.get("final") or {}).get(s, {})
        for k in ("score", "pass1", "repair_rate"):
            if k in b:
                L.append(f"| {s} | {k} | {b[k]:.3f} | {f.get(k, float('nan')):.3f} |")
    L += ["", "## Cycles", "", "| cycle | accepted | gate score | Δ | z | train pass | repair success | self-gen frontier | rank/targets | minutes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h in st["history"]:
        m = h["metrics"]
        L.append(f"| {h['cycle']} | {'✅' if h.get('accepted') else '↩️ reverted'} | {h.get('gate_score', 0):.3f} | {h.get('delta', 0):+.3f} | {h.get('z', 0):.2f} | "
                 f"{m.get('train_pass_rate', 0):.2f} | {m.get('repair_success_rate', 0):.2f} | {m.get('selfgen_frontier', 0)}/{m.get('selfgen_proposed', 0)} | "
                 f"{h.get('rank', '')}/{len(h.get('targets', []))} | {h.get('minutes', 0):.0f} |")
    pop = [h for h in st["history"] if h.get("candidates")]
    if pop:
        L += ["", "## Population cycles (candidate z vs champion; ✔ = passed retention check)", "",
              "| cycle | incremental | dpo | bold | restart | needed |", "|---|---|---|---|---|---|"]
        for h in pop:
            cell = lambda n: (f"{h['candidates'][n]['z']:+.2f} {'✔' if h['candidates'][n]['retain_ok'] else '✘'}" if n in h["candidates"] else "–")
            L.append(f"| {h['cycle']} | {cell('incremental')} | {cell('dpo')} | {cell('bold')} | {cell('restart')} | {h.get('z_needed', 0):.1f} |")
    return "\n".join(L) + "\n"


MERGE = '''"""Rebuild full weights: python merge.py <out_dir>"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen2.5-Coder-7B-Instruct"
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "model/adapter").merge_and_unload()
m.save_pretrained(sys.argv[1]); AutoTokenizer.from_pretrained(base).save_pretrained(sys.argv[1])
'''


def main():
    if SMOKE or os.environ.get("RSI_NO_EXPORT") == "1":
        print("smoke/test run: export disabled (no repo changes, no push)")
        return
    st = json.load(open(os.path.join(STATE, "state.json")))
    os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "run"), exist_ok=True)
    if st.get("champion"):
        dst = os.path.join(ROOT, "model", "adapter")
        shutil.rmtree(dst, ignore_errors=True); shutil.copytree(st["champion"], dst)
    open(os.path.join(ROOT, "REPORT.md"), "w").write(report(st))
    open(os.path.join(ROOT, "merge.py"), "w").write(MERGE)
    for f in ("state.json", "events.jsonl", "eval_baseline.json", "eval_baseline_bf16.json", "eval_final.json", "eval_retain_champion.json"):
        if os.path.exists(os.path.join(STATE, f)):
            shutil.copy(os.path.join(STATE, f), os.path.join(ROOT, "run", f))
    open(os.path.join(ROOT, ".gitignore"), "w").write("__pycache__/\n*.pyc\n")

    if not os.path.exists(os.path.join(ROOT, ".git")):
        sh("git", "init", "-b", "main")   # adapter (~78 MB) is under GitHub's 100 MB limit: plain git, no LFS
    sh("git", "add", "-A")
    sh("git", "-c", "user.name=RSI-ANN2", "-c", "user.email=rsi-ann2@localhost", "commit", "-m",
       "RSI-ANN2: self-improvement run export\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", check=False)

    tok = os.environ.get("GITHUB_TOKEN")
    if not tok and os.path.exists("/workspace/.env"):   # token may be added after the run started
        for line in open("/workspace/.env"):
            if line.startswith("GITHUB_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        print("GITHUB_TOKEN not set: repo committed locally at", ROOT, "- set it and rerun `python -m rsi.export`.")
        return
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "rsi-export"}
    user = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/user", headers=hdr)))["login"]
    try:
        req = urllib.request.Request("https://api.github.com/user/repos", method="POST", headers=hdr,
                                     data=json.dumps({"name": GITHUB_REPO, "private": True, "description": "RSI-ANN2: self-improving coding agent (LoRA on Qwen2.5-Coder-7B)"}).encode())
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code != 422:   # 422 = already exists
            raise
    url = f"https://github.com/{user}/{GITHUB_REPO}.git"
    import base64
    auth = base64.b64encode(f"x-access-token:{tok}".encode()).decode()
    r = sh("git", "-c", f"http.extraheader=Authorization: Basic {auth}", "push", url, "main", check=False)
    print(r.stdout, r.stderr.replace(tok, "***"))
    if r.returncode != 0:
        raise SystemExit("push failed")
    print("pushed to", f"https://github.com/{user}/{GITHUB_REPO}")


if __name__ == "__main__":
    main()
