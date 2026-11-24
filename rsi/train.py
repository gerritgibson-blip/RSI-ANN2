"""Training phase: LoRA fine-tune on verified self-generated samples.
Run as: python -m rsi.train <job.json>

Loss is computed only on the model's own (assistant) tokens. bf16 base weights
(same numerics vLLM serves), gradient checkpointing, 8-bit AdamW, and the
LM head is applied only at supervised positions to keep memory low.
"""
import json, math, os, random, sys, threading, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from .config import BASE_MODEL, MAX_SEQ_LEN, LORA_TARGETS_STAGES
from .agent import TOOLS


def render(tok, msgs, tools, gen=False):
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=False)


def encode(tok, sample):
    msgs = sample["messages"]
    tools = TOOLS if sample["kind"] in ("solve", "repair") else None
    full = tok(render(tok, msgs, tools), add_special_tokens=False).input_ids
    labels = [-100] * len(full)
    for j, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        a = len(tok(render(tok, msgs[:j], tools, gen=True), add_special_tokens=False).input_ids)
        b = len(tok(render(tok, msgs[:j + 1], tools), add_special_tokens=False).input_ids)
        for i in range(a, min(b, len(full))):
            labels[i] = full[i]
    full, labels = full[:MAX_SEQ_LEN], labels[:MAX_SEQ_LEN]
    if all(l == -100 for l in labels):
        return None
    return full, labels


class GPUMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        import pynvml
        pynvml.nvmlInit()
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.util, self.stop = [], False

    def run(self):
        while not self.stop:
            self.util.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            time.sleep(2)


def seq_logp(body, head, ids, lab, grad):
    """Sum log-prob of the model's own (labelled) tokens; chunked LM head."""
    x = torch.tensor([ids], device="cuda")
    y = torch.tensor(lab[1:], device="cuda")
    pos = (y != -100).nonzero().squeeze(1)
    f = lambda hh, yy: -F.cross_entropy(head(hh).float(), yy, reduction="sum")
    with torch.set_grad_enabled(grad):
        h = body(input_ids=x).last_hidden_state[0, :-1][pos]
        yy = y[pos]
        lp = 0.0
        for j in range(0, h.shape[0], 1024):
            lp = lp + (checkpoint(f, h[j:j + 1024], yy[j:j + 1024], use_reentrant=False) if grad else f(h[j:j + 1024], yy[j:j + 1024]))
    return lp, int(pos.numel())


def train_dpo(job, model, body, head, pairs, opt, params, deadline, t0, beta=0.1, alpha=0.2, accum=8):
    """DPO + small NLL anchor on the chosen trajectory (RPO-style), reference = the
    starting adapter. Gradients use exact per-sequence coefficients computed from
    no-grad passes, so only ONE sequence graph is alive at a time (memory-safe)."""
    model.eval()
    ref = []
    for (ci, cl), (ri, rl) in pairs:   # reference log-probs (initial policy)
        try:
            ref.append((seq_logp(body, head, ci, cl, False)[0].item(), seq_logp(body, head, ri, rl, False)[0].item()))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); ref.append(None)
    model.train()
    steps = math.ceil(len(pairs) * job["epochs"] / accum)
    losses, step, skipped, accs = [], 0, 0, []
    for ep in range(job["epochs"]):
        order = list(range(len(pairs))); random.Random(ep).shuffle(order)
        for i in range(0, len(order), accum):
            if time.time() > deadline:
                print("training deadline reached", flush=True); break
            tot = 0.0
            for k in order[i:i + accum]:
                if ref[k] is None:
                    continue
                (ci, cl), (ri, rl) = pairs[k]
                rc, rr = ref[k]
                try:
                    with torch.no_grad():
                        pc, nc = seq_logp(body, head, ci, cl, False)
                        pr, _ = seq_logp(body, head, ri, rl, False)
                    margin = beta * ((pc.item() - rc) - (pr.item() - rr))
                    sig = 1 / (1 + math.exp(margin))            # sigmoid(-margin)
                    coef_c = (-beta * sig - alpha / max(nc, 1)) / accum
                    coef_r = (beta * sig) / accum
                    lp, _ = seq_logp(body, head, ci, cl, True); (coef_c * lp).backward()
                    lp, _ = seq_logp(body, head, ri, rl, True); (coef_r * lp).backward()
                    tot += (math.log1p(math.exp(-margin)) - alpha * pc.item() / max(nc, 1)) / accum
                    accs.append(margin > 0)
                except torch.OutOfMemoryError:
                    skipped += 1; torch.cuda.empty_cache()
                    print("OOM on pair - skipped", flush=True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(tot); step += 1
            if step % 5 == 0:
                print(f"dpo step {step}/{steps} loss {sum(losses[-5:]) / 5:.4f} pref_acc {sum(accs[-40:]) / max(len(accs[-40:]), 1):.2f} {time.time() - t0:.0f}s", flush=True)
    return step, losses, skipped


def main():
    job = json.load(open(sys.argv[1]))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model
    import bitsandbytes as bnb

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    data = [json.loads(l) for l in open(job["data"])]
    dpo = job.get("mode") == "dpo"
    if dpo:   # preference pairs: (chosen, rejected) agent trajectories on the same task
        enc = []
        for d in data:
            c = encode(tok, {"messages": d["chosen"], "kind": "solve"})
            r = encode(tok, {"messages": d["rejected"], "kind": "solve"})
            if c and r:
                enc.append((c, r))
        print(f"{len(enc)} preference pairs", flush=True)
    else:
        enc = [e for e in (encode(tok, s) for s in data) if e]
        print(f"{len(enc)} samples, {sum(len(e[0]) for e in enc)} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    if job.get("init_adapter"):
        model = PeftModel.from_pretrained(model, job["init_adapter"], is_trainable=True)
    else:
        r = job["rank"]
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS_STAGES[job["stage"]], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in params), flush=True)

    opt = bnb.optim.AdamW8bit(params, lr=job["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    accum = 16
    steps = math.ceil(len(enc) * job["epochs"] / accum)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    inner = model.get_base_model()
    body, head = inner.model, inner.lm_head
    mon = GPUMonitor(); mon.start()
    torch.cuda.reset_peak_memory_stats()
    deadline = job["deadline"]
    losses, step, t0, skipped = [], 0, time.time(), 0
    chunk_loss = lambda hh, yy: F.cross_entropy(head(hh).float(), yy, reduction="sum")
    model.train()
    if dpo:
        step, losses, skipped = train_dpo(job, model, body, head, enc, opt, params, deadline, t0)
        steps = step
    for ep in range(0 if dpo else job["epochs"]):
        random.Random(ep).shuffle(enc)
        for i in range(0, len(enc), accum):
            if time.time() > deadline:
                print("training deadline reached", flush=True); break
            batch = enc[i:i + accum]
            ntok = sum(sum(l != -100 for l in lab[1:]) for _, lab in batch)
            tot = 0.0
            for ids, lab in batch:
                try:
                    x = torch.tensor([ids], device="cuda")
                    y = torch.tensor(lab[1:], device="cuda")
                    pos = (y != -100).nonzero().squeeze(1)
                    h = body(input_ids=x).last_hidden_state[0, :-1][pos]
                    yy = y[pos]
                    loss = 0.0
                    for j in range(0, h.shape[0], 1024):   # chunked, recomputed LM head: no full logits tensor
                        loss = loss + checkpoint(chunk_loss, h[j:j + 1024], yy[j:j + 1024], use_reentrant=False)
                    loss = loss / ntok
                    loss.backward()
                    tot += loss.item()
                except torch.OutOfMemoryError:
                    skipped += 1
                    x = y = h = loss = None
                    torch.cuda.empty_cache()
                    print(f"OOM on sample of {len(ids)} tokens - skipped", flush=True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(tot); step += 1
            if step % 5 == 0:
                print(f"step {step}/{steps} loss {sum(losses[-5:]) / 5:.4f} {time.time() - t0:.0f}s", flush=True)
    mon.stop = True
    peak = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
    for p in params:
        p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(job["out_adapter"])
    stats = {"steps": step, "loss_first": sum(losses[:5]) / max(1, len(losses[:5])),
             "loss_last": sum(losses[-5:]) / max(1, len(losses[-5:])), "peak_vram_frac": peak,
             "gpu_util": (sum(mon.util) / max(len(mon.util), 1)) / 100, "seconds": time.time() - t0,
             "train_tokens": (sum(len(c[0]) + len(r[0]) for c, r in enc) if dpo else sum(len(e[0]) for e in enc)) * job["epochs"],
             "skipped_oom": skipped, "mode": job.get("mode", "sft")}
    json.dump(stats, open(job["out_stats"], "w"))
    print("TRAIN DONE", stats, flush=True)


if __name__ == "__main__":
    main()
