"""Parameter-expansion tool.

Grows the trainable capacity of the model (LoRA rank and/or which weight
matrices are adapted) WITHOUT changing its current behaviour:
  * new rows of lora_A are random, new columns of lora_B are zero, and
    lora_alpha is scaled with the rank, so B@A*alpha/r is unchanged;
  * newly adapted modules get B = 0, so they start as an exact no-op.
The expanded model therefore starts from exactly the champion's outputs and
must still win the held-out gate to be kept; otherwise it is reverted.

Trigger (`should_expand`): spare GPU headroom during training (memory/compute
going unused) AND improvement has stalled or the model itself asked for it.

CLI: python -m rsi.expand <adapter_dir> <out_dir> [--rank R] [--stage S]
"""
import json, math, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file
from .config import LORA_RANKS, LORA_TARGETS_STAGES, BASE_MODEL


def _shapes():
    cfg = json.load(open(os.path.join(BASE_MODEL, "config.json")))
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    kv = cfg["num_key_value_heads"] * (h // cfg["num_attention_heads"])
    return cfg["num_hidden_layers"], {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
                                      "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}


def lora_param_count(rank, targets):
    L, sh = _shapes()
    return L * sum(rank * (a + b) for m, (a, b) in sh.items() if m in targets)


def next_capacity(rank, stage):
    """Alternate: widen targets first, then rank."""
    if stage + 1 < len(LORA_TARGETS_STAGES):
        return rank, stage + 1
    idx = LORA_RANKS.index(rank)
    if idx + 1 < len(LORA_RANKS):
        return LORA_RANKS[idx + 1], stage
    return None


def should_expand(state, gpu_peak_frac, gpu_util):
    """Expand when compute is being left on the table and learning has stalled."""
    if next_capacity(state["rank"], state["stage"]) is None:
        return False, "at max capacity"
    headroom = gpu_peak_frac < 0.80          # >20% VRAM unused at training peak
    stalled = state.get("rejects_in_row", 0) >= 2
    asked = bool(state.get("plan", {}).get("expand"))
    if headroom and (stalled or asked):
        return True, f"headroom (peak VRAM {gpu_peak_frac:.0%}, util {gpu_util:.0%}) and {'stall' if stalled else 'model request'}"
    return False, f"headroom={headroom} stalled={stalled} asked={asked}"


def expand(src, dst, new_rank, new_targets, seed=0):
    g = torch.Generator().manual_seed(seed)
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    sd = load_file(os.path.join(src, "adapter_model.safetensors"))
    r0, alpha0 = cfg["r"], cfg["lora_alpha"]
    assert new_rank >= r0
    out = {}
    for k, v in sd.items():
        if "lora_A" in k:     # [r, in]
            extra = torch.empty(new_rank - r0, v.shape[1]).uniform_(-1, 1, generator=g) / math.sqrt(v.shape[1])
            out[k] = torch.cat([v, extra.to(v.dtype)], 0)
        elif "lora_B" in k:   # [out, r]
            out[k] = torch.cat([v, torch.zeros(v.shape[0], new_rank - r0, dtype=v.dtype)], 1)
        else:
            out[k] = v
    # brand-new target modules: A random, B zero  => no-op at start
    L, sh = _shapes()
    dt = next(iter(sd.values())).dtype
    for m in new_targets:
        if m in cfg["target_modules"]:
            continue
        part = "self_attn" if m.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
        fin, fout = sh[m]
        for l in range(L):
            pre = f"base_model.model.model.layers.{l}.{part}.{m}"
            out[pre + ".lora_A.weight"] = (torch.empty(new_rank, fin).uniform_(-1, 1, generator=g) / math.sqrt(fin)).to(dt)
            out[pre + ".lora_B.weight"] = torch.zeros(fout, new_rank, dtype=dt)
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "adapter_model.safetensors"))
    cfg["r"] = new_rank
    cfg["lora_alpha"] = alpha0 * new_rank / r0       # keeps alpha/r constant => function preserved
    cfg["target_modules"] = sorted(set(cfg["target_modules"]) | set(new_targets))
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=2)
    for f in os.listdir(src):
        if f.startswith(("tokenizer", "special_tokens", "vocab", "merges")):
            shutil.copy(os.path.join(src, f), dst)
    return dst


if __name__ == "__main__":
    a = sys.argv[1:]
    src, dst = a[0], a[1]
    rank = int(a[a.index("--rank") + 1]) if "--rank" in a else json.load(open(os.path.join(src, "adapter_config.json")))["r"]
    stage = int(a[a.index("--stage") + 1]) if "--stage" in a else 0
    expand(src, dst, rank, LORA_TARGETS_STAGES[stage])
    print("expanded ->", dst, "rank", rank, "params", lora_param_count(rank, LORA_TARGETS_STAGES[stage]))
