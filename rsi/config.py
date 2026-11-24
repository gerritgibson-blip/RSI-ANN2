"""Global configuration. Everything resumable lives under STATE."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = os.environ.get("RSI_BASE_MODEL", "/workspace/models/base")
STATE = os.environ.get("RSI_STATE", "/workspace/rsi_state")
os.makedirs(STATE, exist_ok=True)

TOTAL_HOURS = float(os.environ.get("RSI_HOURS", "24"))
EXPORT_RESERVE_H = 1.25          # time reserved at the end for final eval + export
GITHUB_REPO = os.environ.get("RSI_GITHUB_REPO", "RSI-ANN2")   # GitHub turns spaces into '-'

MAX_TURNS = 6                    # agent tool-call turns per episode
MAX_NEW_TOKENS = 1024            # per assistant turn
MAX_SEQ_LEN = 6144               # training/rollout context

# Inference engine settings. Gate scores are only comparable under one engine
# config, so the orchestrator re-baselines whenever ENGINE_SIG changes.
ENGINE = {"quantization": "fp8"}
ENGINE_SIG = "fp8w-v1"

# LoRA capacity: start small, grow via expand.py (function-preserving)
LORA_RANKS = [16, 32, 64, 128]
LORA_TARGETS_STAGES = [
    ["q_proj", "k_proj", "v_proj", "o_proj"],
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
]

# Default (overridable by the model's own meta-plan, within BOUNDS)
DEFAULT_PLAN = {
    "lr": 5e-5, "epochs": 1, "n_train_tasks": 384, "attempts": 4, "temperature": 0.8,
    "repair_rounds": 1, "n_selfgen": 96,
    "mix": {"solve": 1.0, "repair": 1.0, "analysis": 0.5, "selfgen": 0.5, "plan": 0.2},
    "expand": False,
}
BOUNDS = {
    "lr": (1e-5, 2e-4), "epochs": (1, 3), "n_train_tasks": (128, 448), "attempts": (2, 6),
    "temperature": (0.4, 1.1), "repair_rounds": (1, 3), "n_selfgen": (32, 192),
}
# self-improvement samples can never be switched off by the model's own plan
MIX_FLOOR = {"solve": 0.3, "repair": 0.75, "analysis": 0.75, "selfgen": 0.5, "plan": 0.3}
# if fewer than this many self-proposed tasks land at the frontier (or the plan is
# unusable), the loop recommends N_RECOMMENDED unseen hard tasks to attempt
SELFGEN_MIN_FRONTIER = 8
N_RECOMMENDED = 96
N_RETAIN = 40          # frozen retention set of previously hard-won tasks
RETAIN_MAX_DROP = 1    # candidate may lose at most this many retention tasks vs champion
# acceptance: paired sign test on gate tasks
ACCEPT_Z = 1.0
ACCEPT_Z_MULTI = 1.5   # stricter bar when the best of several candidates is picked

# ---- smoke-test mode (tiny everything) ----
SMOKE = os.environ.get("RSI_SMOKE") == "1"
if SMOKE:
    EXPORT_RESERVE_H = 0.0
    DEFAULT_PLAN.update(n_train_tasks=12, n_selfgen=6, attempts=2)
    BOUNDS.update(n_train_tasks=(8, 12), n_selfgen=(4, 6), attempts=(2, 2))
    N_RECOMMENDED, N_RETAIN = 4, 6
