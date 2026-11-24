# RSI-ANN2 — final review and recommendations

**Run:** Qwen2.5-Coder-7B-Instruct + LoRA. 22 self-improvement cycles over about 44 h on one RTX PRO 4000 (24 GB). Stopped by the operator after cycle 22.
**Champion:** the cycle 3 adapter (`model/adapter`, rank 16, attention + MLP). Every later candidate was rejected by the gate.

## 1. What the final model actually gained

Every number comes from running code against hidden tests, scored in bf16.

| split | metric | base model | champion | Δ |
|---|---|---|---|---|
| gate (decisions were made on this) | score | 0.652 | 0.703 | **+0.052** |
| gate | pass@1 | 0.606 | 0.659 | +0.053 |
| gate · KodCode | score | 0.554 | 0.667 | **+0.113** |
| gate · MBPP | score | 0.710 | 0.725 | +0.015 |
| **final (never used for any decision)** | score | 0.650 | 0.654 | +0.004 |
| final | pass@1 | 0.606 | 0.621 | +0.015 |
| final · HumanEval | score | 0.476 | 0.473 | −0.003 |
| final · MBPP | score | 0.745 | 0.753 | +0.008 |
| final | self-repair success | 0.227 | 0.187 | −0.040 |

**Honest reading:**
- The model improved clearly on the *kind* of task it trained on: KodCode-style problems gained about 11 points on held-out gate tasks.
- On the never-touched final set it is essentially unchanged. pass@1 is up slightly, HumanEval is flat, and the final-set gains are within noise.
- Self-repair (fixing its own failure after self-analysis) did **not** get measurably better on held-out tasks.
- So the loop produced real but narrow, in-distribution gains. It did not produce a broad capability jump. The independent final set exists precisely to show this, and it worked: gate improvements did not carry over.

## 2. What happened over the cycles

- **Cycles 1–3:** one accepted step (cycle 3, z = 4.26). It came when the LoRA adapter was widened from attention-only to attention + MLP. All of the measured gain comes from that step.
- **Cycles 4–16:** single-candidate SFT on the model's own verified successes and repairs. All 13 were reverted, with gate deltas between −0.033 and +0.011. More capacity (rank 32 from cycle 13) did not help. Training only on hard-won samples (from cycle 14) did not help either.
- **Cycles 17–22, population mode:** 3 candidates per cycle with a stricter acceptance bar (z ≥ 1.5).

| strategy | tries | mean z | best z | retention kept |
|---|---|---|---|---|
| DPO+NLL (own pass-vs-fail and fail-vs-repair pairs) | 6 | **+0.41** | **+1.29** | 5/6 |
| bold (5× learning rate, 2 epochs) | 3 | −0.14 | +0.45 | 1/3 |
| incremental SFT (model's own plan) | 6 | −0.81 | 0.00 | 5/6 |
| restart (fresh rank-32 adapter from base) | 3 | −1.51 | −0.41 | 2/3 |

- **DPO** was the only strategy whose results leaned positive. In cycle 20 it reached gate 0.730 (z = 1.29) but lost 2 retention tasks. In cycle 22 it was eligible at z = 0.91, but below the 1.5 bar. Learning from its own failures through preference contrasts works better than more imitation of its own successes.
- **Plain self-SFT is saturated.** Once the model has learned from its own successful rollouts, further rounds re-teach what it already does.
- **Restarting from the base model loses to the champion.** About 1M tokens of self-generated data is not enough to rebuild what the champion has.
- **The retention check did its job.** It blocked 5 of 18 candidates that would otherwise have quietly forgotten previously hard-won tasks, including both of the "bold" candidates that scored best on z.

## 3. How well the self-improvement machinery worked

| capability | result |
|---|---|
| Self-planning | Plans parsed in most cycles; 9 had to fall back to the default plan. Early on the model tried to turn off its own self-analysis (repair rounds = 0), so minimums had to be enforced. Later plans changed little: it alternated between learning rates of 1e-5 and 5e-5 and never learned which settings paid off, even with per-candidate z-scores in its history. |
| Self-generated tasks | Weak. Of 2,528 proposed tasks, 257 (10%) were valid (reference solution passes, empty solution fails), and only 110 (4%) were at the frontier: hard but solvable. The model mostly proposes tasks that are too easy for itself. |
| Recommended hard tasks (fallback) | Effective. They triggered in every cycle from 15 to 22, and 38–62 of each batch of 96 became hard-won training data. This was the main source of useful training signal late in the run. |
| Failure analysis + repair | Repair success after self-analysis stayed flat at 6–18% per cycle and never trended upward. Its lessons are often generic, so the repair is mostly a re-sample. |
| Accept / revert | Worked as designed. With paired z-tests on hidden tests plus a retention set, no regression was ever accepted, and the final set confirmed the gate was not overfit by an accepted change. |
| Parameter expansion | Function-preserving and triggered by compute headroom. Widening to the MLP layers is what produced the one accepted improvement. Rank 16→32 did not help. |

## 4. Recommendations for the next run

1. **Make DPO the main update, not one candidate among three.** It was the only strategy with a positive mean. Collect far more pairs per cycle: sample 8+ attempts on *frontier* tasks only (solve rate between 20% and 80%). Use the online / iterative variant, regenerating pairs from the current policy each cycle.
2. **Change the gate to reduce noise.** With 320 gate tasks, one standard error is about 0.015. That is the same size as the improvements now being attempted, so true +1-point gains cannot be detected. Use a larger gate (≥ 1,000 tasks) or score each gate task with several samples, and/or use sequential testing that accumulates evidence over several cycles.
3. **Fix the task generator before relying on it.** Reward proposals by how close their measured solve rate is to 50% (learnability), not just by validity. Seed proposals by mutating recommended hard tasks, since those already sit at the right difficulty. A 4% frontier yield is the main bottleneck on self-generated curriculum.
4. **Train the failure analysis directly.** Today only analyses that led to a successful repair are kept, which is too sparse at 10–15%. Instead, score a lesson by how much it raises the repair success rate over several re-samples compared with having no lesson. Then train on the best and worst lessons as preference pairs.
5. **Use a harder, broader training distribution.** The gains stayed within KodCode / MBPP style. Add multi-file and repository-level tasks with longer tool loops (e.g. SWE-style or LiveCodeBench-style tasks with hidden tests). These are where agentic self-correction matters and where the base model is not already saturated.
6. **Teach the planner from outcomes.** Turn the per-candidate z history into plan-preference data, so the planner learns which plans led to accepted models. Otherwise, remove model-written hyperparameters and let it plan only the curriculum.
7. **Stronger base or full fine-tuning if compute allows.** The capacity expansion was the only step that moved the gate meaningfully. A higher-rank adapter on all layers, from the start, with DPO, is the cheapest next experiment.

## 5. Files

- `model/adapter`: champion LoRA adapter (Git LFS). Rebuild full weights with `python merge.py <out_dir>`.
- `REPORT.md`: per-cycle table and per-candidate z table.
- `run/`:
  - `state.json`: full history, including plans, metrics and candidate results.
  - `events.jsonl`: event log.
  - `eval_*.json`: per-task scores (baseline, retention, final).
- `rsi/`: the complete self-improvement loop (orchestrator, vLLM rollouts and sandbox, LoRA/DPO trainer, expansion tool, export).
