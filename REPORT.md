# RSI-ANN2 — 44h self-improvement run report

Run extended by 24h after the first day; from then on training used only hard-won samples, complex tasks were recommended when self-generated tasks/plans fell short, and candidates also had to keep a frozen retention set of 40 previously hard-won tasks (`retain`).

Base model: Qwen2.5-Coder-7B-Instruct. Run: Sun Nov 22 20:27:23 2026 → Tue Nov 24 16:29:23 2026 UTC (stopped early by the operator after cycle 22; scheduled end was Tue Nov 24 20:27:23 2026).

Analysis and recommendations: see [REVIEW.md](REVIEW.md).

All scores come from executing code against hidden tests. `gate` decided accept/revert; `final` (HumanEval + unused MBPP test) was never used for any decision, so it is the honest measure.

Score per task: 1.0 = solved first try, 0.5 = solved after its own failure analysis + retry, 0 = unsolved.

| split | metric | baseline | final champion |
|---|---|---|---|
| gate | score | 0.652 | 0.703 |
| gate | pass1 | 0.606 | 0.659 |
| gate | repair_rate | 0.232 | 0.267 |
| final | score | 0.650 | 0.654 |
| final | pass1 | 0.606 | 0.621 |
| final | repair_rate | 0.227 | 0.187 |

## Cycles

| cycle | accepted | gate score | Δ | z | train pass | repair success | self-gen frontier | rank/targets | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ↩️ reverted | 0.000 | +0.000 | 0.00 | 0.40 | 0.12 | 11/96 | /0 | 0 |
| 2 | ↩️ reverted | 0.000 | +0.000 | 0.00 | 0.39 | 0.06 | 6/128 | /0 | 0 |
| 3 | ✅ | 0.709 | +0.083 | 4.26 | 0.38 | 0.07 | 6/96 | 16/7 | 70 |
| 4 | ↩️ reverted | 0.702 | -0.008 | -0.46 | 0.59 | 0.12 | 5/96 | 16/7 | 70 |
| 5 | ↩️ reverted | 0.686 | -0.023 | -1.51 | 0.58 | 0.10 | 3/96 | 16/7 | 76 |
| 6 | ↩️ reverted | 0.702 | -0.008 | -0.49 | 0.56 | 0.15 | 6/128 | 32/7 | 71 |
| 7 | ↩️ reverted | 0.698 | -0.011 | -0.67 | 0.58 | 0.09 | 5/96 | 32/7 | 74 |
| 8 | ↩️ reverted | 0.697 | -0.013 | -0.65 | 0.59 | 0.06 | 4/128 | 32/7 | 138 |
| 9 | ↩️ reverted | 0.720 | +0.011 | 0.66 | 0.57 | 0.13 | 4/96 | 32/7 | 73 |
| 10 | ↩️ reverted | 0.711 | +0.002 | 0.09 | 0.56 | 0.18 | 9/96 | 32/7 | 74 |
| 11 | ↩️ reverted | 0.677 | -0.033 | -2.08 | 0.61 | 0.09 | 7/96 | 32/7 | 70 |
| 12 | ↩️ reverted | 0.702 | -0.008 | -0.54 | 0.60 | 0.16 | 3/96 | 32/7 | 71 |
| 13 | ↩️ reverted | 0.698 | -0.011 | -0.72 | 0.57 | 0.16 | 3/96 | 32/7 | 72 |
| 14 | ↩️ reverted | 0.681 | -0.028 | -1.66 | 0.56 | 0.17 | 2/128 | 32/7 | 115 |
| 15 | ↩️ reverted | 0.706 | -0.003 | -0.21 | 0.57 | 0.10 | 4/96 | 32/7 | 88 |
| 16 | ↩️ reverted | 0.697 | -0.013 | -0.70 | 0.58 | 0.10 | 3/128 | 32/7 | 171 |
| 17 | ↩️ reverted | 0.716 | +0.006 | 0.36 | 0.58 | 0.12 | 2/128 | 32/7 | 180 |
| 18 | ↩️ reverted | 0.717 | +0.008 | 0.45 | 0.58 | 0.12 | 5/128 | 32/7 | 154 |
| 19 | ↩️ reverted | 0.720 | +0.011 | 0.66 | 0.61 | 0.18 | 4/128 | 32/7 | 155 |
| 20 | ↩️ reverted | 0.730 | +0.020 | 1.29 | 0.52 | 0.12 | 5/128 | 32/7 | 155 |
| 21 | ↩️ reverted | 0.709 | +0.000 | 0.00 | 0.57 | 0.13 | 5/128 | 32/7 | 189 |
| 22 | ↩️ reverted | 0.723 | +0.014 | 0.91 | 0.58 | 0.12 | 8/192 | 32/7 | 165 |

## Population cycles (candidate z vs champion; ✔ = passed retention check)

| cycle | incremental | dpo | bold | restart | needed |
|---|---|---|---|---|---|
| 17 | -2.19 ✔ | +0.36 ✔ | – | -1.83 ✔ | 1.5 |
| 18 | -0.10 ✘ | -0.42 ✔ | +0.45 ✘ | – | 1.5 |
| 19 | -0.72 ✔ | +0.66 ✔ | – | -0.41 ✔ | 1.5 |
| 20 | -0.65 ✔ | +1.29 ✘ | -0.77 ✔ | – | 1.5 |
| 21 | +0.00 ✔ | -0.35 ✔ | – | -2.31 ✘ | 1.5 |
| 22 | -1.21 ✔ | +0.91 ✔ | -0.09 ✘ | – | 1.5 |
