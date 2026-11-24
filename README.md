# RSI-ANN2

A self-improving agentic coding model: a LoRA adapter on Qwen2.5-Coder-7B-Instruct, produced by a 22-cycle autonomous loop. Each cycle the model plans, attempts tasks, analyses its failures, proposes tasks, trains, and is accepted or reverted against hidden tests.

- [REVIEW.md](REVIEW.md): results, an honest assessment, and recommendations
- [REPORT.md](REPORT.md): per-cycle and per-candidate tables
- `model/adapter`: champion adapter. `python merge.py <out_dir>` rebuilds the full weights
- `rsi/`: the training loop. `run/`: logs and evaluation data
