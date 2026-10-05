# qwen — Qwen3 research lab

Hands-on dissection of the Qwen3 model family on Apple Silicon (M4 Pro, 48 GB, no CUDA), followed by a data pipeline, LoRA / QLoRA fine-tuning, and an ablation study. The learning targets (tokenizer, forward pass, optimizer, loss, quantization, LoRA) are implemented from scratch in `src/qwen_lab/` and checked against Hugging Face and PyTorch by `tests/`.

## Setup

```sh
uv sync                                        # .venv with Python 3.12, torch (MPS), MLX, transformers
source .venv/bin/activate
python scripts/check_env.py                    # both frameworks see the GPU
python scripts/download.py Qwen/Qwen3-0.6B Qwen/Qwen3-8B
python scripts/smoke_torch.py                  # transformers forward pass on MPS
python scripts/bench.py Qwen/Qwen3-0.6B        # inference tok/s
python scripts/bench_train.py Qwen/Qwen3-0.6B  # LoRA training tok/s
pytest                                         # reference checks for the from-scratch code
```

Before 8B work: `sudo sysctl iogpu.wired_limit_mb=40960` raises the Metal memory cap (resets on reboot).

## Repository map

```
src/qwen_lab/          from-scratch implementations (the learning targets)
  tokenizer/bpe.py       byte-level BPE matching Qwen3 token-for-token
  model/qwen3_torch.py   readable Qwen3 forward pass, loads HF safetensors
  model/quantization.py  symmetric / asymmetric group quantization
  train/optimizers.py    AdamW (and later SGD, Lion, Adafactor) by hand
  train/lora.py          LoRA around one nn.Linear, merge / unmerge
  eval/sampling.py       temperature, top-k, top-p, min-p, repetition penalty
  eval/losses.py         causal cross-entropy with the label shift
  analysis/hooks.py      activation capture and patching
  analysis/logit_lens.py project a residual stream through the LM head
  data/ train/ eval/     Phase 2+ : data pipeline, MLX LoRA trainer, task metric
tests/                 checks that compare each implementation to HF / torch
lab/NN_topic/*.py      observation and demo scripts, one folder per topic
scripts/               env check, download, smoke test, benchmarks, diff helper
experiments/           configs/A0.yaml baseline, results.csv (one row per run)
models/                locally converted models (e.g. Qwen3-8B 4-bit MLX); gitignored
data/                  raw / cleaned / splits; gitignored
```

Plans, lessons, task sheets, personal notes, and worked solutions are kept local and are not part of this repository (see `.gitignore`).

## Measured on this machine (mlx-lm)

| | 0.6B bf16 | 8B 4-bit |
|---|---|---|
| generation | 170 tok/s | 54 tok/s |
| LoRA training (r=8, batch 4, seq 1024) | 2,050 tok/s | 188 tok/s |
| LoRA training peak memory | 4.8 GB | 11.8 GB |
