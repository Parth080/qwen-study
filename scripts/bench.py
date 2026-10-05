"""Inference tokens/sec with mlx-lm. Fills the §7 worksheet in PLAN.md.

usage: python scripts/bench.py Qwen/Qwen3-0.6B models/Qwen3-8B-4bit ...
"""
import sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate

PROMPT = "Explain what a transformer attention head does, in detail."

for repo in sys.argv[1:] or ["Qwen/Qwen3-0.6B"]:
    model, tok = load(repo)
    msgs = [{"role": "user", "content": PROMPT}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    for _ in stream_generate(model, tok, prompt, max_tokens=8):   # warm-up: compile kernels, wire memory
        pass
    mx.reset_peak_memory()
    t0 = time.perf_counter(); n = 0
    for r in stream_generate(model, tok, prompt, max_tokens=256):
        n += 1
    wall = time.perf_counter() - t0
    print(f"{repo}: prompt {r.prompt_tokens} tok @ {r.prompt_tps:.0f} tok/s | "
          f"gen {r.generation_tokens} tok @ {r.generation_tps:.1f} tok/s (wall {n/wall:.1f}) | "
          f"peak mem {r.peak_memory:.2f} GB")
    del model; mx.clear_cache()
