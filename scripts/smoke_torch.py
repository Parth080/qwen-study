"""Forward pass with transformers on MPS: prints top-5 next tokens and peak memory.

usage: python scripts/smoke_torch.py [Qwen/Qwen3-0.6B]
"""
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("mps").eval()
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").to("mps")
print("prompt tokens:", ids.input_ids.shape[1])
with torch.no_grad():
    t = time.perf_counter(); out = model(**ids); torch.mps.synchronize(); dt = time.perf_counter() - t
logits = out.logits[0, -1].float()
top = torch.topk(logits.softmax(-1), 5)
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {p:.3f}  {tok.decode([i])!r}")
print(f"forward {dt*1000:.0f} ms, logits shape {tuple(out.logits.shape)}, peak mem {torch.mps.driver_allocated_memory()/1e9:.2f} GB")
