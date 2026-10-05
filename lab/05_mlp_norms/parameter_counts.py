"""Compute exact SwiGLU projection counts from Qwen configs."""
from transformers import AutoConfig

for name in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"):
    c = AutoConfig.from_pretrained(name, local_files_only=True)
    projections = 3 * c.hidden_size * c.intermediate_size
    print(f"{name}: 3 x {c.hidden_size} x {c.intermediate_size} = {projections:,} parameters per MLP")
