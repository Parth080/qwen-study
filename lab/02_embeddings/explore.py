"""Print embedding shapes, parameter counts, and tying from cached Qwen configs."""
from transformers import AutoConfig

for name in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"):
    config = AutoConfig.from_pretrained(name, local_files_only=True)
    rows = config.vocab_size * config.hidden_size
    heads = 1 if config.tie_word_embeddings else 2
    print(f"{name}: embedding={config.vocab_size} x {config.hidden_size} = {rows:,}; "
          f"LM head {'tied' if config.tie_word_embeddings else 'untied'}; "
          f"embedding/head parameters={rows * heads:,}")
