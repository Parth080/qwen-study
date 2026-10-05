# models/

Only **locally converted** models live here (e.g. `Qwen3-8B-4bit`, made with `mlx_lm convert`). Gitignored.

The original Hugging Face checkpoints are **not** here. They are in the shared HF cache:

    ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B     (1.4 GB, bf16)
    ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B       (15 GB, bf16)

Both `transformers` and `mlx-lm` look there automatically when you pass the repo id
(`"Qwen/Qwen3-0.6B"`), so nothing needs to be copied. Files inside a snapshot are symlinks
to content-addressed blobs; do not move them by hand.

    hf cache ls          # list cached repos and sizes
    hf cache rm <id>     # delete one
