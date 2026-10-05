#!/usr/bin/env python3
"""Calculate Qwen KV-cache memory directly from a local config.json."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen/Qwen3-8B")
    parser.add_argument("--context", type=int, default=32_000)
    parser.add_argument("--bytes-per-value", type=int, default=2, help="2 for bf16/fp16")
    args = parser.parse_args()
    path = Path(args.model) if Path(args.model).is_dir() else Path(snapshot_download(args.model, local_files_only=True))
    config = json.loads((path / "config.json").read_text())
    per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * args.bytes_per_value
    print(f"{args.model}: {per_token:,} bytes/token = {per_token / 1024:.1f} KiB/token")
    print(f"context {args.context:,}: {per_token * args.context / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
