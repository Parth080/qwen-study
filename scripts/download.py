"""Download Qwen3 checkpoints into the default HF cache (~/.cache/huggingface).

usage: python scripts/download.py Qwen/Qwen3-0.6B Qwen/Qwen3-8B
"""
import sys
from huggingface_hub import snapshot_download

for repo in sys.argv[1:] or ["Qwen/Qwen3-0.6B"]:
    path = snapshot_download(repo)  # full snapshot: hub 1.x rejects partial ones in local-only mode
    print(f"{repo} -> {path}")
