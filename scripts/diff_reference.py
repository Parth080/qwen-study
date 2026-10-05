"""Diff your implementation against the local reference solution. Use AFTER your tests pass.

usage: python scripts/diff_reference.py src/qwen_lab/tokenizer/bpe.py
       (the matching file under reference/ is found automatically)
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

mine = Path(sys.argv[1])
ref = Path("reference") / mine.relative_to("src")
if not ref.exists():
    sys.exit(f"no reference for {mine} (expected {ref})")
sys.stdout.writelines(difflib.unified_diff(
    mine.read_text().splitlines(keepends=True), ref.read_text().splitlines(keepends=True),
    fromfile=str(mine), tofile=str(ref),
))
