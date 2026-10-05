"""Shared, intentionally small helpers for Lab 01 scripts.

These helpers locate the already-downloaded Qwen checkpoint by default.  Pass
``--download`` to a script only if you deliberately want Hugging Face to fetch
missing files.
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "Qwen/Qwen3-0.6B"


def resolve_snapshot(source: str = DEFAULT_REPO, *, allow_download: bool = False) -> Path:
    """Return a local model snapshot from a repo id or an existing directory."""
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return Path(
        snapshot_download(
            source,
            local_files_only=not allow_download,
        )
    )


def load_json(snapshot: Path, name: str) -> dict:
    with (snapshot / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_reference_tokenizer(snapshot: Path):
    """Load the HF tokenizer only for observation/comparison scripts."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte-display mapping used by Qwen's byte-level BPE."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    mapped = visible[:]
    next_codepoint = 256
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            mapped.append(next_codepoint)
            next_codepoint += 1
    return dict(zip(visible, map(chr, mapped)))


def escaped(value: str, *, width: int = 100) -> str:
    """A readable, single-line rendering that keeps whitespace visible."""
    result = repr(value)
    return result if len(result) <= width else result[: width - 3] + "..."
