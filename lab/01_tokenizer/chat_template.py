#!/usr/bin/env python3
"""Make Qwen3's chat formatting and thinking toggle visible.

Run:
    python lab/01_tokenizer/chat_template.py

This script is deliberately allowed to use the Hugging Face tokenizer: the
chat template itself is the object being studied in this step.
"""
from __future__ import annotations

import argparse
import difflib

from _common import DEFAULT_REPO, escaped, load_json, load_reference_tokenizer, resolve_snapshot


MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<think>\nobvious\n</think>\n\n4"},
    {"role": "user", "content": "And 3+3?"},
]


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=DEFAULT_REPO, help="HF repo id or local snapshot directory")
    parser.add_argument("--download", action="store_true", help="download source if it is absent from the HF cache")
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.source, allow_download=args.download)
    tokenizer = load_reference_tokenizer(snapshot)
    tokenizer_json = load_json(snapshot, "tokenizer.json")

    print(f"Snapshot: {snapshot}")
    print("\n=== Special versus added ===")
    print("all_special_tokens:", tokenizer.all_special_tokens)
    print("all_special_ids:   ", tokenizer.all_special_ids)
    think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    print(f"<think> encodes as {think_ids} ({len(think_ids)} token), but is it special? {'<think>' in tokenizer.all_special_tokens}")
    for item in sorted(tokenizer_json.get("added_tokens", []), key=lambda item: item["id"]):
        if item["content"] in {"<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"}:
            print(f"  {item['id']:>6}: {item['content']:<18} special={item.get('special', False)}")

    print("\n=== Raw chat template ===")
    template = getattr(tokenizer, "chat_template", None)
    if template is None and (snapshot / "chat_template.jinja").exists():
        template = (snapshot / "chat_template.jinja").read_text(encoding="utf-8")
    print(template or "No chat template was found in this snapshot.")

    rendered = {}
    for thinking in (True, False):
        text = tokenizer.apply_chat_template(
            MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        rendered[thinking] = text
        ids = tokenizer(text, add_special_tokens=False).input_ids
        print(f"\n=== enable_thinking={thinking} ({len(ids)} tokens) ===")
        print(text)
        print("first 16 ids:", ids[:16])
        marker_id = tokenizer("<|im_start|>", add_special_tokens=False).input_ids
        print(f"<|im_start|> ids: {marker_id} (one token: {len(marker_id) == 1})")

    print("\n=== Unified diff: thinking=True -> thinking=False ===")
    print("".join(difflib.unified_diff(
        rendered[True].splitlines(keepends=True),
        rendered[False].splitlines(keepends=True),
        fromfile="thinking=True",
        tofile="thinking=False",
    )) or "(no difference)")

    empty_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False, enable_thinking=True
    )
    print("\n=== A concrete turn-overhead estimate ===")
    print(f"empty user turn: {escaped(empty_user)}")
    print(f"empty user turn token count: {token_count(tokenizer, empty_user)}")
    print("This includes the role word, turn markers, and newline. Content can change the exact count at a BPE boundary.")
    print("\nSFT reminder: for non-thinking examples, render with the same template so the assistant prompt includes")
    print(repr("<think>\\n\\n</think>\\n\\n"))


if __name__ == "__main__":
    main()
