#!/usr/bin/env python3
"""Measure how Qwen3 tokenizes English, code, Hindi, numbers, and whitespace.

Run:
    python lab/01_tokenizer/encode_compare.py

The final Markdown table is designed to paste directly into NOTES.md.  A
single-token decode can display U+FFFD for a partial UTF-8 token; that is an
observation about byte-level BPE, not a decoding bug.
"""
from __future__ import annotations

import argparse

from _common import DEFAULT_REPO, escaped, load_reference_tokenizer, resolve_snapshot


SAMPLES = {
    "English": (
        "Tokenization decides how much of a document fits into a model's context window. "
        "Frequent English fragments often become compact tokens."
    ),
    "Python": "def doubled(values):\n    return [value * 2 for value in values]\n",
    "Hindi": "नमस्ते दुनिया, यह हिंदी टोकनाइज़ेशन का एक छोटा परीक्षण वाक्य है।",
    "12-digit number": "202609240123",
    "Phone number": "+91 98765 43210",
    "8 leading spaces": "        indented with spaces",
    "8 leading tabs": "\t\t\t\t\t\t\t\tindented with tabs",
    "URL + email": "https://huggingface.co/Qwen/Qwen3-8B?x=1&y=2  user.name+tag@example.com",
    "Emoji sequence": "emoji 🙂🚀👩\u200d💻",
}


def show_sample(tokenizer, label: str, text: str) -> tuple[int, int, float]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    print(f"\n=== {label} ===")
    print(f"text: {escaped(text, width=200)}")
    print(f"characters: {len(text)} | UTF-8 bytes: {len(text.encode('utf-8'))} | tokens: {len(ids)}")
    for position, token_id in enumerate(ids):
        one = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(f"  {position:>3}  id={token_id:>6}  {escaped(one, width=100)}")
    return len(ids), len(text), len(ids) / len(text) if text else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=DEFAULT_REPO, help="HF repo id or local snapshot directory")
    parser.add_argument("--download", action="store_true", help="download source if it is absent from the HF cache")
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.source, allow_download=args.download)
    tokenizer = load_reference_tokenizer(snapshot)
    rows = []
    for label, text in SAMPLES.items():
        tokens, chars, ratio = show_sample(tokenizer, label, text)
        rows.append((label, tokens, chars, ratio))

    print("\n=== Paste this table into NOTES.md ===")
    print("| input | tokens | chars | tokens/char |")
    print("|---|---:|---:|---:|")
    for label, tokens, chars, ratio in rows:
        print(f"| {label} | {tokens} | {chars} | {ratio:.3f} |")
    print("\nQuestions to answer in your own words:")
    print("  1. Which input was least compressed, and what about its bytes/merges explains it?")
    print("  2. Why are long numbers predictable but token-expensive here?")
    print("  3. Compare spaces and tabs: which indentation has learned reusable chunks?")


if __name__ == "__main__":
    main()
