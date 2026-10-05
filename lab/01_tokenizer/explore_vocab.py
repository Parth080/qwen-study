#!/usr/bin/env python3
"""Inspect Qwen3's tokenizer.json with plain JSON, not a tokenizer library.

Run from the repository root:
    python lab/01_tokenizer/explore_vocab.py
    python lab/01_tokenizer/explore_vocab.py --limit 30

The output answers the concrete observations requested in Lab 01, step 1.
Copy the interpretations (not necessarily every line) into NOTES.md.
"""
from __future__ import annotations

import argparse
from collections import Counter

from _common import DEFAULT_REPO, bytes_to_unicode, escaped, load_json, resolve_snapshot


def merge_as_pair(merge: str | list[str]) -> tuple[str, str]:
    if isinstance(merge, str):
        return tuple(merge.split(" ", 1))  # tokenizer.json's usual representation
    return tuple(merge)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=DEFAULT_REPO, help="HF repo id or local snapshot directory")
    parser.add_argument("--download", action="store_true", help="download source if it is absent from the HF cache")
    parser.add_argument("--limit", type=int, default=256, help="number of lowest-id base entries to print")
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.source, allow_download=args.download)
    tokenizer = load_json(snapshot, "tokenizer.json")
    model_config = load_json(snapshot, "config.json")
    vocab: dict[str, int] = tokenizer["model"]["vocab"]
    merges = tokenizer["model"]["merges"]
    added = tokenizer.get("added_tokens", [])
    byte_map = bytes_to_unicode()

    print(f"Snapshot: {snapshot}")
    print("\n=== Pipeline sections from tokenizer.json ===")
    for name in ("normalizer", "pre_tokenizer", "decoder"):
        print(f"\n{name}:")
        print(tokenizer.get(name))

    print("\n=== Merge history ===")
    for label, part in (("first", merges[:30]), ("last", merges[-30:])):
        print(f"{label} 30 merges (their displayed order is their rank):")
        for rank, merge in enumerate(part, 0 if label == "first" else len(merges) - len(part)):
            left, right = merge_as_pair(merge)
            print(f"  {rank:>6}: ({escaped(left, width=35)}, {escaped(right, width=35)})")

    print("\n=== Selected vocabulary entries ===")
    targets = {
        "single space": byte_map[ord(" ")],
        "newline": byte_map[ord("\n")],
        "four spaces": byte_map[ord(" ")] * 4,
        "tab": byte_map[ord("\t")],
        "the": "the",
        " the (the byte-mapped spelling)": byte_map[ord(" ")] + "the",
        "The": "The",
        " The (the byte-mapped spelling)": byte_map[ord(" ")] + "The",
    }
    for label, spelling in targets.items():
        print(f"  {label:<36} {escaped(spelling, width=28):<30} -> {vocab.get(spelling)}")

    print(f"\n=== Lowest {args.limit} ids (the first 256 are the byte alphabet) ===")
    by_id = sorted(vocab.items(), key=lambda item: item[1])
    for spelling, token_id in by_id[: args.limit]:
        print(f"  {token_id:>6}: {escaped(spelling)}")

    print("\n=== Longest vocabulary strings ===")
    for spelling, token_id in sorted(vocab.items(), key=lambda item: (-len(item[0]), item[1]))[:20]:
        print(f"  id={token_id:>6} chars={len(spelling):>3}  {escaped(spelling, width=150)}")

    leading_space = byte_map[ord(" ")]
    count = sum(token.startswith(leading_space) for token in vocab)
    print("\n=== Leading-space tokens ===")
    print(f"  starts with {escaped(leading_space)}: {count:,} / {len(vocab):,} = {count / len(vocab):.2%}")

    print("\n=== Added tokens (matched before BPE) ===")
    for token in sorted(added, key=lambda item: item["id"]):
        print(f"  {token['id']:>6}: {token['content']:<28} special={token.get('special', False)}")

    real_token_count = len(vocab) + len(added)
    padded_rows = model_config["vocab_size"] - real_token_count
    print("\n=== Vocabulary-size arithmetic ===")
    print(f"  BPE vocab entries:       {len(vocab):,}")
    print(f"  added tokens:            {len(added):,}")
    print(f"  tokenizer-real tokens:   {real_token_count:,}")
    print(f"  model vocab_size:        {model_config['vocab_size']:,}")
    print(f"  non-token padding rows:  {padded_rows:,}")
    print(f"  vocab_size / 128:        {model_config['vocab_size'] / 128:g}")
    print(f"  padding rows / 128:      {padded_rows / 128:g}")

    print("\n=== Quick sanity counts ===")
    print(f"  BPE base + merges expected: 256 + {len(merges):,} = {256 + len(merges):,}")
    print(f"  token spelling initials: {Counter(token[:1] for token in vocab).most_common(8)}")


if __name__ == "__main__":
    main()
