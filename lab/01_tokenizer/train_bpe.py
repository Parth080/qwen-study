#!/usr/bin/env python3
"""Train 8k and 32k byte-level BPE tokenizers, then compare compression.

Two easy modes:
    # A local corpus.  Supply a separate held-out file when possible.
    python lab/01_tokenizer/train_bpe.py --train-file path/to/train.txt --eval-file path/to/eval.txt

    # A Hugging Face text dataset.  This downloads data the first time.
    python lab/01_tokenizer/train_bpe.py --dataset roneneldan/TinyStories --text-field text

The metric is UTF-8 bytes/token, rather than tokens/character, because it is
meaningful across scripts with different Unicode character widths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import DEFAULT_REPO, load_reference_tokenizer, resolve_snapshot


def read_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Paragraphs make convenient independent training examples and keep memory modest.
    return [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]


def dataset_text(args) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.text_field not in dataset.column_names:
        raise ValueError(f"{args.text_field!r} is not a dataset column; available: {dataset.column_names}")
    dataset = dataset.shuffle(seed=args.seed)
    texts = [str(value) for value in dataset[args.text_field] if isinstance(value, str) and value.strip()]
    if len(texts) < args.eval_examples + 2:
        raise ValueError("not enough non-empty examples for both training and evaluation")
    texts = texts[: args.max_examples + args.eval_examples]
    return texts[args.eval_examples :], texts[: args.eval_examples]


def compression(name: str, encode, texts: list[str]) -> tuple[int, int, float]:
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    token_count = sum(len(encode(text)) for text in texts)
    if token_count == 0:
        raise ValueError("held-out text produced no tokens")
    return byte_count, token_count, byte_count / token_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-file", type=Path, help="UTF-8 text used to train the two tokenizers")
    source.add_argument("--dataset", help="Hugging Face dataset name; download is allowed in this mode")
    parser.add_argument("--eval-file", type=Path, help="separate UTF-8 held-out text (strongly recommended with --train-file)")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-examples", type=int, default=50_000, help="training examples in dataset mode")
    parser.add_argument("--eval-examples", type=int, default=2_000, help="held-out examples in dataset mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8_000, 32_000])
    parser.add_argument("--output-dir", type=Path, default=Path("lab/01_tokenizer/artifacts"))
    parser.add_argument("--qwen-source", default=DEFAULT_REPO, help="cached Qwen repo id or local snapshot for the comparison")
    args = parser.parse_args()

    if args.train_file:
        train_texts = read_text(args.train_file)
        if args.eval_file:
            eval_texts = read_text(args.eval_file)
        else:
            cutoff = max(1, int(len(train_texts) * 0.95))
            eval_texts, train_texts = train_texts[cutoff:], train_texts[:cutoff]
            print("WARNING: --eval-file was omitted; the last 5% of this file is being held out.")
    else:
        train_texts, eval_texts = dataset_text(args)

    if not train_texts or not eval_texts:
        raise ValueError("both the training and held-out corpora must contain non-empty text")

    from tokenizers import ByteLevelBPETokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[tuple[str, int, int, int, float]] = []
    for vocab_size in args.vocab_sizes:
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
        tokenizer.train_from_iterator(
            train_texts,
            vocab_size=vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=["<|endoftext|>"],
            show_progress=True,
        )
        output = args.output_dir / f"byte_bpe_{vocab_size // 1000}k"
        output.mkdir(exist_ok=True)
        tokenizer.save_model(str(output))
        byte_count, token_count, ratio = compression(
            f"own {vocab_size}", lambda text: tokenizer.encode(text).ids, eval_texts
        )
        measurements.append((f"own byte-BPE", vocab_size, byte_count, token_count, ratio))
        print(f"trained vocab={vocab_size:,}; saved vocab.json and merges.txt to {output}")

    qwen = load_reference_tokenizer(resolve_snapshot(args.qwen_source))
    byte_count, token_count, ratio = compression(
        "Qwen", lambda text: qwen(text, add_special_tokens=False).input_ids, eval_texts
    )
    measurements.append(("Qwen3", qwen.vocab_size, byte_count, token_count, ratio))

    print("\n=== Held-out compression (paste into NOTES.md) ===")
    print(f"Evaluation examples: {len(eval_texts):,}; UTF-8 bytes: {byte_count:,}")
    print("| tokenizer | vocab | held-out bytes | held-out tokens | bytes/token |")
    print("|---|---:|---:|---:|---:|")
    for name, vocab_size, bytes_, tokens, ratio in measurements:
        print(f"| {name} | {vocab_size:,} | {bytes_:,} | {tokens:,} | {ratio:.3f} |")
    print("\nInterpretation prompts:")
    print("  - Does a larger vocabulary reduce tokens per document on this held-out corpus?")
    print("  - Which model matrices become larger when vocab grows?")
    print("  - Is your corpus representative of the languages/domains you want to model?")


if __name__ == "__main__":
    main()
