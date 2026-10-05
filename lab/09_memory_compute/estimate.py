#!/usr/bin/env python3
"""A transparent first-pass run-time estimate for a proposed experiment."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--tokens", type=float, required=True, help="mean tokens/example")
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--tokens-per-second", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    total = args.examples * args.tokens * args.epochs
    minutes = total / args.tokens_per_second / 60
    print(f"tokens/run: {total:,.0f}")
    print(f"minutes/run: {minutes:.1f}")
    print(f"hours for {args.seeds} seed(s): {minutes * args.seeds / 60:.2f}")


if __name__ == "__main__":
    main()
