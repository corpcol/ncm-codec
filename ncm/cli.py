#!/usr/bin/env python3
"""ncm CLI - Command Line Interface for Neural Codec Model."""

import sys
import os
import argparse
from .compressor import NCMCompressor

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "weights", "codec_4x_best.pt")

def main():
    parser = argparse.ArgumentParser(description="NCM: Neural Codec for Code")
    subparsers = parser.add_subparsers(dest="cmd")

    p_bench = subparsers.add_parser("bench", help="Benchmark compression and lossless recovery on a file")
    p_bench.add_argument("input", help="Source code file")
    p_bench.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint .pt")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    if args.cmd == "bench":
        if not os.path.exists(args.input):
            print(f"Error: {args.input} not found.")
            sys.exit(1)
        text = open(args.input, "r", encoding="utf-8", errors="ignore").read()
        print(f"[*] Benchmarking NCM on {args.input} ({len(text)} chars)...")
        comp = NCMCompressor(args.ckpt)
        res = comp.compress_text(text)
        print("\n--- NCM Results ---")
        print(f"Original Tokens  : {res['n_tokens']}")
        print(f"Codes per Token  : {res['codes_per_token']}")
        print(f"Lossless BPT     : {res['bits_per_token']:.2f} bits/token")
        print(f"Exact Lossless   : {res['exact_match']}")
        print(f"Escape Rate      : {res['escape_rate']*100:.2f}%")

if __name__ == "__main__":
    main()
