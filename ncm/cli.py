#!/usr/bin/env python3
"""ncm CLI - High-Performance Neural Codec for Code."""

import sys
import os
import argparse
import hashlib
from .compressor import NCMCompressor

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "weights", "codec_4x_best.pt")

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="NCM: Neural Codec for Code (Lossless Binary Compressor & Tokenizer)")
    subparsers = parser.add_subparsers(dest="cmd")

    # Encode
    p_enc = subparsers.add_parser("encode", help="Compress source code into a packed .ncm binary file")
    p_enc.add_argument("input", help="Source code file (.py, etc.)")
    p_enc.add_argument("-o", "--out", required=True, help="Destination .ncm binary file")
    p_enc.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint .pt")

    # Decode
    p_dec = subparsers.add_parser("decode", help="Decompress a .ncm binary file back to exact source code")
    p_dec.add_argument("input", help="Source .ncm binary file")
    p_dec.add_argument("-o", "--out", required=True, help="Destination restored file")
    p_dec.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint .pt")

    # Bench
    p_bench = subparsers.add_parser("bench", help="Benchmark compression ratio, on-disk size, and exact lossless recovery")
    p_bench.add_argument("input", help="Source code file")
    p_bench.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint .pt")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.ckpt):
        # Look in runs or parent dirs if default doesn't exist
        alt = r"D:\ncm-work\runs\r3-5080-ar-r8d2K4096\codec_best.pt"
        if os.path.exists(alt):
            args.ckpt = alt
        else:
            print(f"Error: Checkpoint not found at {args.ckpt}.")
            print("Run with --ckpt <path/to/codec.pt> or download weights.")
            sys.exit(1)

    comp = NCMCompressor(args.ckpt)

    if args.cmd == "encode":
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            sys.exit(1)
        orig_sz, comp_sz = comp.compress_file(args.input, args.out)
        ratio = orig_sz / max(1, comp_sz)
        print(f"[*] Compressed {args.input} -> {args.out}")
        print(f"    Original Size   : {orig_sz:,} bytes")
        print(f"    Compressed Size : {comp_sz:,} bytes")
        print(f"    Compression     : {ratio:.2f}x ({100 * (1 - comp_sz / orig_sz):.1f}% reduction)")
        print(f"    SHA256 (Source) : {file_sha256(args.input)}")

    elif args.cmd == "decode":
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            sys.exit(1)
        comp_sz, rest_sz = comp.decompress_file(args.input, args.out)
        print(f"[*] Decompressed {args.input} -> {args.out}")
        print(f"    Compressed Size : {comp_sz:,} bytes")
        print(f"    Restored Size   : {rest_sz:,} bytes")
        print(f"    SHA256 (Output) : {file_sha256(args.out)}")

    elif args.cmd == "bench":
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            sys.exit(1)
        with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"[*] Benchmarking NCM on {args.input} ({len(text):,} chars)...")
        res = comp.benchmark_text(text)
        print("\n================== NCM Compression Results ==================")
        print(f"  Exact Bit-for-Bit Lossless : {res['exact_match']}")
        print(f"  Original Tokens            : {res['n_tokens']:,}")
        print(f"  Original File Size         : {res['orig_bytes']:,} bytes")
        print(f"  Packed .ncm Bitstream      : {res['compressed_bytes']:,} bytes")
        print(f"  Empirical Bits / Token     : {res['empirical_bpt']:.2f} bpt")
        print(f"  Physical Data Reduction    : {res['compression_ratio']:.2f}x")
        print(f"  Throughput (Encode / Decode): {res['encode_time_s']:.3f}s / {res['decode_time_s']:.3f}s")
        print("============================================================\n")

if __name__ == "__main__":
    main()
