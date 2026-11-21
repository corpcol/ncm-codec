"""Production drop-in replacement for ncm/cli.py in corpcol/ncm-codec.

Commands:
  ncm compress <input> <output.ncm> [--ckpt weights/codec_4x_best.pt]
  ncm decompress <input.ncm> <output.txt> [--ckpt weights/codec_4x_best.pt]
  ncm bench <input> [--ckpt weights/codec_4x_best.pt]

Aliases supported:
  ncm encode <input> -o <output.ncm>
  ncm decode <input.ncm> -o <output.txt>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

try:
    from .compressor import NCMCompressor
    from .weights import get_weights_path
except ImportError:
    from compressor import NCMCompressor
    try:
        from weights import get_weights_path
    except ImportError:
        def get_weights_path(m="4x"):
            return Path("../weights/codec_4x_best.pt")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def resolve_ckpt(ckpt_arg: Path | None) -> Path:
    if ckpt_arg is not None and ckpt_arg.exists():
        return ckpt_arg
    standard = Path(get_weights_path("4x"))
    if standard.exists():
        return standard
    if ckpt_arg is not None:
        return ckpt_arg
    return Path("weights/codec_4x_best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ncm", description="NCM Neural Arithmetic Codec CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("compress", "decompress", "bench", "encode", "decode"):
        p = sub.add_parser(name)
        p.add_argument("input", type=Path, help="Input file path")
        if name in ("compress", "decompress", "encode", "decode"):
            # Allow positional output or -o/--out
            p.add_argument("output", type=Path, nargs="?", default=None, help="Output destination")
            p.add_argument("-o", "--out", type=Path, default=None, help="Output destination alias")
        p.add_argument("--ckpt", type=Path, default=None, help="Path to checkpoint .pt")
        p.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
        p.add_argument("--top-k", type=int, default=64)
        p.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    cmd = "compress" if args.cmd == "encode" else ("decompress" if args.cmd == "decode" else args.cmd)

    output_path = getattr(args, "output", None) or getattr(args, "out", None)
    if cmd in ("compress", "decompress") and output_path is None:
        parser.error(f"Command '{args.cmd}' requires an output file path (either as positional arg or with -o/--out)")

    ckpt_path = resolve_ckpt(args.ckpt)
    if not ckpt_path.exists():
        print(f"[!] Warning: Checkpoint not found at {ckpt_path}.")
        print("    Run with --ckpt <path> or train weights with `python training/train.py`.")
        sys.exit(1)

    comp = NCMCompressor(
        ckpt_path=str(ckpt_path),
        tokenizer_name=args.tokenizer,
        device=args.device,
        top_k=args.top_k,
    )

    if cmd == "compress":
        text = args.input.read_text(encoding="utf-8")
        t0 = time.perf_counter()
        res = comp.compress_text(text, verify=True)
        dt = time.perf_counter() - t0
        if not res.exact_match:
            raise RuntimeError("Lossless verification FAILED!")
        output_path.write_bytes(res.blob)
        print(
            f"[OK] {args.input.name} -> {output_path.name} | "
            f"{res.raw_bytes:,} B -> {res.compressed_bytes:,} B "
            f"({res.actual_bpt:.3f} bpt vs lzma-9 {res.lzma9_bpt:.3f} bpt) | "
            f"top1={res.top1_acc*100:.1f}% topK={res.topk_hit_rate*100:.1f}% "
            f"oov={res.oov_rate*100:.2f}% | {dt:.2f}s | SHA256={_sha256(text.encode('utf-8'))}"
        )

    elif cmd == "decompress":
        blob = args.input.read_bytes()
        t0 = time.perf_counter()
        text = comp.decompress_text(blob)
        dt = time.perf_counter() - t0
        output_path.write_text(text, encoding="utf-8")
        print(
            f"[OK] {args.input.name} -> {output_path.name} | "
            f"{len(blob):,} B -> {len(text.encode('utf-8')):,} B | "
            f"{dt:.2f}s | SHA256={_sha256(text.encode('utf-8'))}"
        )

    elif cmd == "bench":
        text = args.input.read_text(encoding="utf-8")
        t0 = time.perf_counter()
        res = comp.compress_text(text, verify=True)
        dt = time.perf_counter() - t0
        ratio_vs_raw = res.raw_bytes / max(res.compressed_bytes, 1)
        ratio_vs_lzma = res.lzma9_bytes / max(res.compressed_bytes, 1)
        print(
            f"File: {args.input.name} ({res.orig_num_tokens:,} tokens, {res.raw_bytes:,} B)\n"
            f"  NCM2 Actual : {res.compressed_bytes:,} B ({res.actual_bpt:.3f} bpt, {ratio_vs_raw:.2f}x vs raw)\n"
            f"  lzma -9     : {res.lzma9_bytes:,} B ({res.lzma9_bpt:.3f} bpt) | NCM2 vs lzma: {ratio_vs_lzma:.2f}x\n"
            f"  Stats       : top1={res.top1_acc*100:.2f}% | topK={res.topk_hit_rate*100:.2f}% | "
            f"oov={res.oov_rate*100:.2f}% | exact={res.exact_match} | time={dt:.2f}s"
        )


if __name__ == "__main__":
    main()
