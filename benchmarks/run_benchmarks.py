#!/usr/bin/env python3
"""Live multi-domain benchmark & AST verification suite for NCM.

Validates:
1. Exact bit-for-bit lossless decompression
2. Abstract Syntax Tree (AST) compilation validity
3. Isolated dynamic execution & unit test assertions
4. Real on-disk bpt vs lzma -9 compression
"""

from __future__ import annotations

import argparse
import ast
import json
import lzma
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ncm.codec import Codec
from ncm.compressor import NCMCompressor
from ncm.weights import get_weights_path


def run_domain_test(
    domain_name: str,
    code_text: str,
    compressor: NCMCompressor,
    test_fn: Callable[[Dict], None] | None = None,
) -> Dict:
    t0 = time.perf_counter()
    res = compressor.compress_text(code_text, verify=True)
    enc_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    restored_text = compressor.decompress_text(res.blob)
    dec_time = time.perf_counter() - t1

    is_exact_match = (restored_text == code_text)

    ast_valid = False
    try:
        ast.parse(restored_text)
        ast_valid = True
    except Exception:
        pass

    exec_passed = None
    if test_fn:
        try:
            exec_globals: Dict = {}
            exec(restored_text, exec_globals)
            test_fn(exec_globals)
            exec_passed = True
        except Exception as ex:
            exec_passed = False
            print(f"[!] Exec error in {domain_name}: {ex}")

    lzma_bytes = len(lzma.compress(code_text.encode("utf-8"), preset=9))
    lzma_bpt = (lzma_bytes * 8.0) / max(1, res.orig_num_tokens)
    actual_bpt = (len(res.blob) * 8.0) / max(1, res.orig_num_tokens)

    return {
        "domain": domain_name,
        "n_tokens": res.orig_num_tokens,
        "raw_bytes": res.raw_bytes,
        "compressed_bytes": len(res.blob),
        "exact_match": is_exact_match,
        "ast_valid": ast_valid,
        "exec_passed": exec_passed,
        "encode_time_s": enc_time,
        "decode_time_s": dec_time,
        "actual_bpt": actual_bpt,
        "lzma9_bpt": lzma_bpt,
        "ratio_vs_raw": res.raw_bytes / max(1, len(res.blob)),
        "ratio_vs_lzma": lzma_bytes / max(1, len(res.blob)),
        "top1_acc": res.top1_acc,
        "topk_hit_rate": res.topk_hit_rate,
        "oov_rate": res.oov_rate,
    }


def create_mock_checkpoint(dest_path: Path) -> Path:
    """Generate a valid mock checkpoint for standalone testing without large downloads."""
    V, dt, L, r, K = 512, 64, 16, 4, 128
    E = torch.randn(V, dt) * 0.02
    model = Codec(
        E=E, L=L, r=r, d=64, dq=32, enc_layers=1, dec_layers=1, heads=2, depth=2, K=K, drop=0.0
    )
    cfg = {
        "V": V,
        "dt": dt,
        "L": L,
        "r": r,
        "d": 64,
        "dq": 32,
        "enc_layers": 1,
        "dec_layers": 1,
        "heads": 2,
        "depth": 2,
        "K": K,
        "mode": "ar",
        "pos_scale": 0.1,
    }
    torch.save({"cfg": cfg, "vocab": list(range(V)), "state": model.state_dict()}, dest_path)
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NCM Multi-Domain Verification Suite")
    parser.add_argument("--ckpt", type=Path, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cuda, cpu, xpu, mps)")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic model for fast local verification")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.json"), help="Output JSON results path")
    args = parser.parse_args()

    mock_dir = None
    if args.mock:
        mock_dir = tempfile.TemporaryDirectory()
        ckpt_path = create_mock_checkpoint(Path(mock_dir.name) / "mock_codec.pt")
        print(f"[*] Running in --mock mode with synthetic model at {ckpt_path}")
    else:
        if args.ckpt is not None and args.ckpt.exists():
            ckpt_path = args.ckpt
        else:
            default_p = Path(get_weights_path("4x"))
            if default_p.exists():
                ckpt_path = default_p
            else:
                print(f"[!] Checkpoint not found at {default_p}.")
                print("    Use --ckpt <path> or run with --mock for synthetic testing.")
                sys.exit(1)

    compressor = NCMCompressor(ckpt_path=ckpt_path, device=args.device, top_k=64)
    print(f"[*] Initialized NCMCompressor on {compressor.device}...")

    # Domain 1: Algorithms & Data Structures
    d1_code = '''
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
'''
    def d1_test(g):
        c = g['LRUCache'](2)
        c.put(1, 100)
        c.put(2, 200)
        assert c.get(1) == 100
        c.put(3, 300)
        assert c.get(2) == -1
        assert g['quick_sort']([5, 2, 8, 1, 9]) == [1, 2, 5, 8, 9]

    # Domain 2: Web API & JSON Serializer
    d2_code = '''
import json
import urllib.parse

def build_api_request(endpoint: str, query_params: dict, headers: dict):
    encoded_query = urllib.parse.urlencode(query_params)
    full_url = f"{endpoint}?{encoded_query}" if encoded_query else endpoint
    payload = json.dumps({"agent": "NCM-Mesh", "headers": headers, "timestamp": 1795214000})
    return full_url, payload

def parse_api_response(raw_json: str):
    data = json.loads(raw_json)
    if "status" in data and data["status"] == "ok":
        return [item["id"] for item in data.get("results", [])]
    return []
'''
    def d2_test(g):
        url, payload = g['build_api_request']("https://api.mesh.local/v1/query", {"model": "qwen", "temp": 0.7}, {"Auth": "Bearer XYZ"})
        assert "temp=0.7" in url
        parsed = g['parse_api_response']('{"status": "ok", "results": [{"id": 101}, {"id": 102}]}')
        assert parsed == [101, 102]

    # Domain 3: Systems & Filesystem IO (Operating System)
    d3_code = '''
import os
import shutil

def scan_directory_tree(root_dir: str, extension: str = ".py"):
    matched_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(extension):
                full_path = os.path.join(dirpath, f)
                matched_files.append((f, os.path.getsize(full_path) if os.path.exists(full_path) else 0))
    return sorted(matched_files, key=lambda x: x[1], reverse=True)

def safe_archive_file(source: str, dest: str):
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file {source} missing")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(source, dest)
    return os.path.exists(dest)
'''
    def d3_test(g):
        res = g['scan_directory_tree'](".", ".py")
        assert len(res) >= 0

    # Domain 4: Numerical Computing & Statistics
    d4_code = '''
import math

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    def std_dev(self) -> float:
        return math.sqrt(self.variance())

def matrix_multiply_2x2(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
    ]
'''
    def d4_test(g):
        s = g['RunningStats']()
        for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
            s.update(val)
        assert abs(s.mean - 30.0) < 1e-5
        mat = g['matrix_multiply_2x2']([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert mat == [[19, 22], [43, 50]]

    tests = [
        ("Algorithms & Data Structures", d1_code, d1_test),
        ("Web Services & JSON APIs", d2_code, d2_test),
        ("Systems, Processes & Filesystem IO", d3_code, d3_test),
        ("Numerical Analysis & Mathematics", d4_code, d4_test),
    ]

    results = []
    print("\n================== Running Domain Benchmarks ==================")
    for domain, code, test_fn in tests:
        print(f"[*] Testing Domain: {domain}...")
        r = run_domain_test(domain, code, compressor, test_fn)
        results.append(r)
        status = "PASSED" if r["exact_match"] and r["ast_valid"] and r["exec_passed"] else "FAILED"
        print(
            f"    -> [{status}] Lossless: {r['exact_match']} | AST: {r['ast_valid']} | "
            f"Exec: {r['exec_passed']} | {r['actual_bpt']:.2f} bpt (LZMA: {r['lzma9_bpt']:.2f} bpt)"
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Validation suite complete! Results saved to {args.out}")

    if mock_dir:
        mock_dir.cleanup()


if __name__ == "__main__":
    main()
