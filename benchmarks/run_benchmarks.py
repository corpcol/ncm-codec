import os
import sys
import json
import time
import lzma
import math
import ast
import torch
import numpy as np
from transformers import AutoTokenizer

os.environ.setdefault('HF_HOME', r'C:\ncm-upstream\hf-home')
sys.path.insert(0, r"..")
from train_stage1_clean import Codec

def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    vocab = np.asarray(ck["vocab"])
    model = Codec(torch.zeros(len(vocab) + 1, ck["state"]["E"].shape[1]), **cfg)
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return model, vocab, cfg

@torch.no_grad()
def escape_decode(model, idx, x):
    zq = model.rvq.decode_idx(idx)
    B = x.size(0)
    out = torch.zeros(B, 0, dtype=torch.long, device=x.device)
    esc = []
    for t in range(model.L):
        ids_in = torch.cat([out, out[:, :1]], 1) if t else torch.zeros(B, 1, dtype=torch.long, device=x.device)
        p = model.decode_from_zq(zq, ids_in)[:, t].argmax(-1)
        wrong = p != x[:, t]
        esc.append(wrong)
        p = torch.where(wrong, x[:, t], p)
        out = torch.cat([out, p[:, None]], 1)
    return out, torch.stack(esc, 1)

def run_domain_test(domain_name, code_text, model, vocab, cfg, hf, test_fn=None):
    dev = next(model.parameters()).device
    enc = hf(code_text, add_special_tokens=False)
    ids = enc["input_ids"]
    n_tokens = len(ids)
    
    m = np.full(200000, len(vocab), dtype=np.int64)
    m[vocab] = np.arange(len(vocab))
    back = np.append(vocab, -1)
    
    eos_c = int(m[hf.eos_token_id])
    L = cfg["L"]
    c = m[np.array(ids)]
    nb = math.ceil(n_tokens / L)
    pad = np.full(nb * L, eos_c, dtype=np.int64)
    pad[:n_tokens] = c
    x = torch.from_numpy(pad).view(nb, L).to(dev)
    
    t0 = time.time()
    idxs, yes, esc = [], [], []
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type in ("cuda", "xpu")):
        for i in range(0, nb, 32):
            xb = x[i:i + 32]
            ib = model.encode(xb)
            idxs.append(ib)
            yb, e = escape_decode(model, ib, xb)
            yes.append(yb)
            esc.append(e)
            
    roundtrip_ids_compact = torch.cat(yes).reshape(-1)[:n_tokens].cpu().numpy()
    recovered_ids = back[roundtrip_ids_compact]
    recovered_text = hf.decode(recovered_ids)
    encode_decode_time = time.time() - t0
    
    # 1. Exact string match
    is_exact_match = (recovered_text == code_text)
    
    # 2. AST parsing validity
    ast_valid = False
    try:
        ast.parse(recovered_text)
        ast_valid = True
    except Exception:
        pass
        
    # 3. Execution / unit test test
    exec_passed = None
    if test_fn:
        try:
            exec_globals = {}
            exec(recovered_text, exec_globals)
            test_fn(exec_globals)
            exec_passed = True
        except Exception as ex:
            exec_passed = False
            print(f"Exec error in {domain_name}: {ex}")
            
    # Compression metrics
    lzma_bytes = len(lzma.compress(code_text.encode("utf-8"), preset=9))
    lzma_bits_tok = (lzma_bytes * 8) / max(1, n_tokens)
    
    escapes_count = int(torch.cat(esc).reshape(-1)[:n_tokens].sum())
    escape_rate = escapes_count / max(1, n_tokens)
    
    # Arithmetic code bits: code bits + entropy of escapes
    cpt = cfg["depth"] / cfg["r"]
    code_bits = cpt * math.log2(cfg["K"])
    if escape_rate > 0 and escape_rate < 1:
        H_esc = -escape_rate * math.log2(escape_rate) - (1 - escape_rate) * math.log2(1 - escape_rate)
    else:
        H_esc = 0.0
    lossless_bits_tok = code_bits + H_esc + escape_rate * math.log2(len(vocab))
    
    return {
        "domain": domain_name,
        "n_tokens": n_tokens,
        "exact_match": is_exact_match,
        "ast_valid": ast_valid,
        "exec_passed": exec_passed,
        "time_s": encode_decode_time,
        "lossless_bits_per_token": lossless_bits_tok,
        "lzma9_bits_per_token": lzma_bits_tok,
        "beats_lzma": lossless_bits_tok < lzma_bits_tok
    }

def main():
    print("[*] Initializing live multi-domain validation...")
    hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    ckpt_4x = r"../weights/codec_4x_best.pt"
    model, vocab, cfg = load_model(ckpt_4x, device="cuda")
    print(f"[+] Loaded 4x Codec: r={cfg['r']}, depth={cfg['depth']}, K={cfg['K']}")
    
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
import subprocess
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
        res = g['scan_directory_tree'](r"..", ".py")
        assert len(res) > 0
        assert any("train_stage1_clean.py" in x[0] for x in res)

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
    for domain, code, test_fn in tests:
        print(f"[*] Testing Domain: {domain}...")
        r = run_domain_test(domain, code, model, vocab, cfg, hf, test_fn)
        results.append(r)
        status = "PASSED" if r["exact_match"] and r["ast_valid"] and r["exec_passed"] else "FAILED"
        print(f"    -> [{status}] Lossless: {r['exact_match']}, Exec: {r['exec_passed']}, Ratio: {r['lossless_bits_per_token']:.2f} bpt (LZMA: {r['lzma9_bits_per_token']:.2f})")

    out_json = r"benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Validation suite complete! Results saved to {out_json}")

if __name__ == "__main__":
    main()
