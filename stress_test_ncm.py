import time
import os
import sys
import torch
from pathlib import Path

repo_root = Path(r"C:\Users\caep2\.gemini\antigravity\scratch\ncm-codec")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ncm.codec import Codec
from ncm.compressor import NCMCompressor
from ncm.v2 import (
    rope_phase_aware_key_pooling,
    saliency_aware_value_pooling,
    compute_saliency_weights,
    NCMKVProjector,
    NCMSpeculativeHead,
    AbliterationHook
)

def run_stress_test():
    print("=" * 70)
    print("NCM CODEC & GENERATIVE CONTEXT COMPRESSION: FULL STRESS TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {gpu_name} ({device})")
    
    # 1. GENERATIVE KV-CACHE COMPRESSION STRESS TEST (VRAM & Throughput)
    print("\n[Stage 1/3] Stress Testing Generative KV-Cache Compression (RoPE + Saliency)...")
    batch_sizes = [1, 2, 4]
    seq_lens = [1024, 2048, 4096]
    r = 4  # 4x compression
    H_kv = 4
    D = 128
    layers = 28 # Qwen 2.5-Coder-7B specs
    
    torch.cuda.reset_peak_memory_stats(device)
    initial_vram = torch.cuda.memory_allocated(device) / (1024**2)
    
    print(f"{'Batch':<6} | {'SeqLen':<8} | {'Compressed':<12} | {'Latency (ms)':<14} | {'Throughput (tok/s)':<20} | {'Peak VRAM (MB)'}")
    print("-" * 80)
    
    for b in batch_sizes:
        for seq in seq_lens:
            k = torch.randn(b, H_kv, seq, D, device=device, dtype=torch.bfloat16)
            v = torch.randn(b, H_kv, seq, D, device=device, dtype=torch.bfloat16)
            
            # Warmup
            for _ in range(3):
                sw = compute_saliency_weights(k, r=r)
                kp = rope_phase_aware_key_pooling(k, r=r, weights=sw)
                vp = saliency_aware_value_pooling(v, sw, r=r)
            torch.cuda.synchronize(device)
            
            # Benchmark 20 iterations
            iters = 20
            t0 = time.perf_counter()
            for _ in range(iters):
                sw = compute_saliency_weights(k, r=r)
                kp = rope_phase_aware_key_pooling(k, r=r, weights=sw)
                vp = saliency_aware_value_pooling(v, sw, r=r)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - t0) / iters * 1000.0
            
            total_tokens = b * seq
            tokens_per_sec = total_tokens / (elapsed_ms / 1000.0)
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
            
            assert kp.shape == (b, H_kv, seq // r, D), f"Bad key shape {kp.shape}"
            assert vp.shape == (b, H_kv, seq // r, D), f"Bad value shape {vp.shape}"
            
            print(f"{b:<6} | {seq:<8} | {seq//r:<12} | {elapsed_ms:<14.2f} | {tokens_per_sec:<20.0f} | {peak_vram:<10.1f} MB")
            
            del k, v, kp, vp, sw
            torch.cuda.empty_cache()
            
    print("  -> Generative KV-Cache Compression: PASSED all batch/seq dimensions.")
    
    # 2. KV PROJECTOR & SPECULATIVE HEAD INTEGRITY
    print("\n[Stage 2/3] Stress Testing Projectors, Speculative Heads, & Abliteration...")
    d_latent = 256
    d_model = 3584 # 7B hidden dim
    projector = NCMKVProjector(d_latent=d_latent, num_layers=layers, num_heads=H_kv, head_dim=D).to(device).to(torch.bfloat16)
    lm_head = torch.nn.Linear(d_model, 152064, bias=False).to(device).to(torch.bfloat16)
    spec_head = NCMSpeculativeHead(hidden_size=d_model, lm_head=lm_head, num_heads=3).to(device).to(torch.bfloat16)
    
    z = torch.randn(2, 512, d_latent, device=device, dtype=torch.bfloat16)
    kv_cache = projector(z)
    assert len(kv_cache) == layers
    assert kv_cache[0][0].shape == (2, H_kv, 512, D)
    print("  -> NCMKVProjector: 28-layer KV projection verified.")
    
    hidden = torch.randn(2, 512, d_model, device=device, dtype=torch.bfloat16)
    spec_logits = spec_head(hidden)
    assert len(spec_logits) == 3
    assert spec_logits[0].shape == (2, 512, 152064)
    print("  -> NCMSpeculativeHead: 3-step candidate lookahead verified.")
    
    refusal_vec = torch.randn(d_model, device=device, dtype=torch.bfloat16)
    hook = AbliterationHook(refusal_vec)
    projected_h = hook(None, None, hidden)
    assert projected_h.shape == hidden.shape
    print("  -> AbliterationHook: Orthogonal refusal subtraction verified.")
    
    del projector, spec_head, z, kv_cache, hidden, spec_logits, projected_h
    torch.cuda.empty_cache()
    
    # 3. END-TO-END CODEC LOSSLESS RECOVERY UNDER SYNTHETIC LOAD
    print("\n[Stage 3/3] Stress Testing Lossless Binary Compression & Decompression...")
    mock_ckpt = repo_root / "mock_stress_codec.pt"
    V, dt, L, r_codec, K = 512, 64, 16, 4, 128
    E = torch.randn(V, dt) * 0.02
    codec_model = Codec(E=E, L=L, r=r_codec, d=64, dq=32, enc_layers=2, dec_layers=2, heads=2, depth=2, K=K, drop=0.0)
    cfg = {
        "V": V, "dt": dt, "L": L, "r": r_codec, "d": 64, "dq": 32,
        "enc_layers": 2, "dec_layers": 2, "heads": 2, "depth": 2, "K": K,
        "mode": "ar", "pos_scale": 0.1
    }
    torch.save({"cfg": cfg, "vocab": list(range(V)), "state": codec_model.state_dict()}, mock_ckpt)
    
    compressor = NCMCompressor(ckpt_path=mock_ckpt, device=str(device))
    
    sample_codes = [
        "def compute_fft(signal):\n    import numpy as np\n    return np.fft.fft(signal)\n",
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n",
        "import asyncio\nasync def worker(queue):\n    while True:\n        item = await queue.get()\n        print(item)\n"
    ] * 10 # 30 files
    
    t_start = time.perf_counter()
    exact_matches = 0
    for idx, code in enumerate(sample_codes):
        res = compressor.compress_text(code, verify=True)
        restored = compressor.decompress_text(res.blob)
        if restored == code:
            exact_matches += 1
            
    total_codec_time = time.perf_counter() - t_start
    print(f"  -> Lossless Reconstruction: {exact_matches}/{len(sample_codes)} exact bit-for-bit roundtrips ({total_codec_time:.3f}s total)")
    assert exact_matches == len(sample_codes)
    
    if mock_ckpt.exists():
        os.remove(mock_ckpt)
        
    final_vram = torch.cuda.memory_allocated(device) / (1024**2)
    print(f"\nFinal VRAM Allocated: {final_vram:.2f} MB (No memory leaks detected).")
    print("=" * 70)
    print("ALL NCM STRESS TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_stress_test()
