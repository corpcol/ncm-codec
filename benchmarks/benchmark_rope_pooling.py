#!/usr/bin/env python3
"""Benchmark: RoPE Phase-Aware Key Pooling vs Naive Pooling on Real Qwen 2.5 Keys.

Evaluates:
1. Per-frequency-band norm retention (High frequency i=0 where theta_0=1.0 rad/token vs mid vs low).
2. Overall vector norm retention on real prefill keys from Qwen 2.5.
3. True multi-layer GQA latency profile on GPU (RTX 4050 6GB) with torch.cuda.synchronize() and warmup.
"""

from __future__ import annotations

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ncm.v2 import (
    compute_saliency_weights,
    extract_kv_tensors,
    rope_phase_aware_key_pooling,
    saliency_aware_value_pooling,
)


def benchmark_real_keys_rope_retention():
    print("\n================ ROPE POOLING BENCHMARK (REAL QWEN 2.5 KEYS) ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading real prefill states from {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)

    code_snippet = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    inputs = tokenizer(code_snippet, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past = out.past_key_values

    # Extract Layer 0 Keys: True GQA shape (1, 2, 84, 64)
    k, v = extract_kv_tensors(past, 0)
    B, H, N, D = k.shape
    r = 4
    k = k[:, :, :80, :]  # Chunk into 20 windows of 4 tokens
    M = 20
    inv_freq = model.model.rotary_emb.inv_freq.to(device)

    # 1. Naive mean pooling across chunk of r=4 tokens
    naive_pooled = k.view(B, H, M, r, D).mean(dim=3)

    # 2. Phase-aware harmonic unwrapped pooling
    phase_pooled = rope_phase_aware_key_pooling(k, r=r, inv_freq=inv_freq)

    # Analyze per-frequency channels:
    # High frequency: i=0 (theta_0 = 1.0 rad/token; channels 0 and 32)
    # Mid frequency:  i=8 (channels 8 and 40)
    # Low frequency:  i=31 (channels 31 and 63)
    channel_indices = [
        ("High (i=0, theta=1.000 rad/tok)", 0),
        ("Mid  (i=8, theta=0.031 rad/tok)", 8),
        ("Low  (i=31, theta=0.000001 rad/tok)", 31),
    ]

    print(f"\n{'Frequency Band':<35} | {'Original Norm':<14} | {'Naive Norm':<12} | {'Phase Norm':<12} | {'Amp Delta'}")
    print("-" * 88)

    for label, i in channel_indices:
        ch_orig = torch.stack([k[..., i], k[..., i + D // 2]], dim=-1)
        orig_norm = torch.norm(ch_orig.float().view(B, H, M, r, 2), dim=-1).mean().item()

        ch_naive = torch.stack([naive_pooled[..., i], naive_pooled[..., i + D // 2]], dim=-1)
        naive_norm = torch.norm(ch_naive.float(), dim=-1).mean().item()

        ch_phase = torch.stack([phase_pooled[..., i], phase_pooled[..., i + D // 2]], dim=-1)
        phase_norm = torch.norm(ch_phase.float(), dim=-1).mean().item()

        delta = ((phase_norm - naive_norm) / naive_norm) * 100.0
        print(f"{label:<35} | {orig_norm:<14.4f} | {naive_norm:<12.4f} | {phase_norm:<12.4f} | +{delta:.1f}%")

    # Overall 64-dim norm
    full_orig = torch.norm(k.float().view(B, H, M, r, D), dim=-1).mean().item()
    full_naive = torch.norm(naive_pooled.float(), dim=-1).mean().item()
    full_phase = torch.norm(phase_pooled.float(), dim=-1).mean().item()
    print("-" * 88)
    print(f"{'Overall (All 64 Dimensions)':<35} | {full_orig:<14.4f} | {full_naive:<12.4f} | {full_phase:<12.4f} | +{((full_phase-full_naive)/full_naive)*100.0:.2f}%")
    print("==============================================================================\n")


def benchmark_true_gqa_latency():
    print("================ TRUE GQA MULTI-LAYER LATENCY PROFILE ========================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    dtype = torch.bfloat16
    seq_len = 2048

    print(f"Device: {gpu_name} (Precision: {dtype}, Sequence Length: {seq_len} tokens)")
    print(f"{'Target Model Architecture':<28} | {'KV Heads':<10} | {'Layers':<8} | {'Head Dim':<10} | {'Latency (ms)'}")
    print("-" * 75)

    configs = [
        ("Qwen 2.5-0.5B", 2, 24, 64),
        ("Qwen 2.5-Coder-7B", 4, 28, 128),
    ]

    for name, H_kv, layers, D in configs:
        k = torch.randn(1, H_kv, seq_len, D, device=device, dtype=dtype)
        v = torch.randn(1, H_kv, seq_len, D, device=device, dtype=dtype)

        # Warmup
        for _ in range(5):
            for _ in range(layers):
                sw = compute_saliency_weights(k, r=4)
                _ = rope_phase_aware_key_pooling(k, r=4, weights=sw)
                _ = saliency_aware_value_pooling(v, sw, r=4)
        if device.type == "cuda":
            torch.cuda.synchronize()

        iters = 25
        t0 = time.perf_counter()
        for _ in range(iters):
            for _ in range(layers):
                sw = compute_saliency_weights(k, r=4)
                _ = rope_phase_aware_key_pooling(k, r=4, weights=sw)
                _ = saliency_aware_value_pooling(v, sw, r=4)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) / iters * 1000.0

        print(f"{name:<28} | H_kv={H_kv:<6} | L={layers:<6} | D={D:<8} | {total_time:.2f} ms")

    print("==============================================================================\n")


if __name__ == "__main__":
    benchmark_real_keys_rope_retention()
    benchmark_true_gqa_latency()
