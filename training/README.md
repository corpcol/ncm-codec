# NCM Training Guide

This directory contains the production training pipeline for **NCM (Neural Codec for Code)** Stage 1 autoencoders.

## Key Upgrades
* **Chunked Cross-Entropy (`chunked_cross_entropy`):** Slices 151k-vocab logits computations into manageable chunks, cutting peak VRAM by 4-8x to comfortably train on 6GB-8GB GPUs.
* **True 32-bit Integer Arithmetic Coder:** Lossless neural probability distribution coding replacing raw escape sequences.
* **Safe Positional Scaling (`pos_scale=0.1`):** Ensures token signal magnitude dominates positional embeddings with strict `tok_rms / pos_rms >= 5.0` diagnostic.
* **O(L) KV-Cached AR Inference:** Matches parallel autoregressive forward pass bit-for-bit with `< 1e-4` precision.
* **Instantaneous Codebook Utilization & Shannon Perplexity:** Tracks live dead-code reset and codebook health during training.
* **Built-in `--self-test`:** Instant verification of KV-cache exactness, norm ratios, and arithmetic coder roundtrip.

## Requirements
* Python 3.10+
* PyTorch 2.0+ (CUDA, Intel oneAPI XPU, Apple MPS, or CPU)
* `transformers`

## Verification & Self-Test
Run the built-in self-test to verify mathematical correctness before launching full training:
```bash
python training/train.py --self-test
```

## Quick Start

### 1. Training on NVIDIA GPU (CUDA)
To train the 4x context compression regime ($r=8, \text{depth}=2, K=4096$) on local code:
```bash
python training/train.py \
    --data-dir ./data \
    --output ./weights/codec_4x_best.pt \
    --mode ar \
    --r 8 \
    --depth 2 \
    --K 4096 \
    --batch-size 8 \
    --grad-accum 4 \
    --steps 50000
```

Or streaming directly from a HuggingFace dataset:
```bash
python training/train.py \
    --hf-dataset bigcode/the-stack-smol \
    --text-column content \
    --output ./weights/codec_4x_best.pt \
    --mode ar \
    --r 8 \
    --depth 2 \
    --steps 50000
```

### 2. Training on Intel Arc GPU (Intel oneAPI XPU)
```bash
python training/train.py \
    --data-dir ./data \
    --output ./weights/codec_4x_best.pt \
    --device xpu \
    --mode ar \
    --r 8 \
    --depth 2
```

### 3. Training on Apple Silicon (MPS)
```bash
python training/train.py \
    --data-dir ./data \
    --output ./weights/codec_4x_best.pt \
    --device mps \
    --mode ar \
    --r 8 \
    --depth 2
```
