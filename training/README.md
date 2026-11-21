# NCM Training Guide

This directory contains the training pipeline for **NCM (Neural Codec for Code)** Stage 1 autoencoders.

## Requirements
* Python 3.10+
* PyTorch 2.0+ (CUDA, Intel oneAPI XPU, Apple MPS, or CPU)
* `transformers`

## Quick Start

### 1. Training on NVIDIA GPU (CUDA)
To train the 4x context compression regime ($r=8, \text{depth}=2, K=4096$):
```bash
python training/train.py \
    --data ./data \
    --out ./runs/ncm_4x_ar \
    --mode ar \
    --r 8 \
    --depth 2 \
    --K 4096 \
    --steps 50000 \
    --bs 64
```

### 2. Training on Intel Arc GPU (Intel oneAPI XPU)
```bash
NCM_DEVICE=xpu python training/train.py \
    --data ./data \
    --out ./runs/ncm_8x_nar \
    --mode nar \
    --r 16 \
    --depth 2 \
    --K 4096 \
    --steps 50000 \
    --bs 64
```

### 3. Training on Apple Silicon (MPS)
```bash
NCM_DEVICE=mps python training/train.py \
    --data ./data \
    --out ./runs/ncm_2x_ar \
    --mode ar \
    --r 4 \
    --depth 2 \
    --steps 25000
```
