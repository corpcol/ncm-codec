# NCM: Neural Codec for Code (Lossless Context Compression)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**NCM (Neural Context Model)** is a lightweight (26.9M parameter) neural tokenizer and context compressor designed to expand the effective context window of Large Language Models. 

By combining **Transformer Autoencoders**, **Residual Vector Quantization (RVQ)**, and **Causal Arithmetic Escape Decoding**, NCM compresses raw token streams by up to **4x (0.25 codes/token)** while guaranteeing **100% bit-for-bit exact lossless recovery**, outperforming classical industry-standard compression (`lzma -9`) by **18%**.

---

## 🚀 Key Benchmarks

Evaluated across the standard 12-file heldout benchmark suite (91,434 tokens) from the Python Standard Library:

| Compression Engine | Codes / Token | Bits / Token | 12-File Lossless Recovery | Data Reduction vs Raw |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Qwen 2.5 Tokens** | 1.000 | 17.21 bpt | Baseline (Raw) | 1.00x |
| **Classical Industry Standard (`lzma -9`)** | N/A | 7.92 bpt | 100% Exact | 2.17x |
| **NCM 2x Baseline ($r=4, \text{depth}=2$)** | 0.500 | 7.90 bpt | **100% Exact (12/12)** | **2.18x** |
| **NCM 4x Advanced ($r=8, \text{depth}=2$)** | **0.250** | **6.495 bpt** | **100% Exact (12/12)** | **2.65x (18% better than LZMA)** |

### Multi-Domain Functional Verification
All decompressed code was tested via AST compilation (`ast.parse`) and dynamic isolated runtime execution (`exec()`):

| Domain | Payload Type | Bit-for-Bit Exact | Runtime Tests Passed | NCM 4x Ratio | `lzma -9` Ratio | Advantage vs LZMA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Algorithms** | LRU Cache & Quicksort | **Yes** | **Passed** | **7.60 bpt** | 14.19 bpt | **46% smaller** |
| **Web Services** | URL Query & JSON Parser | **Yes** | **Passed** | **10.21 bpt** | 21.41 bpt | **52% smaller** |
| **Systems / OS** | Directory Walker & Archiver | **Yes** | **Passed** | **9.26 bpt** | 22.02 bpt | **58% smaller** |
| **Mathematics** | Running Stats & Matrix Mult | **Yes** | **Passed** | **7.83 bpt** | 10.67 bpt | **26% smaller** |

---

## 🧠 Architectural Overview

```
Raw Token Stream (x1, x2, ..., xN)
       │
       ▼ [Frozen Qwen 2.5 Embeddings]
       │
┌──────────────┐
│  Transformer │  Encodes L=64 tokens into M = L/r continuous latents
│   Encoder    │  (e.g., r=8 maps 8 tokens to 1 latent vector)
└──────┬───────┘
       ▼
┌──────────────┐
│ Residual VQ  │  Quantizes into depth=2 discrete codebook indices
│ (K=4096, d=2)│  Level 1: Macro semantics (syntax, control flow)
└──────┬───────┘  Level 2: Micro residual (identifiers, literals)
       ▼
┌──────────────┐
│  Causal AR   │  Autoregressive decoder predicts original tokens.
│   Decoder    │  Any prediction mismatch triggers an exact escape.
└──────┬───────┘
       ▼
100% Exact Reconstructed Source Code
```

---

## 📦 Quick Start

### Installation
```bash
git clone https://github.com/corpcol/ncm-codec.git
cd ncm-codec
pip install -e .
```

### Quick Benchmark
```bash
python -m ncm.cli bench path/to/script.py --ckpt weights/codec_4x_best.pt
```

### Python API
```python
from ncm import NCMCompressor

compressor = NCMCompressor("weights/codec_4x_best.pt")
code = open("my_project.py").read()

result = compressor.compress_text(code)
print(f"Compressed to: {result['bits_per_token']:.2f} bits/token")
print(f"Exact lossless match: {result['exact_match']}")
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
