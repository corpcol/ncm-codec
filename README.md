# NCM: Neural Codec for Code & Generative Context Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**NCM (Neural Context Model)** is an open-source neural tokenizer, binary context compressor, and generative KV-cache optimization framework designed to scale LLM context capacity.

NCM bridges the gap between neural discrete tokenization and generative inference:
* **Stage 1 (Binary Codec):** Transformer Autoencoders + Residual Vector Quantization (RVQ) + Causal Escape Arithmetic. Compresses code into a real packed `.ncm` binary bitstream at **4x token reduction (0.25 codes/token)** with **100% exact bit-for-bit lossless recovery**, outperforming `lzma -9` by **18%**.
* **Stage 2 (Generative NCM V2):** In-Context Autoencoder & KV-Cache Injection. Attaches to pretrained LLMs (e.g., Qwen 2.5, Llama 3) to compress prompt context by 4x, slashing active GPU KV-cache memory by **75%** while enabling generative inference directly over compressed states.

---

## 🚀 Benchmark Highlights

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
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Algorithms** | LRU Cache & Quicksort | **Yes** | **Passed** | **7.60 bpt** | 14.19 bpt | **46% smaller** |
| **Web Services** | URL Query & JSON Parser | **Yes** | **Passed** | **10.21 bpt** | 21.41 bpt | **52% smaller** |
| **Systems / OS** | Directory Walker & Archiver | **Yes** | **Passed** | **9.26 bpt** | 22.02 bpt | **58% smaller** |
| **Mathematics** | Running Stats & Matrix Mult | **Yes** | **Passed** | **7.83 bpt** | 10.67 bpt | **26% smaller** |

---

## 🧠 Architectural Overview

```
                                  [STAGE 1: BINARY CODEC]
Raw Token Stream (x1, x2, ..., xN)
       │
       ▼ [Frozen Qwen 2.5 Embeddings]
┌──────────────┐
│  Transformer │  Encodes L=64 tokens into M = L/r continuous latents
│   Encoder    │  (r=8 maps 8 tokens to 1 latent vector)
└──────┬───────┘
       ▼
┌──────────────┐
│ Residual VQ  │  Quantizes into depth=2 discrete codebook indices (K=4096)
│ (K=4096, d=2)│  Level 1: Macro semantics (syntax, control flow)
└──────┬───────┘  Level 2: Micro residual (identifiers, literals)
       ▼
┌──────────────┐
│  Causal AR   │  Autoregressive decoder + deterministic escape bitstream.
│   Decoder    │  Packed into .ncm container bitstream with 12-bit code alignment.
└──────┬───────┘
       ▼
100% Exact Reconstructed Source Code (Verified Cryptographic SHA-256)

                                  [STAGE 2: GENERATIVE NCM V2]
Raw Prompt Context (e.g., 32k Tokens)
       │
       ▼ [NCM Context Encoder]
Compressed Latent Memory (8k Slots — 4x Reduction)
       │
       ▼ [NCM KV-Projector]
Injected into Pretrained LLM Attention Keys & Values (75% VRAM Reduction)
       │
       ▼
LLM Generates & Reasons Directly Over 4x Compressed KV Cache!
```

---

## 📦 Quick Start & CLI Usage

### 1. Installation
```bash
git clone https://github.com/corpcol/ncm-codec.git
cd ncm-codec
pip install -e .
```

### 2. Compress a File into a Binary `.ncm` Bitstream
```bash
python -m ncm.cli encode my_script.py -o my_script.ncm
```

### 3. Decompress `.ncm` Back to Bit-for-Bit Exact Source
```bash
python -m ncm.cli decode my_script.ncm -o restored.py
```

### 4. Benchmark Compression & Lossless Fidelity
```bash
python -m ncm.cli bench my_script.py
```

### 5. Generative Inference over Compressed Context (NCM V2)
```python
from ncm.v2 import NCMGenerativeModel

# Wrap pretrained Qwen 2.5 with 4x context compression
model = NCMGenerativeModel("Qwen/Qwen2.5-0.5B", compression_factor=4)

context_code = open("large_file.py").read()
query = "What classes are defined in this module?"

# Ingests context into 75% compressed KV cache and generates response
result = model.generate_with_compressed_context(context_code, query)
print(result["response"])
print(f"KV Cache Savings: {result['kv_memory_savings']}")
```

---

## 🏋️ Training Pipeline
The full, multi-accelerator training pipeline is located under `training/`:
```bash
# Train 4x compression regime on CUDA
python training/train.py --data ./data --out ./runs/4x --mode ar --r 8 --depth 2 --steps 50000

# Train on Intel Arc Pro (oneAPI XPU)
NCM_DEVICE=xpu python training/train.py --data ./data --out ./runs/8x --mode nar --r 16
```
See [training/README.md](training/README.md) for details.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
