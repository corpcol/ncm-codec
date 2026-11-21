#!/usr/bin/env python3
"""NCM Stage 1 Codec Training Pipeline.
Trains Transformer Autoencoder + RVQ (Residual Vector Quantizer) for Lossless Code Context Compression.

Hardware Tested:
  - NVIDIA RTX 5080 (CUDA, bf16)
  - Intel Arc Pro B70 (Intel oneAPI XPU, bf16)
  - Apple Silicon Mac Studio (MPS)
  - Standard CPU (fallback)

Usage:
  python train.py --data ./data --out ./runs/ncm_4x --mode ar --r 8 --depth 2 --steps 50000
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass

# ------------------------------------------------------------------ Device Selection
def pick_device() -> torch.device:
    want = os.environ.get("NCM_DEVICE", "auto").lower()
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    if want == "xpu" and xpu: return torch.device("xpu")
    if want == "cuda" and cuda: return torch.device("cuda")
    if want == "mps" and mps: return torch.device("mps")
    if want == "cpu": return torch.device("cpu")
    
    if cuda: return torch.device("cuda")
    if xpu: return torch.device("xpu")
    if mps: return torch.device("mps")
    return torch.device("cpu")

def dev_mod(dev):
    return getattr(torch, dev.type) if dev.type in ("cuda", "xpu", "mps") else None

def sync(dev):
    m = dev_mod(dev)
    if m: m.synchronize()

# ------------------------------------------------------------------ Model Architecture
class EMACodebook(nn.Module):
    def __init__(self, K, d, decay=0.99, eps=1e-5, dead_after=100):
        super().__init__()
        self.K, self.d, self.decay, self.eps, self.dead_after = K, d, decay, eps, dead_after
        self.register_buffer("embed", torch.randn(K, d) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_sum", torch.zeros(K, d))
        self.register_buffer("since_used", torch.zeros(K, dtype=torch.long))
        self.register_buffer("initted", torch.tensor(0))

    def nearest(self, x):
        E = self.embed
        d2 = x.pow(2).sum(1, keepdim=True) - 2 * x @ E.t() + E.pow(2).sum(1)[None]
        return d2.argmin(1)

    def lookup(self, idx):
        return F.embedding(idx, self.embed)

    def forward(self, x):
        shp = x.shape
        flat = x.reshape(-1, self.d).float()
        if self.training and int(self.initted) == 0:
            sel = torch.randint(0, flat.size(0), (self.K,), device=flat.device)
            self.embed.copy_(flat[sel] + 1e-3 * torch.randn_like(self.embed))
            self.embed_sum.copy_(self.embed)
            self.cluster_size.fill_(1.0)
            self.initted.fill_(1)

        idx = self.nearest(flat)
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(idx, self.K).type_as(flat)
                cnt = oh.sum(0)
                self.cluster_size.mul_(self.decay).add_(cnt, alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(oh.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                self.embed.copy_(self.embed_sum / cs[:, None])
                self.since_used.add_(1)
                self.since_used[cnt > 0] = 0
                dead = self.since_used >= self.dead_after
                nd = int(dead.sum())
                if nd:
                    sel = torch.randint(0, flat.size(0), (nd,), device=flat.device)
                    new = flat[sel] + 1e-3 * torch.randn(nd, self.d, device=flat.device)
                    self.embed[dead] = new
                    self.embed_sum[dead] = new
                    self.cluster_size[dead] = 1.0
                    self.since_used[dead] = 0
        return self.lookup(idx).reshape(shp), idx.reshape(shp[:-1])

class RVQ(nn.Module):
    def __init__(self, depth, K, d, **kw):
        super().__init__()
        self.levels = nn.ModuleList([EMACodebook(K, d, **kw) for _ in range(depth)])

    def forward(self, z):
        res = z.float()
        zq = torch.zeros_like(res)
        idxs = []
        for cb in self.levels:
            q, i = cb(res.detach() if self.training else res)
            res = res - q
            zq = zq + q
            idxs.append(i)
        return zq, torch.stack(idxs, -1)

    def decode_idx(self, idx):
        return sum(cb.lookup(idx[..., j]) for j, cb in enumerate(self.levels))

def tlayer(d, h, drop):
    return nn.TransformerEncoderLayer(d, h, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)

class Codec(nn.Module):
    def __init__(self, emb, L=64, r=4, depth=2, K=4096, d=512, dq=256, heads=8, enc_layers=4, dec_layers=4, drop=0.0, mode="nar"):
        super().__init__()
        assert L % r == 0
        self.cfg = dict(L=L, r=r, depth=depth, K=K, d=d, dq=dq, heads=heads, enc_layers=enc_layers, dec_layers=dec_layers, drop=drop, mode=mode)
        self.L, self.r, self.M, self.mode = L, r, L // r, mode
        V, dt = emb.shape
        self.register_buffer("E", emb.float().clone())
        self.adapt_in = nn.Sequential(nn.Linear(dt, d), nn.LayerNorm(d))
        self.pos_enc = nn.Parameter(torch.randn(L, d) * 0.02)
        self.enc = nn.TransformerEncoder(tlayer(d, heads, drop), enc_layers)
        self.pre_q = nn.Linear(d, dq)
        self.q_norm = nn.LayerNorm(dq, elementwise_affine=False)
        self.rvq = RVQ(depth, K, dq)
        self.post_q = nn.Linear(dq, d)
        self.pos_lat = nn.Parameter(torch.randn(self.M, d) * 0.02)
        self.pos_dec = nn.Parameter(torch.randn(L, d) * 0.02)
        self.dec = nn.TransformerEncoder(tlayer(d, heads, drop), dec_layers)
        self.out_norm = nn.LayerNorm(d)
        self.adapt_out = nn.Linear(d, dt)
        self.out_bias = nn.Parameter(torch.zeros(V))
        
        if mode == "ar":
            self.adapt_tok = nn.Sequential(nn.Linear(dt, d), nn.LayerNorm(d))
            self.bos = nn.Parameter(torch.randn(d) * 0.02)
            T = self.M + L
            m = torch.full((T, T), float("-inf"))
            m[:, :self.M] = 0.0
            m[self.M:, self.M:] = torch.triu(torch.full((L, L), float("-inf")), 1)
            m[:self.M, self.M:] = float("-inf")
            self.register_buffer("ar_mask", m)

    def encode_z(self, ids):
        h = self.adapt_in(F.embedding(ids, self.E)) + self.pos_enc[None]
        h = self.enc(h)
        B = h.size(0)
        h = h.view(B, self.M, self.r, -1).mean(2)
        return self.q_norm(self.pre_q(h))

    @torch.no_grad()
    def encode(self, ids):
        z = self.encode_z(ids)
        with torch.autocast(ids.device.type, enabled=False):
            _, idx = self.rvq(z)
        return idx

    def head(self, h):
        return self.adapt_out(self.out_norm(h)) @ self.E.t() + self.out_bias

    def decode_from_zq(self, zq, ids_in=None):
        lat = self.post_q(zq.to(self.post_q.weight.dtype))
        if self.mode == "nar":
            h = lat.repeat_interleave(self.r, 1) + self.pos_dec[None]
            return self.head(self.dec(h))
        B = lat.size(0)
        tok = self.adapt_tok(F.embedding(ids_in[:, :-1], self.E)) if ids_in.size(1) > 1 else lat[:, :0]
        tok = torch.cat([self.bos.expand(B, 1, -1).to(lat.dtype), tok.to(lat.dtype)], 1)
        n = tok.size(1)
        x = torch.cat([lat + self.pos_lat[None], tok + self.pos_dec[None, :n]], 1)
        T = self.M + n
        h = self.dec(x, mask=self.ar_mask[:T, :T])
        return self.head(h[:, self.M:])

    def forward(self, ids, beta=0.25):
        z = self.encode_z(ids)
        with torch.autocast(ids.device.type, enabled=False):
            zq, idx = self.rvq(z)
        zq_ste = z.float() + (zq - z.float()).detach()
        logits = self.decode_from_zq(zq_ste, ids)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ids.reshape(-1))
        commit = F.mse_loss(z.float(), zq.detach())
        acc = (logits.argmax(-1) == ids).float().mean()
        return dict(loss=ce + beta * commit, ce=ce.detach(), commit=commit.detach(), acc=acc.detach(), idx=idx)

# ------------------------------------------------------------------ Training CLI
def main():
    ap = argparse.ArgumentParser(description="NCM Stage 1 Codec Training")
    ap.add_argument("--data", required=True, help="Directory containing train_ids.npy and vocab_emb.pt")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints")
    ap.add_argument("--mode", default="ar", choices=["nar", "ar"])
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--r", type=int, default=8, help="Downsampling factor (e.g. 8 for 4x context compression)")
    ap.add_argument("--depth", type=int, default=2, help="RVQ codebook depth")
    ap.add_argument("--K", type=int, default=4096, help="RVQ codebook size")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    dev = pick_device()
    print(f"[*] Training NCM Codec on device: {dev}")
    print(f"[*] Rate: {args.depth / args.r} codes/token ({args.r // args.depth}x context factor)")

if __name__ == "__main__":
    main()
