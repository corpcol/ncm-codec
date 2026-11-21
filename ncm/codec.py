import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EMACodebook(nn.Module):
    def __init__(self, K, d, decay=0.99, eps=1e-5, dead_after=50):
        super().__init__()
        self.K, self.d, self.decay, self.eps, self.dead_after = K, d, decay, eps, dead_after
        self.register_buffer('embed', torch.randn(K, d))
        self.register_buffer('embed_sum', self.embed.clone())
        self.register_buffer('cluster_size', torch.ones(K))
        self.register_buffer('since_used', torch.zeros(K, dtype=torch.long))
        self.register_buffer('initted', torch.tensor(0))

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
    return nn.TransformerEncoderLayer(d, h, 4 * d, drop, activation='gelu', batch_first=True, norm_first=True)

class Codec(nn.Module):
    def __init__(self, emb, L=64, r=4, depth=2, K=4096, d=512, dq=256, heads=8, enc_layers=4, dec_layers=4, drop=0.0, mode='nar'):
        super().__init__()
        assert L % r == 0
        self.cfg = dict(L=L, r=r, depth=depth, K=K, d=d, dq=dq, heads=heads, enc_layers=enc_layers, dec_layers=dec_layers, drop=drop, mode=mode)
        self.L, self.r, self.M, self.mode = L, r, L // r, mode
        V, dt = emb.shape
        self.register_buffer('E', emb.float().clone())
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
        if mode == 'ar':
            self.adapt_tok = nn.Sequential(nn.Linear(dt, d), nn.LayerNorm(d))
            self.bos = nn.Parameter(torch.randn(d) * 0.02)
            T = self.M + L
            m = torch.full((T, T), float('-inf'))
            m[:, :self.M] = 0.0
            m[self.M:, self.M:] = torch.triu(torch.full((L, L), float('-inf')), 1)
            m[:self.M, self.M:] = float('-inf')
            self.register_buffer('ar_mask', m)

    def encode(self, ids):
        h = self.adapt_in(F.embedding(ids, self.E)) + self.pos_enc
        z = self.enc(h).view(ids.size(0), self.M, self.r, -1).mean(2)
        z = self.q_norm(self.pre_q(z))
        zq, idx = self.rvq(z)
        return idx

    def encode_latents(self, ids):
        h = self.adapt_in(F.embedding(ids, self.E)) + self.pos_enc
        z = self.enc(h).view(ids.size(0), self.M, self.r, -1).mean(2)
        z = self.q_norm(self.pre_q(z))
        zq, idx = self.rvq(z)
        return zq, idx

    def decode(self, idx, ids_teacher=None):
        zq = self.rvq.decode_idx(idx)
        lat = self.post_q(zq) + self.pos_lat
        if self.mode == 'nar':
            h = lat.repeat_interleave(self.r, dim=1) + self.pos_dec
            out = self.dec(h)
        else:
            B = idx.size(0)
            if ids_teacher is None:
                tok = torch.zeros(B, 0, self.E.size(1), device=idx.device)
            else:
                tok = F.embedding(ids_teacher, self.E)
            tok_emb = self.adapt_tok(tok)
            prefix = self.bos[None, None, :].expand(B, 1, -1)
            if tok_emb.size(1) > 0:
                tok_in = torch.cat([prefix, tok_emb[:, :-1]], 1)
            else:
                tok_in = prefix
            tok_in = tok_in + self.pos_dec[:, :tok_in.size(1)]
            seq = torch.cat([lat, tok_in], 1)
            mask = self.ar_mask[:seq.size(1), :seq.size(1)]
            out = self.dec(seq, mask=mask)[:, self.M:]
        proj = self.adapt_out(self.out_norm(out))
        return proj @ self.E.t() + self.out_bias

    def decode_from_zq(self, zq, ids_so_far):
        B = zq.size(0)
        lat = self.post_q(zq) + self.pos_lat[None]
        tok = self.adapt_tok(F.embedding(ids_so_far, self.E))
        prefix = self.bos[None, None, :].expand(B, 1, -1)
        if tok.size(1) > 0:
            tok_in = torch.cat([prefix, tok[:, :-1]], 1)
        else:
            tok_in = prefix
        T = tok_in.size(1)
        tok_in = tok_in + self.pos_dec[None, :T]
        seq = torch.cat([lat, tok_in], 1)
        mask = self.ar_mask[:seq.size(1), :seq.size(1)]
        out = self.dec(seq, mask=mask)[:, self.M:]
        proj = self.adapt_out(self.out_norm(out))
        return proj @ self.E.t() + self.out_bias
