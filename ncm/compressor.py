import os
import math
import time
import numpy as np
import torch
from .codec import Codec

class NCMCompressor:
    def __init__(self, ckpt_path: str, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.vocab = np.asarray(ck["vocab"])
        self.model = Codec(torch.zeros(len(self.vocab) + 1, ck["state"]["E"].shape[1]), **self.cfg)
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()

        from transformers import AutoTokenizer
        self.hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.m = np.full(200000, len(self.vocab), dtype=np.int64)
        self.m[self.vocab] = np.arange(len(self.vocab))
        self.back = np.append(self.vocab, -1)
        self.eos_c = int(self.m[self.hf.eos_token_id])

    @torch.no_grad()
    def _escape_decode(self, idx, x):
        zq = self.model.rvq.decode_idx(idx)
        B = x.size(0)
        out = torch.zeros(B, 0, dtype=torch.long, device=x.device)
        esc = []
        for t in range(self.model.L):
            ids_in = torch.cat([out, out[:, :1]], 1) if t else torch.zeros(B, 1, dtype=torch.long, device=x.device)
            p = self.model.decode_from_zq(zq, ids_in)[:, t].argmax(-1)
            wrong = p != x[:, t]
            esc.append(wrong)
            p = torch.where(wrong, x[:, t], p)
            out = torch.cat([out, p[:, None]], 1)
        return out, torch.stack(esc, 1)

    def compress_text(self, text: str):
        enc = self.hf(text, add_special_tokens=False)
        ids = enc["input_ids"]
        n_tokens = len(ids)
        if n_tokens == 0:
            return {"codes": [], "escapes": [], "n_tokens": 0, "bits_per_token": 0.0}

        L = self.cfg["L"]
        c = self.m[np.array(ids)]
        nb = math.ceil(n_tokens / L)
        pad = np.full(nb * L, self.eos_c, dtype=np.int64)
        pad[:n_tokens] = c
        x = torch.from_numpy(pad).view(nb, L).to(self.device)

        idxs, yes, esc = [], [], []
        autocast_dev = self.device.type if self.device.type in ("cuda", "xpu") else "cpu"
        autocast_enabled = self.device.type in ("cuda", "xpu")
        
        with torch.autocast(autocast_dev, dtype=torch.bfloat16, enabled=autocast_enabled):
            for i in range(0, nb, 32):
                xb = x[i:i + 32]
                ib = self.model.encode(xb)
                idxs.append(ib)
                yb, e = self._escape_decode(ib, xb)
                yes.append(yb)
                esc.append(e)

        roundtrip_compact = torch.cat(yes).reshape(-1)[:n_tokens].cpu().numpy()
        recovered_ids = self.back[roundtrip_compact]
        decompressed_text = self.hf.decode(recovered_ids)
        
        escapes_count = int(torch.cat(esc).reshape(-1)[:n_tokens].sum())
        escape_rate = escapes_count / max(1, n_tokens)
        cpt = self.cfg["depth"] / self.cfg["r"]
        code_bits = cpt * math.log2(self.cfg["K"])
        if 0 < escape_rate < 1:
            H_esc = -escape_rate * math.log2(escape_rate) - (1 - escape_rate) * math.log2(1 - escape_rate)
        else:
            H_esc = 0.0
        bits_per_tok = code_bits + H_esc + escape_rate * math.log2(len(self.vocab))

        return {
            "decompressed_text": decompressed_text,
            "exact_match": (decompressed_text == text),
            "n_tokens": n_tokens,
            "codes_per_token": cpt,
            "bits_per_token": bits_per_tok,
            "escape_rate": escape_rate
        }
