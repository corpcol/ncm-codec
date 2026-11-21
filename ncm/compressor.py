import os
import math
import time
import numpy as np
import torch
from .codec import Codec
from .container import serialize_container, deserialize_container

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
        
        # Mappings: full vocab to compact index; compact index to full vocab
        self.m = np.full(200000, len(self.vocab), dtype=np.int64)
        self.m[self.vocab] = np.arange(len(self.vocab))
        self.back = np.append(self.vocab, -1)
        self.eos_c = int(self.m[self.hf.eos_token_id]) if self.hf.eos_token_id < len(self.m) else 0

    @torch.no_grad()
    def compress_to_bytes(self, text: str) -> bytes:
        """Compress raw text into an exact binary .ncm container bitstream."""
        raw_bytes = text.encode("utf-8")
        orig_bytes = len(raw_bytes)
        if orig_bytes == 0:
            return serialize_container(0, 0, 0, self.cfg["L"], self.cfg["r"], self.cfg["depth"],
                                       np.empty((0, 0, 0), dtype=np.int64),
                                       np.empty(0, dtype=bool),
                                       np.empty(0, dtype=np.uint32))

        enc = self.hf(text, add_special_tokens=False)
        orig_token_ids = np.array(enc["input_ids"], dtype=np.uint32)
        n_tokens = len(orig_token_ids)

        L = self.cfg["L"]
        nb = math.ceil(n_tokens / L)
        pad = np.full(nb * L, self.eos_c, dtype=np.int64)
        pad[:n_tokens] = self.m[orig_token_ids]
        x = torch.from_numpy(pad).view(nb, L).to(self.device)

        esc_mask_list = []
        esc_tokens_list = []
        all_codes = []

        for b in range(nb):
            xb = x[b:b+1]
            ib = self.model.encode(xb)
            all_codes.append(ib.cpu().numpy())
            zq = self.model.rvq.decode_idx(ib)
            out = torch.zeros(1, 0, dtype=torch.long, device=self.device)

            for t in range(L):
                g_idx = b * L + t
                ids_in = torch.cat([out, out[:, :1]], 1) if t else torch.zeros(1, 1, dtype=torch.long, device=self.device)
                p_logits = self.model.decode_from_zq(zq, ids_in)[:, t]
                pred_c = int(p_logits.argmax(-1).item())

                if g_idx < n_tokens:
                    true_tok = orig_token_ids[g_idx]
                    pred_full = self.vocab[pred_c] if pred_c < len(self.vocab) else -1
                    if pred_full == true_tok:
                        esc_mask_list.append(False)
                        next_tok = pred_c
                    else:
                        esc_mask_list.append(True)
                        esc_tokens_list.append(true_tok)
                        next_tok = int(self.m[true_tok]) if true_tok < len(self.m) else self.eos_c
                else:
                    next_tok = self.eos_c

                out = torch.cat([out, torch.tensor([[next_tok]], device=self.device)], 1)

        rvq_codes = np.concatenate(all_codes, axis=0)
        esc_mask = np.array(esc_mask_list, dtype=bool)
        esc_tokens = np.array(esc_tokens_list, dtype=np.uint32)

        return serialize_container(
            orig_bytes=orig_bytes,
            n_tokens=n_tokens,
            n_blocks=nb,
            L=L,
            r=self.cfg["r"],
            depth=self.cfg["depth"],
            rvq_codes=rvq_codes,
            esc_mask=esc_mask,
            esc_tokens=esc_tokens
        )

    @torch.no_grad()
    def decompress_from_bytes(self, container_bytes: bytes) -> str:
        """Decompress binary .ncm container bitstream back to 100% exact original text."""
        d = deserialize_container(container_bytes)
        if d["n_tokens"] == 0:
            return ""

        n_tokens = d["n_tokens"]
        L = d["L"]
        nb = d["n_blocks"]
        rvq_codes = d["rvq_codes"]
        esc_mask = d["esc_mask"]
        esc_tokens = d["esc_tokens"]

        esc_ptr = 0
        rec_tokens = []

        for b in range(nb):
            ib = torch.from_numpy(rvq_codes[b:b+1]).to(self.device)
            zq = self.model.rvq.decode_idx(ib)
            out = torch.zeros(1, 0, dtype=torch.long, device=self.device)

            for t in range(L):
                g_idx = b * L + t
                if g_idx >= n_tokens:
                    break

                ids_in = torch.cat([out, out[:, :1]], 1) if t else torch.zeros(1, 1, dtype=torch.long, device=self.device)
                p_logits = self.model.decode_from_zq(zq, ids_in)[:, t]
                pred_c = int(p_logits.argmax(-1).item())

                if esc_mask[g_idx]:
                    true_tok = int(esc_tokens[esc_ptr])
                    esc_ptr += 1
                    rec_tokens.append(true_tok)
                    next_tok = int(self.m[true_tok]) if true_tok < len(self.m) else self.eos_c
                else:
                    pred_full = int(self.vocab[pred_c])
                    rec_tokens.append(pred_full)
                    next_tok = pred_c

                out = torch.cat([out, torch.tensor([[next_tok]], device=self.device)], 1)

        return self.hf.decode(rec_tokens)

    def compress_file(self, src_path: str, dst_path: str):
        with open(src_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            text = f.read()
        blob = self.compress_to_bytes(text)
        with open(dst_path, "wb") as f:
            f.write(blob)
        return len(text.encode("utf-8")), len(blob)

    def decompress_file(self, src_path: str, dst_path: str):
        with open(src_path, "rb") as f:
            blob = f.read()
        text = self.decompress_from_bytes(blob)
        with open(dst_path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return len(blob), len(text.encode("utf-8"))

    def benchmark_text(self, text: str):
        """Full benchmark testing on-disk binary size, theoretical BPT, and exact lossless verification."""
        t0 = time.time()
        blob = self.compress_to_bytes(text)
        t_enc = time.time() - t0

        t1 = time.time()
        restored = self.decompress_from_bytes(blob)
        t_dec = time.time() - t1

        exact = (restored == text)
        orig_bytes = len(text.encode("utf-8"))
        comp_bytes = len(blob)
        ratio = orig_bytes / max(1, comp_bytes)
        
        enc = self.hf(text, add_special_tokens=False)
        n_tokens = len(enc["input_ids"])
        empirical_bpt = (comp_bytes * 8) / max(1, n_tokens)
        
        return {
            "exact_match": exact,
            "orig_bytes": orig_bytes,
            "compressed_bytes": comp_bytes,
            "compression_ratio": ratio,
            "n_tokens": n_tokens,
            "empirical_bpt": empirical_bpt,
            "encode_time_s": t_enc,
            "decode_time_s": t_dec
        }
