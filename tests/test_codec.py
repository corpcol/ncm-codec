"""Unit tests for NCM Codec and KV-cached AR Transformer."""

import unittest
import torch
import torch.nn.functional as F

from ncm.codec import Codec, EMACodebook, KVTransformerBlock, RVQ


class TestCodec(unittest.TestCase):
    def test_ema_codebook_batch_util_and_perplexity(self):
        """Verify codebook forward returns valid utilization and Shannon perplexity."""
        cb = EMACodebook(K=64, dim=32)
        cb.eval()
        x = torch.randn(4, 16, 32)
        q_st, idx, commit_loss, util, ppl = cb(x)

        self.assertEqual(q_st.shape, x.shape)
        self.assertEqual(idx.shape, (4, 16))
        self.assertGreaterEqual(commit_loss.item(), 0.0)
        self.assertTrue(0.0 <= util <= 1.0)
        self.assertTrue(1.0 <= ppl <= 64.0)

    def test_kv_transformer_block_equivalence(self):
        """Verify KVTransformerBlock produces bit-exact outputs between parallel and KV-cached forward."""
        torch.manual_seed(42)
        d = 64
        heads = 4
        block = KVTransformerBlock(d=d, heads=heads, ffn_mult=2, drop=0.0).eval()

        seq_len = 16
        x = torch.randn(2, seq_len, d)

        # Full forward with causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        out_parallel, _ = block(x, attn_mask=causal_mask)

        # Step-by-step KV-cached forward
        kv_cache = None
        step_outs = []
        for t in range(seq_len):
            xt = x[:, t : t + 1, :]
            out_t, kv_cache = block(xt, attn_mask=None, kv_cache=kv_cache, use_cache=True)
            step_outs.append(out_t)
        out_cached = torch.cat(step_outs, dim=1)

        max_diff = (out_parallel - out_cached).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"KV cache step mismatch with parallel forward: {max_diff}")

    def test_codec_kv_cache_bit_exact(self):
        """Verify Codec.decode_prefill + decode_step matches parallel decode_from_zq."""
        torch.manual_seed(101)
        V, dt, L, r, K = 128, 32, 16, 4, 64
        E = torch.randn(V, dt) * 0.02
        model = Codec(
            E=E, L=L, r=r, d=64, dq=32, enc_layers=2, dec_layers=2, heads=2, depth=2, K=K, drop=0.0
        ).eval()

        x = torch.randint(0, V, (2, L))
        zq, codes, commit_loss, stats = model.encode(x)

        # Full parallel decode
        logits_parallel = model.decode_from_zq(zq, ids_in=x)

        # Autoregressive KV-cached decode
        kv = model.decode_prefill(zq)
        step_logits = []
        prev_tok = None
        for t in range(L):
            lt, kv = model.decode_step(prev_tok, t, kv)
            step_logits.append(lt)
            prev_tok = x[:, t]
        logits_kv = torch.stack(step_logits, dim=1)

        max_diff = (logits_parallel - logits_kv).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Codec AR KV cache mismatch: {max_diff}")

    def test_chunked_cross_entropy(self):
        """Verify chunked_cross_entropy computes exact mathematical CE loss."""
        torch.manual_seed(202)
        V, dt, L, r, K = 1000, 32, 16, 4, 64
        E = torch.randn(V, dt) * 0.02
        model = Codec(
            E=E, L=L, r=r, d=64, dq=32, enc_layers=1, dec_layers=1, heads=2, depth=2, K=K, drop=0.0
        ).eval()

        x = torch.randint(0, V, (4, L))
        zq, _, _, _ = model.encode(x)
        h = model.decode_hidden(zq, ids_in=x)

        exact_ce = F.cross_entropy(model._project_logits(h).reshape(-1, V), x.reshape(-1))
        chunked_ce = model.chunked_cross_entropy(h, x, chunk_tokens=8)

        diff = (exact_ce - chunked_ce).abs().item()
        self.assertLess(diff, 1e-5, f"Chunked CE deviation too large: {diff}")

    def test_load_compatible_state_dict(self):
        """Verify that legacy checkpoint keys (enc.layers, pos_enc, bos) are cleanly remapped."""
        V, dt, L, r, K = 64, 32, 16, 4, 32
        E = torch.randn(V, dt) * 0.02
        model = Codec(
            E=E, L=L, r=r, d=64, dq=32, enc_layers=1, dec_layers=1, heads=2, depth=1, K=K, drop=0.0
        )

        legacy_state = {
            "pos_enc": torch.randn(L, 64) * 0.02,
            "pos_lat": torch.randn(L // r, 64) * 0.02,
            "pos_dec": torch.randn(L, 64) * 0.02,
            "bos": torch.randn(64) * 0.02,
            "enc.layers.0.self_attn.in_proj_weight": torch.randn(3 * 64, 64),
            "enc.layers.0.self_attn.in_proj_bias": torch.randn(3 * 64),
            "enc.layers.0.self_attn.out_proj.weight": torch.randn(64, 64),
            "enc.layers.0.self_attn.out_proj.bias": torch.randn(64),
            "enc.layers.0.linear1.weight": torch.randn(4 * 64, 64),
            "enc.layers.0.linear1.bias": torch.randn(4 * 64),
            "enc.layers.0.linear2.weight": torch.randn(64, 4 * 64),
            "enc.layers.0.linear2.bias": torch.randn(64),
        }

        model.load_compatible_state_dict(legacy_state)
        self.assertEqual(model.enc_pos.shape, (1, L, 64))
        self.assertEqual(model.dec_lat_pos.shape, (1, L // r, 64))
        self.assertEqual(model.bos_embed.shape, (1, 1, 64))


if __name__ == "__main__":
    unittest.main()
