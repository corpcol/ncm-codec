"""Unit tests for NCM Version 2 (Generative Context Model, KV Projector, and Speculative Heads)."""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm.v2 import (
    AbliterationHook,
    NCMGenerativeModel,
    NCMKVProjector,
    NCMSpeculativeHead,
    ResBlock,
    build_dynamic_cache,
    extract_kv_tensors,
)


class TestNCMV2(unittest.TestCase):
    def test_kv_projector_shapes(self):
        """Verify NCMKVProjector maps (B, M, d_latent) into (B, num_heads, M, head_dim) KV pairs."""
        B, M, d_latent = 2, 8, 128
        num_layers, num_heads, head_dim = 4, 4, 32

        projector = NCMKVProjector(
            d_latent=d_latent,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        latents = torch.randn(B, M, d_latent)
        kv_pairs = projector(latents)

        self.assertEqual(len(kv_pairs), num_layers)
        for k, v in kv_pairs:
            self.assertEqual(k.shape, (B, num_heads, M, head_dim))
            self.assertEqual(v.shape, (B, num_heads, M, head_dim))

    def test_dynamic_cache_builder_and_extractor(self):
        """Verify build_dynamic_cache and extract_kv_tensors roundtrip."""
        B, H, M, D = 1, 2, 6, 32
        k1 = torch.randn(B, H, M, D)
        v1 = torch.randn(B, H, M, D)
        k2 = torch.randn(B, H, M, D)
        v2 = torch.randn(B, H, M, D)

        cache = build_dynamic_cache([(k1, v1), (k2, v2)])
        out_k1, out_v1 = extract_kv_tensors(cache, 0)
        out_k2, out_v2 = extract_kv_tensors(cache, 1)

        self.assertTrue(torch.allclose(k1, out_k1))
        self.assertTrue(torch.allclose(v1, out_v1))
        self.assertTrue(torch.allclose(k2, out_k2))
        self.assertTrue(torch.allclose(v2, out_v2))

    def test_abliteration_hook(self):
        """Verify AbliterationHook projects out the refusal component orthogonally."""
        D = 64
        v_hat = torch.randn(D)
        hook = AbliterationHook(v_hat)

        # Construct input with significant component along v_hat
        v_norm = F.normalize(v_hat, dim=-1)
        h_orig = torch.randn(2, 5, D) + 3.0 * v_norm
        h_clean = hook(None, None, h_orig)

        # Dot product with v_norm should be ~0.0
        dot_product = (h_clean @ v_norm).abs().max().item()
        self.assertLess(dot_product, 1e-5, f"Residual refusal projection remained: {dot_product}")

    def test_speculative_heads(self):
        """Verify NCMSpeculativeHead produces candidate logits for K future tokens."""
        hidden_size = 64
        vocab_size = 100
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        spec_head = NCMSpeculativeHead(hidden_size=hidden_size, lm_head=lm_head, num_heads=3)

        h = torch.randn(2, 10, hidden_size)
        logits_list = spec_head(h)

        self.assertEqual(len(logits_list), 3)
        for logits in logits_list:
            self.assertEqual(logits.shape, (2, 10, vocab_size))


if __name__ == "__main__":
    unittest.main()
