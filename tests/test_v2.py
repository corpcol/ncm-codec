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

    def test_rope_phase_aware_key_pooling_norm_preservation(self):
        """Verify harmonic phase unwrapping preserves vector norm against destructive RoPE cancellation."""
        from ncm.v2 import apply_rope, rope_phase_aware_key_pooling

        B, H, N, D, r = 1, 1, 16, 64, 4
        rope_theta = 10000.0  # Use smaller theta base to induce high-frequency rotation
        M = N // r

        # Generate constant semantic vector and rotate it across sequence
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        inv_freq = torch.cat([inv_freq, inv_freq], dim=-1)
        pos = torch.arange(N, dtype=torch.float32).unsqueeze(-1)
        angles = (pos * inv_freq.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        base_key = torch.randn(1, 1, 1, D)
        base_key = base_key / base_key.norm(dim=-1, keepdim=True)
        keys = apply_rope(base_key.expand(B, H, N, D), angles)

        # Naive mean pooling across chunks
        naive_pooled = keys.view(B, H, M, r, D).mean(dim=3)
        naive_norm = naive_pooled.norm(dim=-1).mean().item()

        # Phase-aware harmonic unwrapped pooling
        phase_pooled = rope_phase_aware_key_pooling(keys, r=r, rope_theta=rope_theta)
        phase_norm = phase_pooled.norm(dim=-1).mean().item()

        self.assertEqual(phase_pooled.shape, (B, H, M, D))
        # Phase-aware pooling must achieve virtually 100% norm preservation (>= 0.999)
        self.assertGreater(phase_norm, 0.999)
        self.assertGreater(phase_norm, naive_norm)

    def test_saliency_aware_value_pooling(self):
        """Verify saliency weighting allocates retention proportionally to Key L2 norm."""
        from ncm.v2 import compute_saliency_weights, saliency_aware_value_pooling

        B, H, N, D, r = 1, 1, 8, 32, 4
        M = N // r

        k = torch.ones(B, H, N, D) * 0.1
        # Make token 1 in chunk 0 a dominant heavy-hitter identifier
        k[0, 0, 1] = 50.0

        v = torch.randn(B, H, N, D)
        weights = compute_saliency_weights(k, r=r, temperature=1.0)
        v_pooled = saliency_aware_value_pooling(v, weights, r=r)

        self.assertEqual(v_pooled.shape, (B, H, M, D))
        # The pooled value for chunk 0 should closely match token 1's value (> 0.999 cosine sim)
        chunk0_v = v_pooled[0, 0, 0]
        token1_v = v[0, 0, 1]
        sim = F.cosine_similarity(chunk0_v, token1_v, dim=0).item()
        self.assertGreater(sim, 0.999)


if __name__ == "__main__":
    unittest.main()
