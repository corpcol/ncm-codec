"""End-to-end tests for NCMCompressor and neural arithmetic bitstreams."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ncm.codec import Codec
from ncm.compressor import NCMCompressor


class TestCompressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a lightweight synthetic Codec checkpoint for fast unit testing."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.ckpt_path = Path(cls.temp_dir.name) / "mock_codec.pt"

        V, dt, L, r, K = 512, 64, 16, 4, 128
        E = torch.randn(V, dt) * 0.02
        model = Codec(
            E=E, L=L, r=r, d=64, dq=32, enc_layers=1, dec_layers=1, heads=2, depth=2, K=K, drop=0.0
        )

        cfg = {
            "V": V,
            "dt": dt,
            "L": L,
            "r": r,
            "d": 64,
            "dq": 32,
            "enc_layers": 1,
            "dec_layers": 1,
            "heads": 2,
            "depth": 2,
            "K": K,
            "mode": "ar",
            "pos_scale": 0.1,
        }
        vocab = list(range(V))
        torch.save({"cfg": cfg, "vocab": vocab, "state": model.state_dict()}, cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_compressor_tokens_roundtrip_lossless(self):
        """Verify bit-exact lossless roundtrip for token IDs within compact vocab."""
        comp = NCMCompressor(ckpt_path=self.ckpt_path, device="cpu", top_k=32)

        # Sequence of 45 tokens (crossing multiple chunks with padding)
        raw_ids = [int(i % 500) for i in range(45)]
        res = comp.compress_tokens(raw_ids, verify=True)

        self.assertTrue(res.exact_match)
        self.assertEqual(res.orig_num_tokens, 45)
        self.assertGreater(len(res.blob), 0)

        recovered = comp.decompress_tokens(res.blob)
        self.assertEqual(recovered, raw_ids)

    def test_compressor_oov_tokens_lossless(self):
        """Verify bit-exact lossless preservation of out-of-vocabulary (OOV) tokens."""
        comp = NCMCompressor(ckpt_path=self.ckpt_path, device="cpu", top_k=32)

        # Inject several token IDs well beyond compact_V=512 (simulating raw Qwen 151k tokens)
        raw_ids = [10, 20, 150000, 30, 40, 149999, 50]
        res = comp.compress_tokens(raw_ids, verify=True)

        self.assertTrue(res.exact_match)
        self.assertGreater(res.oov_rate, 0.0)

        recovered = comp.decompress_tokens(res.blob)
        self.assertEqual(recovered, raw_ids)

    def test_compressor_file_api_roundtrip(self):
        """Verify compress_file and decompress_file helper methods."""
        comp = NCMCompressor(ckpt_path=self.ckpt_path, device="cpu", top_k=32)

        with tempfile.TemporaryDirectory() as td:
            input_file = Path(td) / "sample.py"
            output_ncm = Path(td) / "sample.ncm"
            restored_file = Path(td) / "restored.py"

            code_sample = 'def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n'
            input_file.write_text(code_sample, encoding="utf-8")

            orig_sz, comp_sz = comp.compress_file(input_file, output_ncm, verify=True)
            self.assertTrue(output_ncm.exists())
            self.assertGreater(comp_sz, 0)

            blob_sz, rest_sz = comp.decompress_file(output_ncm, restored_file)
            self.assertTrue(restored_file.exists())
            self.assertEqual(restored_file.read_text(encoding="utf-8"), code_sample)


if __name__ == "__main__":
    unittest.main()
