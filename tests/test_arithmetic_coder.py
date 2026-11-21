"""Unit tests for 32-bit Integer Arithmetic Coder and RVQ Codebook Compressor."""

import unittest
import numpy as np

from ncm.compressor import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    decode_rvq_codes_ac,
    encode_rvq_codes_ac,
    quantize_topk_cdf,
)


class TestArithmeticCoder(unittest.TestCase):
    def test_uniform_arithmetic_coding_roundtrip(self):
        """Test uniform arithmetic encoding across small and large alphabets (e.g., 151k)."""
        np.random.seed(42)
        for alphabet_size in [2, 10, 256, 4096, 65536, 151936]:
            symbols = np.random.randint(0, alphabet_size, size=200).tolist()

            enc = ArithmeticEncoder()
            for s in symbols:
                enc.encode_uniform(s, alphabet_size)
            blob = enc.finish()

            dec = ArithmeticDecoder(blob)
            recovered = [dec.decode_uniform(alphabet_size) for _ in range(len(symbols))]

            self.assertEqual(recovered, symbols, f"Uniform AC roundtrip failed for alphabet size {alphabet_size}")

    def test_quantize_topk_cdf_properties(self):
        """Test that quantize_topk_cdf generates valid monotonic int64 CDF ending at PROB_TOTAL."""
        for k in [4, 16, 64, 128]:
            raw_p = np.random.dirichlet(np.ones(k)) * 0.95  # sum < 1 leaves tail
            cdf = quantize_topk_cdf(raw_p)

            self.assertEqual(len(cdf), k + 2)
            self.assertEqual(cdf[0], 0)
            self.assertEqual(cdf[-1], 65536)
            diffs = np.diff(cdf)
            self.assertTrue(np.all(diffs >= 1), f"Non-positive frequency found: {diffs.min()}")

    def test_skewed_distribution_arithmetic_coding(self):
        """Test arithmetic coding with varying step-by-step top-K CDFs."""
        np.random.seed(123)
        k = 32
        seq_len = 100
        symbols = []
        cdfs = []

        for _ in range(seq_len):
            raw_p = np.random.dirichlet(np.ones(k) * 0.5) * 0.98
            cdf = quantize_topk_cdf(raw_p)
            cdfs.append(cdf)
            sym = np.random.randint(0, k + 1)
            symbols.append(sym)

        enc = ArithmeticEncoder()
        for s, cdf in zip(symbols, cdfs):
            enc.encode_symbol(int(cdf[s]), int(cdf[s + 1]), int(cdf[-1]))
        blob = enc.finish()

        dec = ArithmeticDecoder(blob)
        recovered = [dec.decode_symbol(cdf) for cdf in cdfs]
        self.assertEqual(recovered, symbols, "Skewed CDF arithmetic coding roundtrip failed")

    def test_rvq_codes_ac_roundtrip(self):
        """Test RVQ indices lossless arithmetic compression."""
        np.random.seed(99)
        K = 4096
        M = 8
        depth = 2
        n_blocks = 15

        codes = np.random.randint(0, K, size=(n_blocks, M, depth), dtype=np.uint32)
        blob = encode_rvq_codes_ac(codes, K=K)
        self.assertGreater(len(blob), 0)

        decoded = decode_rvq_codes_ac(blob, K=K, M=M)
        self.assertTrue(np.array_equal(codes, decoded), "RVQ codes arithmetic decoding mismatch")


if __name__ == "__main__":
    unittest.main()
