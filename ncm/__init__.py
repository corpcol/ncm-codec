"""NCM: Neural Codec for Code.
Lossless code compression beating LZMA -9 via discrete neural vector quantization.
"""

from .codec import Codec, RVQ, EMACodebook
from .compressor import NCMCompressor

__version__ = "0.1.0"
__all__ = ["Codec", "RVQ", "EMACodebook", "NCMCompressor"]
