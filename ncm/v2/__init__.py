"""NCM Version 2: Generative Neural Context Models & KV-Cache Compression."""

from .generative_context import (
    AbliterationHook,
    NCMGenerativeModel,
    NCMKVProjector,
    NCMSpeculativeHead,
    ResBlock,
    apply_rope,
    build_dynamic_cache,
    compute_saliency_weights,
    extract_kv_tensors,
    rope_phase_aware_key_pooling,
    rotate_half,
    saliency_aware_value_pooling,
    unapply_rope,
)

__all__ = [
    "NCMGenerativeModel",
    "NCMKVProjector",
    "NCMSpeculativeHead",
    "ResBlock",
    "AbliterationHook",
    "build_dynamic_cache",
    "extract_kv_tensors",
    "rotate_half",
    "apply_rope",
    "unapply_rope",
    "compute_saliency_weights",
    "rope_phase_aware_key_pooling",
    "saliency_aware_value_pooling",
]
