"""NCM Version 2: Generative Context Model & LLM KV-Cache Compression.

Converts pretrained HuggingFace Causal LLMs into Neural Context Models (NCM) capable of:
1. Compressing prompt context into 4x-reduced latent memory slots (75% KV-cache VRAM savings).
2. Direct generative next-token inference over compressed KV states.
3. Universal compatibility across Transformers 4.x and 5.x DynamicCache implementations.
4. On-demand bit-for-bit lossless context recovery via Stage 1 NCM neural arithmetic coding.
5. Quantized execution (4-bit NF4 / 8-bit) and speculative decoding support.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_kv_tensors(cache: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Universal extractor for key/value tensors across Transformers 4.x and 5.x."""
    if hasattr(cache, "layers") and layer_idx < len(cache.layers):
        layer = cache.layers[layer_idx]
        k = getattr(layer, "keys", None)
        if k is None:
            k = getattr(layer, "key_states", None)
        v = getattr(layer, "values", None)
        if v is None:
            v = getattr(layer, "value_states", None)
        if k is not None and v is not None:
            return k, v

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    if isinstance(cache, (tuple, list)) and layer_idx < len(cache):
        layer_tuple = cache[layer_idx]
        if isinstance(layer_tuple, (tuple, list)) and len(layer_tuple) >= 2:
            return layer_tuple[0], layer_tuple[1]

    raise ValueError(f"Unable to extract key/value tensors from cache type: {type(cache)}")


def build_dynamic_cache(kv_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Instantiate a Transformers DynamicCache populated with precomputed KV tensors."""
    try:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for l, (k, v) in enumerate(kv_pairs):
            cache.update(k, v, layer_idx=l)
        return cache
    except Exception:
        return tuple(kv_pairs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Forward Rotary Position Embedding rotation: x * cos(angles) + rotate_half(x) * sin(angles)."""
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return (x * cos) + (rotate_half(x) * sin)


def unapply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Inverse Rotary Position Embedding rotation (harmonic unwrapping to position 0)."""
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return (x * cos) - (rotate_half(x) * sin)


def compute_saliency_weights(
    k: torch.Tensor,
    r: int = 4,
    temperature: float = 1.0,
    normalize_by_dim: bool = True,
) -> torch.Tensor:
    """Computes attention-saliency softmax weights based on Key L2 norm within each r-token chunk.

    Args:
        k: Key tensor of shape (B, H, N, D).
        r: Compression/pooling factor (N must be divisible by r).
        temperature: Softmax scaling temperature.
        normalize_by_dim: If True, scales raw L2 norms by 1/sqrt(D) to calibrate against dimension scaling.

    Returns:
        Normalized weights of shape (B, H, M, r) summing to 1.0 along the chunk dimension.
    """
    B, H, N, D = k.shape
    M = N // r
    norms = torch.norm(k.view(B, H, M, r, D), p=2, dim=-1)
    if normalize_by_dim:
        norms = norms / math.sqrt(D)
    weights = F.softmax(norms / max(1e-4, temperature), dim=-1)
    return weights


def rope_phase_aware_key_pooling(
    k: torch.Tensor,
    r: int = 4,
    rope_theta: float = 1000000.0,
    weights: Optional[torch.Tensor] = None,
    inv_freq: Optional[torch.Tensor] = None,
    centroid_in_original_space: bool = False,
) -> torch.Tensor:
    """Automorphic Harmonic Phase-Unwrapped Key Pooling.

    Unwraps rotary position embedding rotations from position p back to position 0,
    pools base semantic vectors across chunks of size r (with optional saliency weights),
    and re-rotates the pooled vector to target coordinates.
    Eliminates destructive phase cancellation and norm decay in high-frequency RoPE channels.

    Args:
        k: Key tensor of shape (B, H, N, D).
        r: Pooling factor (N must be divisible by r).
        rope_theta: Base frequency for rotary embeddings (used when inv_freq is None).
        weights: Optional saliency weights of shape (B, H, M, r).
        inv_freq: Optional precomputed inverse frequencies from the model's rotary_emb module
                  (supports dynamic / piecewise RoPE scaling e.g. LLaMA 3.1).
        centroid_in_original_space: If True, re-rotates to centroid positions (m * r + (r - 1) / 2).
                                    If False (default), re-rotates to compressed macro-slot coordinates (m),
                                    ensuring continuous sequence length indexing with DynamicCache.

    Returns:
        Pooled Key tensor of shape (B, H, M, D) where M = N // r.
    """
    B, H, N, D = k.shape
    M = N // r
    device, dtype = k.device, k.dtype

    if inv_freq is None:
        # RoPE inverse frequencies: 1 / (theta ** (2i / D))
        inv_freq_half = 1.0 / (rope_theta ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
        inv_freq = torch.cat([inv_freq_half, inv_freq_half], dim=-1).to(dtype)
    else:
        inv_freq = inv_freq.to(device=device, dtype=dtype)
        if inv_freq.shape[-1] == D // 2:
            inv_freq = torch.cat([inv_freq, inv_freq], dim=-1)

    # Positions for all N tokens: (1, 1, N, D)
    pos = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(-1)
    angles = (pos * inv_freq.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(dtype)

    # 1. Harmonic Phase-Unwrapping back to position 0
    k_unwrapped = unapply_rope(k, angles)

    # 2. Vectorized pooling across chunk r
    k_unwrapped_chunks = k_unwrapped.view(B, H, M, r, D)
    if weights is not None:
        k_base_pooled = (k_unwrapped_chunks * weights.unsqueeze(-1)).sum(dim=3)
    else:
        k_base_pooled = k_unwrapped_chunks.mean(dim=3)

    # 3. Re-rotation to target position
    if centroid_in_original_space:
        target_pos = (torch.arange(M, device=device, dtype=torch.float32) * r + (r - 1) / 2.0).unsqueeze(-1)
    else:
        target_pos = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(-1)

    target_angles = (target_pos * inv_freq.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(dtype)

    k_pooled = apply_rope(k_base_pooled, target_angles)
    return k_pooled


def saliency_aware_value_pooling(
    v: torch.Tensor,
    weights: torch.Tensor,
    r: int = 4,
) -> torch.Tensor:
    """Pools value states weighted by token saliency.

    Allocates greater representation to high-norm / high-information tokens
    (keywords, syntax symbols) over low-entropy boilerplate or whitespace.

    Args:
        v: Value tensor of shape (B, H, N, D).
        weights: Saliency weights of shape (B, H, M, r).
        r: Pooling factor (N must be divisible by r).

    Returns:
        Pooled Value tensor of shape (B, H, M, D) where M = N // r.
    """
    B, H, N, D = v.shape
    M = N // r
    v_chunks = v.view(B, H, M, r, D)
    return (v_chunks * weights.unsqueeze(-1)).sum(dim=3)


class NCMKVProjector(nn.Module):
    """Learned cross-attention / KV projection head mapping NCM latents into LLM KV cache tensors."""

    def __init__(self, d_latent: int, num_layers: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim

        self.proj_k = nn.ModuleList([nn.Linear(d_latent, self.d_model) for _ in range(num_layers)])
        self.proj_v = nn.ModuleList([nn.Linear(d_latent, self.d_model) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_latent)

        # Initialize with small scale to prevent uncalibrated early layer blowup
        for p in self.proj_k:
            nn.init.normal_(p.weight, mean=0.0, std=0.02)
            nn.init.zeros_(p.bias)
        for p in self.proj_v:
            nn.init.normal_(p.weight, mean=0.0, std=0.02)
            nn.init.zeros_(p.bias)

    def forward(self, latents: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Maps (B, M, d_latent) into a list of (Key, Value) tuples for each layer:
        Key:   (B, num_heads, M, head_dim)
        Value: (B, num_heads, M, head_dim)
        """
        B, M, _ = latents.shape
        h = self.ln(latents)
        past_key_values = []
        for l in range(self.num_layers):
            k = self.proj_k[l](h).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.proj_v[l](h).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
            past_key_values.append((k, v))
        return past_key_values


class AbliterationHook:
    """Forward hook removing projections along refusal direction v_hat: h' = h - (h @ v_hat) * v_hat."""

    def __init__(self, v_hat: torch.Tensor):
        self.v_hat = F.normalize(v_hat.float(), dim=-1)

    def __call__(self, module: nn.Module, input_tensors: Any, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        v = self.v_hat.to(device=h.device, dtype=h.dtype)
        proj = torch.einsum("...d,d->...", h, v).unsqueeze(-1) * v
        h_clean = h - proj
        return (h_clean, *output[1:]) if is_tuple else h_clean


class ResBlock(nn.Module):
    """Residual projection block for multiscale speculative drafting."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class NCMSpeculativeHead(nn.Module):
    """Multi-token speculative prediction heads tied to base model's lm_head for fast drafting."""

    def __init__(self, hidden_size: int, lm_head: nn.Module, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.lm_head = lm_head
        self.blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_heads)])

    def forward(self, h: torch.Tensor) -> List[torch.Tensor]:
        logits_list = []
        lm_weight = getattr(self.lm_head, "weight", None)
        target_dtype = lm_weight.dtype if lm_weight is not None else torch.bfloat16
        for i in range(self.num_heads):
            feat = self.blocks[i](h.to(self.blocks[i].linear.weight.dtype))
            logits = self.lm_head(feat.to(target_dtype))
            logits_list.append(logits)
        return logits_list


class NCMGenerativeModel(nn.Module):
    """NCM Version 2 Generative Model: Pretrained LLM with 4x Context Compression & Lossless Recovery."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        compression_factor: int = 4,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        stage1_ckpt: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.compression_factor = compression_factor
        self.stage1_ckpt = stage1_ckpt
        self.stage1_compressor: Any = None

        print(f"[*] Initializing NCM V2 Generative Backbone: {model_name_or_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        kwargs: Dict[str, Any] = {}
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                if load_in_4bit:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                print("[!] bitsandbytes not found; continuing in standard precision.")

        if device_map:
            kwargs["device_map"] = device_map
        elif not (load_in_4bit or load_in_8bit):
            kwargs["dtype"] = dtype or (torch.bfloat16 if self.device.type in ("cuda", "xpu") else torch.float32)

        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if not device_map and not (load_in_4bit or load_in_8bit):
            self.llm = self.llm.to(self.device)
        self.llm.eval()

        cfg = self.llm.config
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // cfg.num_attention_heads)
        self.rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))

        # Extract native rotary embedding inverse frequencies if available (supports LLaMA 3.1+ piecewise scaling)
        self.inv_freq: Optional[torch.Tensor] = None
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "rotary_emb"):
            self.inv_freq = getattr(self.llm.model.rotary_emb, "inv_freq", None)
        elif hasattr(self.llm, "rotary_emb"):
            self.inv_freq = getattr(self.llm.rotary_emb, "inv_freq", None)

        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        ).to(self.device)

        # KV cache injection adapter
        self.kv_projector = NCMKVProjector(
            d_latent=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).to(self.device)

    def _get_stage1_compressor(self):
        if self.stage1_compressor is None and self.stage1_ckpt is not None:
            from ncm.compressor import NCMCompressor
            self.stage1_compressor = NCMCompressor(ckpt_path=self.stage1_ckpt, device=str(self.device))
        return self.stage1_compressor

    @torch.no_grad()
    def compress_context(
        self,
        context_text: str,
        mode: str = "kv_pool",
        preserve_lossless: bool = True,
        use_rope_unwrapping: bool = True,
        use_saliency: bool = True,
        saliency_temperature: float = 1.0,
        centroid_in_original_space: bool = False,
    ) -> Dict[str, Any]:
        """Compress context text into 4x-downsampled KV cache states.

        Args:
            context_text: Prompt context string.
            mode: 'kv_pool' (zero-shot calibrated pooling of backbone KV tensors)
                  or 'learned' (forward through context_encoder + kv_projector).
            preserve_lossless: Whether to archive Stage 1 arithmetic bitstream for 100% exact lossless recovery.
            use_rope_unwrapping: If True, uses automorphic harmonic phase-unwrapping on Keys to eliminate norm decay.
            use_saliency: If True, weights Value (and Key) pooling by attention saliency (Key L2 norm).
            saliency_temperature: Softmax scaling temperature for saliency weights.
            centroid_in_original_space: If True, re-rotates to centroid positions (m * r + (r-1)/2).
                                        If False (default), re-rotates to compressed macro-slot coordinates (m)
                                        for seamless DynamicCache sequence length indexing.
        """
        enc = self.tokenizer(context_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        B, N = input_ids.shape
        r = self.compression_factor

        pad_len = (r - (N % r)) % r
        if pad_len > 0:
            pad = torch.full((B, pad_len), self.tokenizer.eos_token_id or 0, dtype=torch.long, device=self.device)
            input_ids_padded = torch.cat([input_ids, pad], dim=1)
        else:
            input_ids_padded = input_ids

        total_tokens = input_ids_padded.size(1)
        M = total_tokens // r

        lossless_blob: Optional[bytes] = None
        if preserve_lossless:
            comp = self._get_stage1_compressor()
            if comp is not None:
                res = comp.compress_text(context_text, verify=False)
                lossless_blob = res.blob

        if mode == "kv_pool":
            # Extract true calibrated Key and Value states from backbone and pool by factor r
            fwd = self.llm(input_ids_padded, use_cache=True)
            orig_kv = fwd.past_key_values
            compressed_kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

            for l in range(self.num_layers):
                k, v = extract_kv_tensors(orig_kv, l)
                B_l, H_l, N_l, D_l = k.shape

                s_weights = (
                    compute_saliency_weights(k, r=r, temperature=saliency_temperature)
                    if use_saliency
                    else None
                )

                if use_rope_unwrapping:
                    k_pooled = rope_phase_aware_key_pooling(
                        k,
                        r=r,
                        rope_theta=self.rope_theta,
                        weights=s_weights,
                        inv_freq=self.inv_freq,
                        centroid_in_original_space=centroid_in_original_space,
                    )
                else:
                    if s_weights is not None:
                        k_pooled = (k.view(B_l, H_l, M, r, D_l) * s_weights.unsqueeze(-1)).sum(dim=3)
                    else:
                        k_pooled = k.view(B_l, H_l, M, r, D_l).mean(dim=3)

                if use_saliency and s_weights is not None:
                    v_pooled = saliency_aware_value_pooling(v, s_weights, r=r)
                else:
                    v_pooled = v.view(B_l, H_l, M, r, D_l).mean(dim=3)

                compressed_kv_pairs.append((k_pooled, v_pooled))

            compressed_cache = build_dynamic_cache(compressed_kv_pairs)
            latents = None

        else:
            # Learned adapter mode
            embed_fn = self.llm.get_input_embeddings()
            embeddings = embed_fn(input_ids_padded).to(self.context_encoder.layers[0].linear1.weight.dtype)
            h = self.context_encoder(embeddings)
            latents = h.view(B, M, r, self.hidden_size).mean(dim=2)
            compressed_kv_pairs = self.kv_projector(latents)
            compressed_cache = build_dynamic_cache(compressed_kv_pairs)

        return {
            "compressed_kv": compressed_cache,
            "latents": latents,
            "orig_tokens": N,
            "compressed_slots": M,
            "compression_ratio": N / max(1, M),
            "lossless_blob": lossless_blob,
            "raw_text": context_text,
        }

    def decompress_context(self, compressed_result: Dict[str, Any]) -> str:
        """Bit-for-bit lossless context recovery fulfilling NCM V2 Stage 1/2 integration."""
        blob = compressed_result.get("lossless_blob", None)
        if blob is not None:
            comp = self._get_stage1_compressor()
            if comp is not None:
                return comp.decompress_text(blob)
        return compressed_result.get("raw_text", "")

    @torch.no_grad()
    def generate_with_compressed_context(
        self,
        context_text: str,
        query: str,
        max_new_tokens: int = 64,
        mode: str = "kv_pool",
        temperature: float = 0.0,
        use_rope_unwrapping: bool = True,
        use_saliency: bool = True,
        saliency_temperature: float = 1.0,
        centroid_in_original_space: bool = False,
    ) -> Dict[str, Any]:
        """Performs generative inference: LLM attends to 4x compressed context to answer query."""
        comp = self.compress_context(
            context_text,
            mode=mode,
            preserve_lossless=True,
            use_rope_unwrapping=use_rope_unwrapping,
            use_saliency=use_saliency,
            saliency_temperature=saliency_temperature,
            centroid_in_original_space=centroid_in_original_space,
        )
        compressed_cache = comp["compressed_kv"]

        # Encode query prompt
        q_enc = self.tokenizer(f"\nQuestion: {query}\nAnswer:", return_tensors="pt")
        q_ids = q_enc["input_ids"].to(self.device)

        B = q_ids.size(0)
        M = comp["compressed_slots"]
        Q = q_ids.size(1)

        # Position IDs covering prefilled compressed slots + query
        start_pos = comp["orig_tokens"] if centroid_in_original_space else M

        attention_mask = torch.ones((B, M + Q), device=self.device)
        position_ids = torch.arange(start_pos, start_pos + Q, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(M, M + Q, dtype=torch.long, device=self.device)

        gen_kwargs: Dict[str, Any] = {
            "input_ids": q_ids,
            "past_key_values": compressed_cache,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.llm.generate(**gen_kwargs)
        response = self.tokenizer.decode(outputs[0][Q:], skip_special_tokens=True).strip()

        return {
            "query": query,
            "response": response,
            "orig_context_tokens": comp["orig_tokens"],
            "compressed_kv_tokens": comp["compressed_slots"],
            "kv_memory_savings": f"{(1.0 - (comp['compressed_slots'] / comp['orig_tokens'])) * 100:.1f}%",
            "lossless_blob_size": len(comp["lossless_blob"]) if comp["lossless_blob"] else 0,
        }
