import struct
import zlib
import numpy as np

MAGIC = b"NCM2"
HEADER_FMT = "<4sIIIIHHH"  # magic, orig_bytes, n_tokens, n_blocks, n_escapes, L, r, depth
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def pack_12bit_pairs(arr: np.ndarray) -> bytes:
    """Pack an array of integers in [0, 4095] with even length into 12-bit pairs (3 bytes per pair)."""
    flat = np.asarray(arr, dtype=np.uint32).flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.uint32(0))
    
    n_pairs = len(flat) // 2
    raw = np.empty(n_pairs * 3, dtype=np.uint8)
    c0 = flat[0::2]
    c1 = flat[1::2]
    
    raw[0::3] = (c0 & 0xFF).astype(np.uint8)
    raw[1::3] = (((c0 >> 8) & 0x0F) | ((c1 & 0x0F) << 4)).astype(np.uint8)
    raw[2::3] = ((c1 >> 4) & 0xFF).astype(np.uint8)
    return raw.tobytes()

def unpack_12bit_pairs(data: bytes, count: int) -> np.ndarray:
    """Unpack count integers in [0, 4095] from packed 12-bit pairs."""
    b = np.frombuffer(data, dtype=np.uint8)
    n_pairs = len(b) // 3
    b0 = b[0::3].astype(np.uint32)
    b1 = b[1::3].astype(np.uint32)
    b2 = b[2::3].astype(np.uint32)
    
    c0 = b0 | ((b1 & 0x0F) << 8)
    c1 = ((b1 >> 4) & 0x0F) | (b2 << 4)
    
    out = np.empty(n_pairs * 2, dtype=np.int64)
    out[0::2] = c0
    out[1::2] = c1
    return out[:count]

def serialize_container(
    orig_bytes: int,
    n_tokens: int,
    n_blocks: int,
    L: int,
    r: int,
    depth: int,
    rvq_codes: np.ndarray,
    esc_mask: np.ndarray,
    esc_tokens: np.ndarray
) -> bytes:
    """Serializes NCM codes and escapes into an executable binary bitstream."""
    n_escapes = len(esc_tokens)
    header = struct.pack(HEADER_FMT, MAGIC, orig_bytes, n_tokens, n_blocks, n_escapes, L, r, depth)
    
    # 1. Packed RVQ codes (12-bit packed)
    codes_bytes = pack_12bit_pairs(rvq_codes)
    
    # 2. Packed escape mask (bit-packed)
    mask_bool = esc_mask.astype(bool)
    mask_bytes = np.packbits(mask_bool).tobytes()
    
    # 3. Packed escape tokens (compressed uint32)
    tok_bytes = esc_tokens.astype(np.uint32).tobytes()
    
    # Compress escape payload with zlib level 9
    esc_payload = zlib.compress(struct.pack("<II", len(mask_bytes), len(tok_bytes)) + mask_bytes + tok_bytes, level=9)
    
    # Full container layout: [Header][Codes Len (4B)][Codes Bytes][Escapes Payload]
    return header + struct.pack("<I", len(codes_bytes)) + codes_bytes + esc_payload

def deserialize_container(data: bytes):
    """Deserializes binary NCM bitstream back to components."""
    if len(data) < HEADER_SIZE + 4:
        raise ValueError("Corrupt or incomplete NCM file.")
    
    magic, orig_bytes, n_tokens, n_blocks, n_escapes, L, r, depth = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"Invalid magic identifier: {magic}")
    
    offset = HEADER_SIZE
    codes_len = struct.unpack("<I", data[offset:offset+4])[0]
    offset += 4
    
    codes_data = data[offset:offset+codes_len]
    offset += codes_len
    
    n_codes = n_blocks * (L // r) * depth
    rvq_codes = unpack_12bit_pairs(codes_data, n_codes).reshape((n_blocks, L // r, depth))
    
    # Decompress escape payload
    esc_raw = zlib.decompress(data[offset:])
    mask_len, tok_len = struct.unpack("<II", esc_raw[:8])
    mask_bytes = esc_raw[8:8+mask_len]
    tok_bytes = esc_raw[8+mask_len:8+mask_len+tok_len]
    
    # Unpack mask bits
    esc_mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))[:n_tokens].astype(bool)
    esc_tokens = np.frombuffer(tok_bytes, dtype=np.uint32)
    
    return {
        "orig_bytes": orig_bytes,
        "n_tokens": n_tokens,
        "n_blocks": n_blocks,
        "n_escapes": n_escapes,
        "L": L,
        "r": r,
        "depth": depth,
        "rvq_codes": rvq_codes,
        "esc_mask": esc_mask,
        "esc_tokens": esc_tokens
    }
