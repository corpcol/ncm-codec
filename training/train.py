"""Production Stage 1 training script for corpcol/ncm-codec (CUDA / Intel XPU B70 / MPS / CPU).

Upgrades:
- Chunked 151k-vocab cross-entropy (`chunked_cross_entropy`) cutting logits VRAM 4-8x.
- Linear warmup + cosine LR decay + proper weight-decay parameter groups.
- Gradient accumulation + live ETA per eval step + checkpoint resume (`--resume`).
- Supports local code/text directories (`--data-dir`) OR HuggingFace datasets (`--hf-dataset`).
- Built-in `--self-test` verifying tok/pos norm ratio, KV-cache equivalence (`< 1e-4`),
  AND full 32-bit neural arithmetic coding roundtrip including OOV tokens.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure local ncm package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ncm.codec import Codec
from ncm.compressor import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    decode_rvq_codes_ac,
    encode_rvq_codes_ac,
    quantize_topk_cdf,
    select_device,
)


def unigram_entropy_bits(token_ids: np.ndarray) -> float:
    if token_ids.size == 0:
        return 0.0
    _, counts = np.unique(token_ids, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def load_packed_chunks_from_dir(
    data_dir: Path,
    tokenizer,
    L: int,
    exts: Tuple[str, ...] = (".py", ".txt", ".md", ".c", ".cpp", ".h", ".rs", ".ts", ".js"),
    val_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix in exts)
    if not files:
        raise FileNotFoundError(f"No source files ({exts}) found in {data_dir}")

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    stream: List[int] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        stream.extend(tokenizer.encode(text, add_special_tokens=False))
        stream.append(int(eos_id))

    n_chunks = len(stream) // L
    if n_chunks < 2:
        raise ValueError(f"Not enough tokens ({len(stream)}) for >= 2 chunks of L={L}")
    arr = np.asarray(stream[: n_chunks * L], dtype=np.int64).reshape(n_chunks, L)
    n_val = max(1, int(round(n_chunks * val_fraction)))
    return arr[:-n_val], arr[-n_val:]


def load_packed_chunks_from_hf(
    dataset_name: str,
    text_column: str,
    tokenizer,
    L: int,
    max_docs: int = 10000,
    val_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train", streaming=True)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    stream: List[int] = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        text = row.get(text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue
        stream.extend(tokenizer.encode(text, add_special_tokens=False))
        stream.append(int(eos_id))

    n_chunks = len(stream) // L
    if n_chunks < 2:
        raise ValueError(f"Not enough tokens ({len(stream)}) for >= 2 chunks of L={L}")
    arr = np.asarray(stream[: n_chunks * L], dtype=np.int64).reshape(n_chunks, L)
    n_val = max(1, int(round(n_chunks * val_fraction)))
    return arr[:-n_val], arr[-n_val:]


@torch.no_grad()
def run_self_test(device: torch.device) -> None:
    """Verify tok/pos norm ratio, KV-cache exactness, and 32-bit neural AC roundtrip."""
    torch.manual_seed(0)
    V, dt, L, r, K = 256, 64, 32, 4, 64
    E = torch.randn(V, dt) * 0.02
    model = Codec(
        E=E, L=L, r=r, d=128, dq=64, enc_layers=2, dec_layers=2, heads=4, depth=2, K=K, drop=0.0
    ).to(device).eval()

    x = torch.randint(0, V, (2, L), device=device)
    ratio = model.position_signal_ratio(x)
    assert ratio >= 5.0, f"Position signal dominates! tok/pos ratio = {ratio:.2f}"

    zq, codes, _, _ = model.encode(x)
    logits_full = model.decode_from_zq(zq, ids_in=x)

    kv = model.decode_prefill(zq)
    step_logits = []
    prev = None
    for t in range(L):
        lt, kv = model.decode_step(prev, t, kv)
        step_logits.append(lt)
        prev = x[:, t]
    logits_kv = torch.stack(step_logits, dim=1)
    max_diff = (logits_full - logits_kv).abs().max().item()
    assert max_diff < 1e-4, f"KV-cache mismatch: max_abs_diff={max_diff:.6f}"

    # Verify RVQ empirical arithmetic coding + neural top-K arithmetic coding roundtrip
    codes_np = codes.cpu().numpy().astype(np.uint32)
    codes_blob = encode_rvq_codes_ac(codes_np, K=K)
    codes_dec = decode_rvq_codes_ac(codes_blob, K=K, M=L // r)
    assert np.array_equal(codes_np, codes_dec), "RVQ arithmetic coding roundtrip mismatch"

    # Simulate top-K + OOV arithmetic roundtrip on sequence 0
    K_top = 32
    raw_V = 151936
    probs = F.softmax(logits_kv[0].float(), dim=-1)
    tp, ti = torch.topk(probs, k=K_top, dim=-1)
    tp_np, ti_np = tp.cpu().numpy(), ti.cpu().numpy()
    raw_seq = x[0].cpu().tolist()
    raw_seq[5] = 149999  # Inject an OOV raw Qwen token

    enc = ArithmeticEncoder()
    for t in range(L):
        cdf = quantize_topk_cdf(tp_np[t])
        rid = raw_seq[t]
        matches = np.where(ti_np[t] == rid)[0] if rid < V else np.array([], dtype=np.int64)
        if matches.size > 0:
            rk = int(matches[0])
            enc.encode_symbol(int(cdf[rk]), int(cdf[rk + 1]), int(cdf[-1]))
        else:
            enc.encode_symbol(int(cdf[K_top]), int(cdf[K_top + 1]), int(cdf[-1]))
            enc.encode_uniform(rid, raw_V)
    blob = enc.finish()

    dec = ArithmeticDecoder(blob)
    recovered = []
    for t in range(L):
        cdf = quantize_topk_cdf(tp_np[t])
        sym = dec.decode_symbol(cdf)
        if sym == K_top:
            recovered.append(dec.decode_uniform(raw_V))
        else:
            recovered.append(int(ti_np[t, sym]))
    assert recovered == raw_seq, "Neural AC roundtrip mismatch!"
    print(
        f"[PASS] tok/pos={ratio:.2f} | KV max_diff={max_diff:.2e} | "
        f"RVQ AC + Top-{K_top} Neural AC (with OOV) 100% bit-exact ({len(blob)} bytes)"
    )


def build_optimizer(model: Codec, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "bias" in name or "pos" in name or "bos" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


@torch.no_grad()
def evaluate(
    model: Codec,
    val_chunks: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
    tokenizer=None,
) -> dict:
    model.eval()
    total_ce = 0.0
    total_correct = 0
    total_tokens = 0
    all_preds: List[np.ndarray] = []
    util_accum = np.zeros(model.depth, dtype=np.float64)
    ppl_accum = np.zeros(model.depth, dtype=np.float64)
    n_batches = 0

    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16 and device.type in ("cuda", "xpu")
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    for start in range(0, len(val_chunks), batch_size):
        x = torch.from_numpy(val_chunks[start : start + batch_size]).long().to(device)
        with ctx:
            zq, _, _, stats = model.encode(x)
            h = model.decode_hidden(zq, ids_in=x)
            ce = model.chunked_cross_entropy(h, x, chunk_tokens=256)
            preds_chunks = []
            h_flat = h.reshape(-1, model.d)
            for c0 in range(0, h_flat.shape[0], 256):
                preds_chunks.append(model._project_logits(h_flat[c0 : c0 + 256]).argmax(dim=-1))
            preds = torch.cat(preds_chunks, dim=0).view_as(x)

        total_ce += float(ce.item()) * x.numel()
        total_correct += int((preds == x).sum().item())
        total_tokens += x.numel()
        all_preds.append(preds.cpu().numpy())
        util_accum += np.asarray(stats["batch_util"])
        ppl_accum += np.asarray(stats["perplexity"])
        n_batches += 1

    pred_cat = np.concatenate(all_preds, axis=0).reshape(-1)
    sample_gt = tokenizer.decode(val_chunks[0].tolist(), skip_special_tokens=False) if tokenizer else ""
    sample_rec = tokenizer.decode(all_preds[0][0].tolist(), skip_special_tokens=False) if tokenizer else ""

    return {
        "val_ce": total_ce / max(total_tokens, 1),
        "exact_token_acc": total_correct / max(total_tokens, 1),
        "escape_rate": 1.0 - (total_correct / max(total_tokens, 1)),
        "decoded_H": unigram_entropy_bits(pred_cat),
        "batch_util": (util_accum / max(n_batches, 1)).tolist(),
        "perplexity": (ppl_accum / max(n_batches, 1)).tolist(),
        "sample_gt": sample_gt,
        "sample_rec": sample_rec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train corpcol/ncm-codec Stage 1 Codec")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=Path, default=Path("weights/codec_4x_best.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--dq", type=int, default=256)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--mode", type=str, choices=("ar", "nar"), default="ar")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")
    if args.self_test:
        run_self_test(device)
        return

    if args.data_dir is None and args.hf_dataset is None:
        raise SystemExit("Pass --data-dir or --hf-dataset (or --self-test)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    E = base.get_input_embeddings().weight.detach().float().clone()
    del base

    if args.data_dir is not None:
        train_chunks, val_chunks = load_packed_chunks_from_dir(args.data_dir, tokenizer, L=args.L)
    else:
        train_chunks, val_chunks = load_packed_chunks_from_hf(
            args.hf_dataset, args.text_column, tokenizer, L=args.L, max_docs=args.max_docs
        )
    print(f"Loaded {len(train_chunks)} train chunks, {len(val_chunks)} val chunks (L={args.L})")

    model = Codec(
        E=E,
        L=args.L,
        r=args.r,
        d=args.d,
        dq=args.dq,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        heads=args.heads,
        depth=args.depth,
        K=args.K,
        mode=args.mode,
    ).to(device)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_compatible_state_dict(ckpt["state"])
        print(f"Resumed weights from {args.resume}")

    sample_x = torch.from_numpy(train_chunks[: min(4, len(train_chunks))]).long().to(device)
    ratio = model.position_signal_ratio(sample_x)
    print(f"[Step 0 Check] tok_rms / pos_rms = {ratio:.2f} (requires >= 5.0)")
    assert ratio >= 5.0, f"Position signal dominates token signal (ratio={ratio:.2f})"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    opt = build_optimizer(model, lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_bf16 = device.type in ("cuda", "xpu")
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    best_ce = float("inf")
    t0 = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum):
            idx = np.random.randint(0, len(train_chunks), size=args.batch_size)
            x = torch.from_numpy(train_chunks[idx]).long().to(device)
            with ctx:
                zq, _, commit_loss, _ = model.encode(x)
                h = model.decode_hidden(zq, ids_in=x)
                ce_loss = model.chunked_cross_entropy(h, x, chunk_tokens=256)
                loss = (ce_loss + commit_loss) / args.grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, val_chunks, args.batch_size, device, use_bf16, tokenizer=tokenizer
            )
            elapsed = time.perf_counter() - t0
            eta_s = (elapsed / step) * (args.steps - step)
            util_str = ", ".join(f"{u*100:.1f}%" for u in metrics["batch_util"])
            ppl_str = ", ".join(f"{p:.1f}" for p in metrics["perplexity"])
            vram_mib = (
                torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
            )
            print(
                f"step={step:5d}/{args.steps} | val_ce={metrics['val_ce']:.4f} | "
                f"acc={metrics['exact_token_acc']*100:.2f}% | "
                f"H={metrics['decoded_H']:.2f}b | util=[{util_str}] | ppl=[{ppl_str}] | "
                f"vram={vram_mib:.0f}MiB | ETA={eta_s/60:.1f}m"
            )
            print(f"  GT : {metrics['sample_gt'][:120]!r}")
            print(f"  REC: {metrics['sample_rec'][:120]!r}")

            if metrics["val_ce"] < best_ce:
                best_ce = metrics["val_ce"]
                cfg = {
                    "V": model.V,
                    "dt": model.dt,
                    "L": model.L,
                    "r": model.r,
                    "d": model.d,
                    "dq": model.dq,
                    "enc_layers": args.enc_layers,
                    "dec_layers": args.dec_layers,
                    "heads": args.heads,
                    "depth": args.depth,
                    "K": args.K,
                    "mode": args.mode,
                    "pos_scale": model.pos_scale,
                }
                torch.save({"cfg": cfg, "vocab": None, "state": model.state_dict()}, args.output)
                print(f"  -> saved best checkpoint to {args.output} (val_ce={best_ce:.4f})")


if __name__ == "__main__":
    main()
