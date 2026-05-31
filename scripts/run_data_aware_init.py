"""
Data-Aware delta-Mem Initialization Script
==========================================

Supports two initialization modes (--init-mode):

  data_aware (default):
    Full pipeline using activation statistics from calibration data.
    1. Load frozen base model
    2. Run calibration data through model, collect hidden-state subspaces
    3. Attach delta-Mem adapters
    4. Initialize memory projections from activation subspaces
    5. Save adapter

  naive_svd (ablation):
    Initialize directly from model weight matrices — no data required.
    1. Load frozen base model
    2. Attach delta-Mem adapters
    3. For each layer, SVD-decompose W_Q / W_K / W_V / W_O
    4. Use top-r right singular vectors of W_Q → memory_q_proj
                                           W_K → memory_k_proj
                                           W_V → memory_v_proj
       Use top-r left singular vectors  of W_Q → delta_q_proj
                                           W_O → delta_o_proj
    5. Save adapter

Usage (data_aware):
    python scripts/run_data_aware_init.py \
        --model-path /path/to/Qwen3-4B-Instruct-2507 \
        --calib-file /path/to/qasper.jsonl \
        --output-dir /path/to/output \
        --init-mode data_aware

Usage (naive_svd):
    python scripts/run_data_aware_init.py \
        --model-path /path/to/Qwen3-4B-Instruct-2507 \
        --output-dir /path/to/output \
        --init-mode naive_svd
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltamem.core import HFDeltaMemConfig, attach_delta_mem, save_delta_mem_adapter
from deltamem.data_aware import collect_layer_subspaces, init_delta_mem_from_subspaces
from deltamem.data_aware.init_from_subspace import init_delta_mem_swift_svd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Data-aware delta-Mem adapter initialization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True,
                   help="Path to the frozen base model (HuggingFace format)")
    p.add_argument("--calib-file", default=None,
                   help="JSONL calibration file (required for data_aware mode). "
                        "Each line: {'messages': [...]} or {'text': '...'}")
    p.add_argument("--init-mode", default="data_aware",
                   choices=["data_aware", "naive_svd", "swift_svd"],
                   help="data_aware: original impl — hidden-state subspace only (V_h). "
                        "swift_svd: correct Swift-SVD impl — memory_k/q/v_proj = V_r.T @ W. "
                        "naive_svd: SVD directly on weight matrices, no data needed.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to save the initialized delta-Mem adapter")

    # delta-Mem config
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--num-state-heads", type=int, default=1,
                   help="1=TSW/SSW, 4=MSW")
    p.add_argument("--delta-heads", type=str, default="q,o",
                   help="Comma-separated subset of {q,k,v,o}")
    p.add_argument("--write-granularity", type=str, default="token",
                   choices=["token", "message_mean", "sentence_mean"])
    p.add_argument("--online-gain", type=float, default=0.05)
    p.add_argument("--beta-bias-init", type=float, default=-1.5)

    # Calibration settings
    p.add_argument("--calib-samples", type=int, default=512,
                   help="Number of calibration sequences to use")
    p.add_argument("--calib-seqlen", type=int, default=2048,
                   help="Token length to truncate each calibration sequence to")
    p.add_argument("--calib-batch-size", type=int, default=1,
                   help="Batch size during calibration forward passes")
    p.add_argument("--seed", type=int, default=42)

    # Init options
    p.add_argument("--scale-by-energy", action="store_true",
                   help="Scale each subspace direction by its relative singular value")
    p.add_argument("--memory-v-scale", type=float, default=None,
                   help="Scale factor applied only to memory_v_proj init. "
                        "memory_q/k_proj are unaffected (tanh+L2norm makes their scale irrelevant). "
                        "Recommended: set to --online-gain (0.05) to prevent IFEval regression "
                        "caused by unit-norm memory_v_proj writing over-large values into S.")
    p.add_argument("--no-init-delta-q", action="store_true",
                   help="Skip data-aware init for delta_q_proj")
    p.add_argument("--no-init-delta-o", action="store_true",
                   help="Skip data-aware init for delta_o_proj")

    # Runtime
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    p.add_argument("--hf-cache-dir", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calibration data loading
# ---------------------------------------------------------------------------

def load_calib_batches(
    calib_file: str,
    tokenizer,
    *,
    max_samples: int,
    seqlen: int,
    batch_size: int,
    seed: int,
    device: torch.device,
):
    """
    Load JSONL calibration file and return a list of tokenized batches.

    Each line is expected to be one of:
      - {"messages": [{"role": "...", "content": "..."}]}   — chat format
      - {"text": "..."}                                      — raw text
    """
    logger.info("Loading calibration data from %s", calib_file)
    records = []
    with open(calib_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:max_samples]
    logger.info("Using %d calibration records", len(records))

    batches = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        texts = []
        for rec in chunk:
            if "messages" in rec:
                try:
                    text = tokenizer.apply_chat_template(
                        rec["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                except Exception:
                    text = " ".join(m.get("content", "") for m in rec["messages"])
            elif "text" in rec:
                text = rec["text"]
            else:
                # Try to concatenate all string values
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
            texts.append(text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=seqlen,
            padding="max_length",
        )
        batches.append({k: v.to(device) for k, v in enc.items()})

    logger.info("Created %d calibration batches (batch_size=%d)", len(batches), batch_size)
    return batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def naive_svd_init(model, config: "HFDeltaMemConfig") -> dict:
    """
    Initialize delta-Mem memory projections from SVD of attention weight matrices.

    For each transformer layer with delta-Mem attached:
      memory_q/k/v_proj  ← top-r right singular vectors of W_Q / W_K / W_V
      delta_q_proj        ← top-r left  singular vectors of W_Q  (if q in delta_heads)
      delta_o_proj        ← top-r left  singular vectors of W_O  (if o in delta_heads)

    Returns a dict of per-layer energy fractions captured by rank-r approximation.
    """
    rank = config.rank
    energy_fractions = {}
    n_layers = 0

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        # Skip layers where delta-Mem was not attached
        if not hasattr(attn, "memory_q_proj"):
            continue

        device = attn.memory_q_proj.device
        dtype  = attn.memory_q_proj.dtype

        def _top_r_svd(W: torch.Tensor, r: int):
            """Return (U_r, Sigma_r, Vt_r) and energy fraction for W."""
            W_f = W.float()
            U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
            energy = (S[:r].pow(2).sum() / S.pow(2).sum()).item()
            return U[:, :r], S[:r], Vh[:r, :], energy

        # W_Q: shape [q_out, hidden]  →  right SV → memory_q_proj [r, hidden]
        W_Q = attn.base.q_proj.weight.data   # [q_out, hidden]
        _, _, Vt_Q, e_q = _top_r_svd(W_Q, rank)
        attn.memory_q_proj.data.copy_(Vt_Q.to(device=device, dtype=dtype))

        # W_K: shape [kv_out, hidden]
        W_K = attn.base.k_proj.weight.data
        _, _, Vt_K, e_k = _top_r_svd(W_K, rank)
        attn.memory_k_proj.data.copy_(Vt_K.to(device=device, dtype=dtype))

        # W_V: shape [kv_out, hidden]
        W_V = attn.base.v_proj.weight.data
        _, _, Vt_V, e_v = _top_r_svd(W_V, rank)
        attn.memory_v_proj.data.copy_(Vt_V.to(device=device, dtype=dtype))

        # W_Q left SVs → delta_q_proj [hidden, r]  (if q in active heads)
        if hasattr(attn, "delta_q_proj") and "q" in config.delta_heads:
            U_Q, _, _, _ = _top_r_svd(W_Q, rank)
            attn.delta_q_proj.data.copy_(U_Q.to(device=device, dtype=dtype))

        # W_O left SVs → delta_o_proj [hidden, r]  (if o in active heads)
        if hasattr(attn, "delta_o_proj") and "o" in config.delta_heads:
            W_O = attn.base.o_proj.weight.data   # [hidden, v_out]
            U_O, _, _, _ = _top_r_svd(W_O, rank)
            attn.delta_o_proj.data.copy_(U_O[:, :rank].to(device=device, dtype=dtype))

        avg_e = (e_q + e_k + e_v) / 3
        energy_fractions[layer_idx] = avg_e
        logger.info("Layer %2d | naive SVD init  energy_avg@%d=%.4f  (q=%.3f k=%.3f v=%.3f)",
                    layer_idx, rank, avg_e, e_q, e_k, e_v)
        n_layers += 1

    logger.info("Naive SVD init done for %d layers.", n_layers)
    return energy_fractions


def main() -> None:
    args = parse_args()

    if args.init_mode == "data_aware" and args.calib_file is None:
        raise ValueError("--calib-file is required for --init-mode data_aware")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # 1. Load base model (frozen)
    # ------------------------------------------------------------------
    logger.info("Loading base model: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        cache_dir=args.hf_cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.hf_cache_dir,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("Model loaded.  Parameters: %.2fB", sum(p.numel() for p in model.parameters()) / 1e9)

    # ------------------------------------------------------------------
    # 2. Build delta-Mem config
    # ------------------------------------------------------------------
    config = HFDeltaMemConfig(
        rank=args.rank,
        alpha=args.alpha,
        num_state_heads=args.num_state_heads,
        delta_heads=tuple(h.strip() for h in args.delta_heads.split(",")),
        memory_write_granularity=args.write_granularity,
        online_gain=args.online_gain,
        beta_bias_init=args.beta_bias_init,
        couple_lambda=True,
        state_update_mode="standard",
        output_init="base_slice_fixed",   # keep original non-data-aware init for delta heads
        base_slice_ref_width=8,
        rankwise_gates=True,
        normalize_qk=True,
        memory_readout_mode="delta",
        memory_write_source="learned_hidden",
        target_modules=("self_attn",),
        target_layers=(),
    )
    logger.info("delta-Mem config: rank=%d  state_heads=%d  delta_heads=%s  granularity=%s",
                args.rank, args.num_state_heads, args.delta_heads, args.write_granularity)

    # ------------------------------------------------------------------
    # 3. Collect subspaces BEFORE attaching delta-Mem
    #    (hooks must target the raw q/k/v_proj, not DeltaMemAttention.base)
    # ------------------------------------------------------------------
    subspaces = None
    if args.init_mode in ("data_aware", "swift_svd"):
        collect_kv = (args.init_mode == "swift_svd")
        mode_label = "swift_svd Q/K/V output" if collect_kv else "data_aware hidden-state"
        logger.info("=== Phase 1: Collecting %s subspaces (before attach) ===", mode_label)
        calib_batches = load_calib_batches(
            args.calib_file,
            tokenizer,
            max_samples=args.calib_samples,
            seqlen=args.calib_seqlen,
            batch_size=args.calib_batch_size,
            seed=args.seed,
            device=device,
        )
        subspaces = collect_layer_subspaces(
            model,
            calib_batches,
            collect_kv=collect_kv,
            device=str(device),
            show_progress=True,
            log_stats=True,
        )
        logger.info("Collected subspaces for %d layers.", len(subspaces))

    # ------------------------------------------------------------------
    # 4. Attach delta-Mem adapters
    # ------------------------------------------------------------------
    logger.info("=== Phase 2: Attaching delta-Mem adapters ===")
    replaced = attach_delta_mem(model, config)
    logger.info("Replaced %d attention modules.", len(replaced))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.2fM", trainable / 1e6)

    # ------------------------------------------------------------------
    # 5a. DATA-AWARE (original): hidden-state subspace only
    # ------------------------------------------------------------------
    if args.init_mode == "data_aware":
        logger.info("=== Phase 3 (data_aware): Initializing from hidden-state subspaces ===")
        energy_fractions = init_delta_mem_from_subspaces(
            model,
            subspaces,
            online_gain=args.online_gain,
            init_memory_proj=True,
            init_delta_q=not args.no_init_delta_q,
            init_delta_o=not args.no_init_delta_o,
            scale_by_energy=args.scale_by_energy,
            memory_v_scale=args.memory_v_scale,
        )

    # ------------------------------------------------------------------
    # 5b. SWIFT-SVD: correct — memory_proj = V_r.T @ W
    # ------------------------------------------------------------------
    elif args.init_mode == "swift_svd":
        logger.info("=== Phase 3 (swift_svd): Initializing with Swift-SVD formula ===")
        energy_fractions = init_delta_mem_swift_svd(
            model,
            subspaces,
            online_gain=args.online_gain,
            init_delta_q=not args.no_init_delta_q,
        )

    # ------------------------------------------------------------------
    # 5c. NAIVE SVD: directly from weight matrices, no data
    # ------------------------------------------------------------------
    else:
        logger.info("=== Phase 3 (naive_svd): Initializing from weight matrix SVD ===")
        energy_fractions = naive_svd_init(model, config)

    avg_energy = sum(energy_fractions.values()) / max(len(energy_fractions), 1)
    logger.info(
        "Average energy captured by rank-%d subspace across %d layers: %.3f",
        args.rank, len(energy_fractions), avg_energy,
    )

    # ------------------------------------------------------------------
    # 6. Save initialized adapter
    # ------------------------------------------------------------------
    logger.info("=== Phase 4: Saving adapter to %s ===", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    save_delta_mem_adapter(model, args.output_dir, config)
    config.save_pretrained(args.output_dir)

    # Save init metadata for reproducibility
    meta = {
        "init_mode": args.init_mode,
        "model_path": args.model_path,
        "calib_file": args.calib_file,
        "calib_samples": args.calib_samples if args.init_mode == "data_aware" else None,
        "calib_seqlen": args.calib_seqlen if args.init_mode == "data_aware" else None,
        "seed": args.seed,
        "rank": args.rank,
        "num_state_heads": args.num_state_heads,
        "delta_heads": args.delta_heads,
        "write_granularity": args.write_granularity,
        "scale_by_energy": args.scale_by_energy,
        "avg_energy_at_rank": avg_energy,
        "per_layer_energy": {str(k): round(v, 4) for k, v in energy_fractions.items()},
    }
    with open(Path(args.output_dir) / "data_aware_init_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done. Adapter saved to: %s", args.output_dir)
    logger.info(
        "\nNext step — fine-tune from this initialization:\n"
        "  DELTA_MEM_INIT_ADAPTER_DIR=%s \\\n"
        "  bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
