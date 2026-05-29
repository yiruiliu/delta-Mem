"""
Data-Aware delta-Mem Initialization Script
==========================================

Full pipeline:
  1. Load a frozen base model (Qwen3-4B / 8B or SmolLM3-3B)
  2. Load calibration data (long-context / memory-task distribution)
  3. Collect per-layer activation subspaces via forward hooks (Swift-SVD method)
  4. Attach delta-Mem adapters to the model
  5. Re-initialize memory read/write projections from the collected subspaces
  6. Save the initialized adapter for subsequent SFT fine-tuning

After running this script, pass the saved adapter directory as the starting
point to the standard SFT script (run_qasper_multimodel_write8192_train_and_benchmark_suite.sh)
by setting  DELTA_MEM_INIT_ADAPTER_DIR=<output_dir>.

Usage:
    python scripts/run_data_aware_init.py \
        --model-path /root/huggingface/hub/Qwen3-4B-Instruct-2507 \
        --calib-file /root/data/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
        --output-dir /root/models/delta_mem_data_aware_init_rank8 \
        --rank 8 \
        --calib-samples 512 \
        --calib-seqlen 2048 \
        --write-granularity token \
        --delta-heads q,o \
        [--scale-by-energy]  # optional: weight directions by singular values
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
    p.add_argument("--calib-file", required=True,
                   help="JSONL calibration file. Each line should be a JSON object "
                        "with a 'messages' field (list of {role, content}) or raw 'text'.")
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

def main() -> None:
    args = parse_args()

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
    # 3. Collect data-aware subspaces from the FROZEN base model
    #    (before attaching delta-Mem, so hooks target vanilla attention)
    # ------------------------------------------------------------------
    logger.info("=== Phase 1: Collecting activation subspaces ===")
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
        device=str(device),
        show_progress=True,
        log_stats=True,
    )
    logger.info("Collected subspaces for %d layers.", len(subspaces))

    # ------------------------------------------------------------------
    # 4. Attach delta-Mem adapters to the model
    # ------------------------------------------------------------------
    logger.info("=== Phase 2: Attaching delta-Mem adapters ===")
    replaced = attach_delta_mem(model, config)
    logger.info("Replaced %d attention modules.", len(replaced))

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.2fM", trainable / 1e6)

    # ------------------------------------------------------------------
    # 5. Re-initialize memory projections from data-aware subspaces
    # ------------------------------------------------------------------
    logger.info("=== Phase 3: Data-aware initialization ===")
    energy_fractions = init_delta_mem_from_subspaces(
        model,
        subspaces,
        online_gain=args.online_gain,
        init_memory_proj=True,
        init_delta_q=not args.no_init_delta_q,
        init_delta_o=not args.no_init_delta_o,
        scale_by_energy=args.scale_by_energy,
    )

    # Summary statistics
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
        "model_path": args.model_path,
        "calib_file": args.calib_file,
        "calib_samples": args.calib_samples,
        "calib_seqlen": args.calib_seqlen,
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
