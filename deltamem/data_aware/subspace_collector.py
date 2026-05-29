"""
Collect per-layer activation subspaces from a frozen LLM via forward hooks.

For each attention layer we track:
  - hidden_states:  the residual-stream input to the attention block     → V_h [H, H]
  - q_proj output:  the raw Q activations before reshape/RoPE            → V_q [q_out, q_out]

These subspaces are then used by init_from_subspace.py to set the initial
direction of delta-Mem's memory read/write projections.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel

from deltamem.data_aware.incremental_svd import IncrementalSVD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class LayerSubspaces:
    """
    Holds the principal subspace for one transformer layer.

    V_h:  [hidden_size, hidden_size]  — principal directions of residual stream
    S_h:  [hidden_size]               — corresponding singular values (sqrt eigenvalues)
    V_q:  [q_out, q_out]              — principal directions of Q-projection output
    S_q:  [q_out]
    ner_h: normalized effective rank of hidden states
    ner_q: normalized effective rank of Q outputs
    energy_h: fraction of hidden-state variance in top-rank directions
    """
    layer_idx: int
    V_h: torch.Tensor
    S_h: torch.Tensor
    V_q: torch.Tensor
    S_q: torch.Tensor
    ner_h: float = 0.0
    ner_q: float = 0.0


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def _make_input_hook(svd: IncrementalSVD):
    """Hook that feeds hidden_states input of an attn module into SVD.

    Registered via register_forward_pre_hook(with_kwargs=True).
    Called as hook(module, args, kwargs).
    Newer transformers (5.x) passes most inputs as kwargs, so we check both.
    """
    def hook(module: nn.Module, args, kwargs):
        # Try positional args first, then 'hidden_states' kwarg
        if args and args[0] is not None:
            x = args[0]
        else:
            x = kwargs.get("hidden_states", None)
        if x is not None:
            svd.feed(x)
    return hook


def _make_output_hook(svd: IncrementalSVD):
    """Hook that feeds the OUTPUT of a module into SVD."""
    def hook(module: nn.Module, inp, output):  # noqa: ARG001
        x = output[0] if isinstance(output, tuple) else output
        if x is not None:
            svd.feed(x)
    return hook


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

def collect_layer_subspaces(
    model: PreTrainedModel,
    calib_inputs: List[Dict[str, torch.Tensor]],
    *,
    target_layer_indices: Optional[List[int]] = None,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
    log_stats: bool = True,
) -> Dict[int, LayerSubspaces]:
    """
    Run calibration data through a frozen model, collecting per-layer
    activation subspaces for hidden states and Q projections.

    Args:
        model:               Frozen base model (Qwen3 / SmolLM3).
        calib_inputs:        List of tokenized batches, each a dict with
                             at least "input_ids" (and optionally "attention_mask").
                             Sequences should cover the target distribution
                             (e.g. long-context dialogues for memory tasks).
        target_layer_indices: Which layers to profile. None = all layers.
        device:              GPU device for accumulation.
        show_progress:       Show tqdm bar over calibration batches.
        log_stats:           Log NER and energy-retention per layer.

    Returns:
        Dict mapping layer_idx → LayerSubspaces.
    """
    model.eval()
    layers = _get_attention_layers(model)

    if target_layer_indices is None:
        target_layer_indices = list(range(len(layers)))

    # ------------------------------------------------------------------
    # Build SVD accumulators + register hooks
    # ------------------------------------------------------------------
    svd_h: Dict[int, IncrementalSVD] = {}   # hidden-state SVDs
    svd_q: Dict[int, IncrementalSVD] = {}   # q-proj-output SVDs
    hooks = []

    for idx in target_layer_indices:
        layer = layers[idx]
        attn = _get_attn_module(layer)
        hidden_size = attn.q_proj.in_features
        q_out = attn.q_proj.out_features

        svd_h[idx] = IncrementalSVD(dim=hidden_size, device=device, name=f"hidden_L{idx}")
        svd_q[idx] = IncrementalSVD(dim=q_out, device=device, name=f"q_out_L{idx}")

        # Hook hidden states: capture INPUT to the attention sub-module
        # with_kwargs=True: newer transformers passes inputs as kwargs
        hooks.append(attn.register_forward_pre_hook(_make_input_hook(svd_h[idx]), with_kwargs=True))
        # Hook Q outputs: capture OUTPUT of q_proj linear
        hooks.append(attn.q_proj.register_forward_hook(_make_output_hook(svd_q[idx])))

    # ------------------------------------------------------------------
    # Forward passes (no grad)
    # ------------------------------------------------------------------
    iterator = tqdm(calib_inputs, desc="Collecting subspaces", disable=not show_progress)
    with torch.no_grad():
        for batch in iterator:
            batch_gpu = {k: v.to(model.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
            try:
                model(**batch_gpu, use_cache=False)
            except Exception as exc:
                logger.warning("Skipping batch due to error: %s", exc)

    # Remove all hooks
    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------
    # Finalize SVDs and build LayerSubspaces
    # ------------------------------------------------------------------
    result: Dict[int, LayerSubspaces] = {}
    for idx in target_layer_indices:
        V_h, S_h = svd_h[idx].finalize()
        V_q, S_q = svd_q[idx].finalize()
        ner_h = svd_h[idx].normalized_effective_rank()
        ner_q = svd_q[idx].normalized_effective_rank()

        result[idx] = LayerSubspaces(
            layer_idx=idx,
            V_h=V_h, S_h=S_h,
            V_q=V_q, S_q=S_q,
            ner_h=ner_h,
            ner_q=ner_q,
        )

        if log_stats:
            e_h = _energy(S_h, min(8, S_h.numel()))
            e_q = _energy(S_q, min(8, S_q.numel()))
            logger.info(
                "Layer %3d | NER_h=%.3f  energy_h@8=%.3f | NER_q=%.3f  energy_q@8=%.3f",
                idx, ner_h, e_h, ner_q, e_q,
            )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attention_layers(model: PreTrainedModel):
    """Return the list of transformer blocks (works for Qwen3 / SmolLM3)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        "Cannot locate transformer layers. "
        "Expected model.model.layers (Qwen3/LLaMA style)."
    )


def _get_attn_module(layer):
    """Return the self_attn sub-module of a transformer block."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    raise ValueError(f"Layer {layer} has no .self_attn attribute.")


def _energy(S: torch.Tensor, r: int) -> float:
    total = (S.float() ** 2).sum().item()
    kept = (S.float()[:r] ** 2).sum().item()
    return kept / (total + 1e-12)
