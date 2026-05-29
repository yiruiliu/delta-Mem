"""
Initialize delta-Mem adapter parameters from data-aware activation subspaces.

Core idea (from Swift-SVD applied to memory):
  The write/read projections of delta-Mem currently start from random or
  zero initialization and are learned via SFT.  But the *direction* of
  these projections can be set analytically: we want each memory dimension
  to capture a genuinely informative direction of the hidden-state space,
  not a random one.

  Given V_h[:, :R]  (top-R principal directions of hidden activations),
  we initialize:

    memory_k_proj  [R, H]  ← V_h[:, :R].T   (project h → top-R memory write keys)
    memory_v_proj  [R, H]  ← V_h[:, :R].T   (project h → top-R memory write values)
    memory_q_proj  [R, H]  ← V_h[:, :R].T   (project h → top-R memory read queries)

  For the delta output projections (how memory readout maps back to attention):

    delta_o_proj   [o_out, R]  ← V_h[:, :R]  × online_gain
        (o_out == hidden_size for Qwen3; same subspace as hidden states)

    delta_q_proj   [q_out, R]  ← V_q[:, :R]  × online_gain
        (use Q-space subspace if available; fall back to V_h if q_out == H)

  Parameters that control *timing* (beta_proj) and disabled heads (delta_k/v)
  keep their original zero initialization — we only change *direction*, not
  the gating logic.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from deltamem.data_aware.subspace_collector import LayerSubspaces
from deltamem.core.delta_impl import DeltaMemAttention, iter_delta_mem_modules

logger = logging.getLogger(__name__)


def init_delta_mem_from_subspaces(
    model: torch.nn.Module,
    subspaces: Dict[int, LayerSubspaces],
    *,
    online_gain: Optional[float] = None,
    init_memory_proj: bool = True,
    init_delta_q: bool = True,
    init_delta_o: bool = True,
    scale_by_energy: bool = False,
) -> Dict[int, float]:
    """
    Re-initialize the memory projections of every DeltaMemAttention layer
    using the data-aware subspaces collected by collect_layer_subspaces().

    Call this AFTER attach_delta_mem() and BEFORE SFT fine-tuning.

    Args:
        model:            Model with DeltaMemAttention layers already attached.
        subspaces:        Per-layer subspaces from collect_layer_subspaces().
        online_gain:      Scale factor for delta output projections.
                          None → use the value stored in each layer's config.
        init_memory_proj: Whether to re-init memory_{q,k,v}_proj.
        init_delta_q:     Whether to re-init delta_q_proj.
        init_delta_o:     Whether to re-init delta_o_proj.
        scale_by_energy:  If True, scale each direction by its relative energy
                          (singular value / sum), amplifying the most important
                          directions.  Default False (keep unit vectors).

    Returns:
        Dict mapping layer_idx → fraction of hidden-state variance captured
        by the initialized rank-R subspace (diagnostic).
    """
    energy_fractions: Dict[int, float] = {}

    for name, module in iter_delta_mem_modules(model):
        layer_idx = module.layer_idx
        if layer_idx not in subspaces:
            logger.debug("Layer %d not in subspaces, skipping.", layer_idx)
            continue

        sub = subspaces[layer_idx]
        gain = online_gain if online_gain is not None else module.online_gain
        rank = module.rank
        state_read_dim = module.state_read_dim  # = rank * num_state_heads

        # ------------------------------------------------------------------
        # 1. Memory write/read projection initialization
        #    memory_{q,k,v}_proj shape: [state_read_dim, hidden_size]
        #    We want each row to be one principal direction of h.
        # ------------------------------------------------------------------
        if init_memory_proj:
            V_h_r = _get_top_r(sub.V_h, sub.S_h, state_read_dim, scale_by_energy)
            # V_h_r: [H, R]  →  need [R, H]
            mem_proj = V_h_r.T.contiguous()  # [R, H]
            _check_shape(mem_proj, module.memory_k_proj, "memory_k_proj", layer_idx)
            _check_shape(mem_proj, module.memory_q_proj, "memory_q_proj", layer_idx)
            _check_shape(mem_proj, module.memory_v_proj, "memory_v_proj", layer_idx)

            with torch.no_grad():
                module.memory_k_proj.data.copy_(mem_proj.to(module.memory_k_proj.dtype))
                module.memory_v_proj.data.copy_(mem_proj.to(module.memory_v_proj.dtype))
                module.memory_q_proj.data.copy_(mem_proj.to(module.memory_q_proj.dtype))

            energy_fractions[layer_idx] = _energy(sub.S_h, state_read_dim)
            logger.info(
                "Layer %3d | memory_proj init  rank=%d  energy=%.3f",
                layer_idx, state_read_dim, energy_fractions[layer_idx],
            )

        # ------------------------------------------------------------------
        # 2. delta_q_proj initialization
        #    Shape: [q_out, state_read_dim]
        #    Maps memory readout (R-dim) → q correction (q_out-dim).
        #    Use Q-space subspace; fall back to hidden subspace if dims match.
        # ------------------------------------------------------------------
        if init_delta_q and "q" in module.active_delta_heads:
            q_out = module.delta_q_proj.shape[0]
            # Choose subspace
            if sub.V_q.shape[0] == q_out:
                V_q_r = _get_top_r(sub.V_q, sub.S_q, state_read_dim, scale_by_energy)
            elif sub.V_h.shape[0] == q_out:
                logger.debug(
                    "Layer %d: using V_h for delta_q_proj (q_out == hidden_size)", layer_idx
                )
                V_q_r = _get_top_r(sub.V_h, sub.S_h, state_read_dim, scale_by_energy)
            else:
                logger.warning(
                    "Layer %d: cannot init delta_q_proj — q_out=%d, V_q.shape=%s, V_h.shape=%s",
                    layer_idx, q_out, tuple(sub.V_q.shape), tuple(sub.V_h.shape),
                )
                V_q_r = None

            if V_q_r is not None:
                # V_q_r: [q_out, R]  (already in output space)
                delta_q_init = V_q_r * gain  # [q_out, R]
                _check_shape(delta_q_init, module.delta_q_proj, "delta_q_proj", layer_idx)
                with torch.no_grad():
                    module.delta_q_proj.data.copy_(delta_q_init.to(module.delta_q_proj.dtype))
                logger.info("Layer %3d | delta_q_proj init  gain=%.4f", layer_idx, gain)

        # ------------------------------------------------------------------
        # 3. delta_o_proj initialization
        #    Shape: [o_out, state_read_dim]  where o_out == hidden_size.
        #    Maps memory readout (R-dim) → o correction (hidden_size-dim).
        #    Use hidden-state subspace (same space as residual stream output).
        # ------------------------------------------------------------------
        if init_delta_o and "o" in module.active_delta_heads:
            o_out = module.delta_o_proj.shape[0]
            if sub.V_h.shape[0] == o_out:
                V_o_r = _get_top_r(sub.V_h, sub.S_h, state_read_dim, scale_by_energy)
                delta_o_init = V_o_r * gain  # [o_out, R]
                _check_shape(delta_o_init, module.delta_o_proj, "delta_o_proj", layer_idx)
                with torch.no_grad():
                    module.delta_o_proj.data.copy_(delta_o_init.to(module.delta_o_proj.dtype))
                logger.info("Layer %3d | delta_o_proj init  gain=%.4f", layer_idx, gain)
            else:
                logger.warning(
                    "Layer %d: cannot init delta_o_proj — o_out=%d != hidden_size=%d",
                    layer_idx, o_out, sub.V_h.shape[0],
                )

        # beta_proj / lambda_proj / delta_k_proj / delta_v_proj:
        # keep zero init — they control write timing and disabled heads.

    return energy_fractions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_top_r(
    V: torch.Tensor,
    S: torch.Tensor,
    r: int,
    scale_by_energy: bool,
) -> torch.Tensor:
    """
    Return the top-r columns of V (shape [D, r]), optionally scaled by
    relative singular values so more important directions have larger norm.
    """
    r_actual = min(r, V.shape[1])
    V_r = V[:, :r_actual].float()  # [D, r]

    if r_actual < r:
        # Pad with zeros if we have fewer components than requested
        pad = torch.zeros(V.shape[0], r - r_actual, dtype=V_r.dtype)
        V_r = torch.cat([V_r, pad], dim=1)

    if scale_by_energy:
        s = S[:r_actual].float()
        s_norm = s / (s.sum() + 1e-12)  # relative energy weights
        if r_actual < r:
            s_norm = F.pad(s_norm, (0, r - r_actual))
        V_r = V_r * s_norm.unsqueeze(0)

    return V_r  # [D, r]


def _check_shape(tensor: torch.Tensor, param: torch.nn.Parameter, name: str, layer_idx: int) -> None:
    if tensor.shape != param.shape:
        raise RuntimeError(
            f"Layer {layer_idx}: shape mismatch for {name}. "
            f"Computed {tuple(tensor.shape)}, parameter is {tuple(param.shape)}."
        )


def _energy(S: torch.Tensor, r: int) -> float:
    total = (S.float() ** 2).sum().item()
    kept = (S.float()[:r] ** 2).sum().item()
    return kept / (total + 1e-12)
