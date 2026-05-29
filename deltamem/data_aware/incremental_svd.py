"""
Incremental SVD for collecting data-aware activation subspaces.

Adapted from Swift-SVD (https://github.com/yiruiliu/Swift-SVD).
Core idea: accumulate X^T X online (no full buffer), then do one eigendecomposition
at the end to get the principal subspace V_r.
"""
from __future__ import annotations

import torch


class IncrementalSVD:
    """
    Accumulates activations online via X^T X, then extracts the top-r
    principal directions via eigendecomposition.

    Usage:
        svd = IncrementalSVD(dim=hidden_size, device='cuda')
        for batch in dataloader:
            svd.feed(batch_activations)   # [B, T, D] or [N, D]
        V, S = svd.finalize()             # V: [D, D], S: [D]  (sorted descending)
        V_r = V[:, :r]                    # [D, r] — top-r subspace
    """

    def __init__(
        self,
        dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        name: str = "IncrementalSVD",
    ) -> None:
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.device = torch.device(device)
        # Accumulate X^T X on GPU in float32 to avoid overflow
        self.XtX: torch.Tensor | None = torch.zeros(
            (dim, dim), dtype=torch.float32, device=self.device
        )
        self.total_samples: int = 0
        self.V: torch.Tensor | None = None   # [D, D] eigenvectors, CPU
        self.S: torch.Tensor | None = None   # [D] eigenvalues (as sqrt), CPU

    # ------------------------------------------------------------------
    # Online accumulation
    # ------------------------------------------------------------------

    def feed(self, x: torch.Tensor) -> None:
        """
        Feed a batch of activations.

        Args:
            x: Tensor of any shape with last dim == self.dim.
               Typical shapes: [B, T, D], [N, D].
        """
        assert self.XtX is not None, "Cannot feed after finalize()"
        x = x.detach().reshape(-1, self.dim).to(torch.float32)
        if x.device != self.device:
            x = x.to(self.device)
        # Accumulate  XtX += x.T @ x  in-place (efficient)
        self.XtX.addmm_(x.T, x)
        self.total_samples += x.size(0)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the principal subspace from accumulated X^T X.

        Returns:
            V: [D, D] eigenvectors sorted by eigenvalue descending (on CPU)
            S: [D] sqrt(eigenvalues) sorted descending (on CPU)
        """
        assert self.XtX is not None, "finalize() already called"

        # Symmetric eigendecomposition: X^T X = Q Λ Q^T
        eigvals, eigvecs = torch.linalg.eigh(self.XtX)

        # Sort descending
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # S[i] = sqrt(λ_i)  (i.e., singular values of X)
        S = torch.sqrt(eigvals.clamp(min=0.0))

        # Move to CPU to free GPU memory
        self.V = eigvecs.cpu()
        self.S = S.cpu()

        # Release accumulator
        del self.XtX
        self.XtX = None
        torch.cuda.empty_cache()

        return self.V, self.S

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def normalized_effective_rank(self) -> float:
        """
        NER: spectral entropy of singular values, normalized by full rank.
        Higher NER → activations span the space more uniformly.
        Lower NER → information is concentrated in a few directions.
        """
        assert self.S is not None, "Call finalize() first"
        s = self.S.float()
        s = s[s > 1e-9]
        if len(s) == 0:
            return 0.0
        p = s / s.sum()
        entropy = -(p * (p + 1e-12).log()).sum()
        erank = entropy.exp()
        return (erank / len(self.S)).item()

    def energy_retained(self, r: int) -> float:
        """Fraction of total variance captured by the top-r components."""
        assert self.S is not None, "Call finalize() first"
        total = (self.S ** 2).sum().item()
        kept = (self.S[:r] ** 2).sum().item()
        return kept / (total + 1e-12)
