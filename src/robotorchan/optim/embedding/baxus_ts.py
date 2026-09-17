"""Thompson-sampling candidate generation for BAxUS."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.generation.sampling import MaxPosteriorSampling
from torch import Tensor
from torch.quasirandom import SobolEngine

from robotorchan.optim.base import SearchResult
from robotorchan.optim.embedding.baxus import BAxUSStrategy


class BAxUSThompsonSamplingStrategy(BAxUSStrategy):
    """Generate BAxUS candidates with sparse trust-region Thompson sampling."""

    def __init__(self, *args, n_candidates: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if n_candidates is not None and n_candidates < 1:
            raise ValueError("n_candidates must be at least 1.")
        self._n_candidates = n_candidates

    @property
    def n_candidates(self) -> int:
        """Return the explicit pool size or the current target-space default."""
        if self._n_candidates is not None:
            return self._n_candidates
        return min(5000, max(2000, 200 * self.target_dim))

    def _candidate_pool(self, target_bounds: Tensor) -> tuple[Tensor, Tensor]:
        """Draw sparse perturbations around the current target-space incumbent."""
        sobol_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                device=self.bounds.device,
                generator=self._generator,
            ).item()
        )
        sobol = SobolEngine(self.target_dim, scramble=True, seed=sobol_seed)
        unit = sobol.draw(self.n_candidates).to(self.bounds)
        perturbations = target_bounds[0] + (target_bounds[1] - target_bounds[0]) * unit

        probability = min(20.0 / self.target_dim, 1.0)
        mask = (
            torch.rand(
                self.n_candidates,
                self.target_dim,
                dtype=self.bounds.dtype,
                device=self.bounds.device,
                generator=self._generator,
            )
            <= probability
        )
        empty_rows = torch.where(mask.sum(dim=-1) == 0)[0]
        if empty_rows.numel() > 0:
            forced_dims = torch.randint(
                self.target_dim,
                (empty_rows.numel(),),
                device=self.bounds.device,
                generator=self._generator,
            )
            mask[empty_rows, forced_dims] = True

        target_pool = self.target_center.expand(self.n_candidates, self.target_dim).clone()
        target_pool[mask] = perturbations[mask]
        return target_pool, self.project(target_pool)

    @staticmethod
    def _match_target_candidates(
        selected: Tensor,
        input_pool: Tensor,
        target_pool: Tensor,
    ) -> Tensor:
        distances = torch.cdist(selected, input_pool)
        indices = distances.argmin(dim=-1)
        return target_pool.index_select(0, indices)

    def optimize(self, acq_function: AcquisitionFunction, *, q: int = 1) -> SearchResult:
        if q < 1:
            raise ValueError("q must be at least 1.")
        if q > self.n_candidates:
            raise ValueError("q must not exceed n_candidates.")
        if self.state.restart_triggered:
            raise RuntimeError("BAxUS subspace expansion is required before optimization.")

        target_weights = self._target_lengthscale_weights(acq_function)
        target_bounds = self._target_bounds(target_weights)
        target_pool, input_pool = self._candidate_pool(target_bounds)
        sampler = MaxPosteriorSampling(model=acq_function.model, replacement=False)
        with torch.no_grad():
            candidates = sampler(input_pool, num_samples=q)
            acquisition_value = acq_function(candidates)
        target_candidates = self._match_target_candidates(candidates, input_pool, target_pool)

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={
                "target_candidates": target_candidates.detach(),
                "target_dim": self.target_dim,
                "target_center": self.target_center.detach().clone(),
                "target_lengthscale_weights": target_weights.detach().clone(),
                "target_bounds": target_bounds.detach().clone(),
                "trust_region_length": self.state.length,
                "failure_tolerance": self.state.failure_tolerance,
                "split_budget": self.state.split_budget,
                "embedding": self.embedding.detach().clone(),
                "candidate_generation": "thompson_sampling",
                "n_candidates": self.n_candidates,
            },
        )
