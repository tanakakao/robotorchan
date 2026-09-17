"""Bayesian optimization with adaptively expanding sparse subspaces (BAxUS)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


@dataclass(frozen=True)
class BAxUSState:
    """Persistent state for adaptive subspace expansion."""

    target_dim: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 4
    best_value: float = float("-inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if self.target_dim < 1:
            raise ValueError("target_dim must be at least 1.")
        if not 0 < self.length_min <= self.length_max:
            raise ValueError("BAxUS lengths must satisfy 0 < length_min <= length_max.")
        if self.length <= 0 or self.length > self.length_max:
            raise ValueError("length must satisfy 0 < length <= length_max.")
        if self.length < self.length_min and not self.restart_triggered:
            raise ValueError("length below length_min requires restart_triggered=True.")
        if self.success_tolerance < 1 or self.failure_tolerance < 1:
            raise ValueError("success_tolerance and failure_tolerance must be at least 1.")


def update_baxus_state(
    state: BAxUSState,
    values: Tensor,
    *,
    relative_improvement: float = 1e-3,
) -> BAxUSState:
    """Update trust-region state after evaluating a BAxUS batch."""
    if values.numel() < 1:
        raise ValueError("values must contain at least one observation.")
    if relative_improvement < 0:
        raise ValueError("relative_improvement must be non-negative.")

    candidate_best = float(values.max().item())
    if math.isfinite(state.best_value):
        threshold = relative_improvement * max(1.0, abs(state.best_value))
        success = candidate_best > state.best_value + threshold
    else:
        success = True
    successes = state.success_counter + 1 if success else 0
    failures = 0 if success else state.failure_counter + 1
    length = state.length

    if successes >= state.success_tolerance:
        length = min(2.0 * length, state.length_max)
        successes = 0
    elif failures >= state.failure_tolerance:
        length = length / 2.0
        failures = 0

    return replace(
        state,
        length=length,
        success_counter=successes,
        failure_counter=failures,
        best_value=max(state.best_value, candidate_best),
        restart_triggered=length < state.length_min,
    )


class _BAxUSAcquisition(AcquisitionFunction):
    def __init__(self, acq_function: AcquisitionFunction, strategy: BAxUSStrategy) -> None:
        super().__init__(model=acq_function.model)
        self.acq_function = acq_function
        self.strategy = strategy

    def forward(self, Z: Tensor) -> Tensor:
        return self.acq_function(self.strategy.project(Z))


class BAxUSStrategy(SearchStrategy):
    """Optimize acquisitions with a sparse embedding that expands by bin splitting.

    Each original input dimension belongs to exactly one target-space bin with a
    random sign. On expansion, every splittable target bin is partitioned into
    up to ``new_bins_on_split + 1`` child bins, matching the nested BAxUS
    embedding construction.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        initial_target_dim: int = 1,
        new_bins_on_split: int = 3,
        seed: int | None = None,
        state: BAxUSState | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
    ) -> None:
        super().__init__(bounds)
        if initial_target_dim < 1 or initial_target_dim > self.input_dim:
            raise ValueError("initial_target_dim must be between 1 and input_dim.")
        if new_bins_on_split < 1:
            raise ValueError("new_bins_on_split must be at least 1.")
        if num_restarts < 1 or raw_samples < 1:
            raise ValueError("num_restarts and raw_samples must be at least 1.")
        if state is not None and state.target_dim != initial_target_dim:
            raise ValueError("state.target_dim must equal initial_target_dim.")

        self.new_bins_on_split = new_bins_on_split
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential
        self._generator = torch.Generator(device=bounds.device)
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(seed)
        self.embedding = self._new_sparse_embedding(initial_target_dim)
        self.state = BAxUSState(target_dim=initial_target_dim) if state is None else state

    @property
    def target_dim(self) -> int:
        return self.embedding.shape[-1]

    @property
    def target_bounds(self) -> Tensor:
        half = self.state.length
        return torch.stack(
            [
                torch.full(
                    (self.target_dim,),
                    -half,
                    dtype=self.bounds.dtype,
                    device=self.bounds.device,
                ),
                torch.full(
                    (self.target_dim,),
                    half,
                    dtype=self.bounds.dtype,
                    device=self.bounds.device,
                ),
            ]
        )

    def _new_sparse_embedding(self, target_dim: int) -> Tensor:
        """Create a balanced sparse signed embedding of shape ``[D, d]``."""
        assignments = torch.arange(self.input_dim, device=self.bounds.device) % target_dim
        permutation = torch.randperm(
            self.input_dim,
            device=self.bounds.device,
            generator=self._generator,
        )
        assignments = assignments[permutation]
        signs = torch.randint(
            0,
            2,
            (self.input_dim,),
            device=self.bounds.device,
            generator=self._generator,
        )
        signs = signs.to(dtype=self.bounds.dtype).mul_(2).sub_(1)
        embedding = torch.zeros(
            self.input_dim,
            target_dim,
            dtype=self.bounds.dtype,
            device=self.bounds.device,
        )
        rows = torch.arange(self.input_dim, device=self.bounds.device)
        embedding[rows, assignments] = signs
        return embedding

    def _split_embedding(self) -> Tensor:
        """Split every populated bin into nested child bins in one expansion."""
        columns = [self.embedding[:, index].clone() for index in range(self.target_dim)]
        new_columns: list[Tensor] = []

        for source, column in enumerate(columns):
            members = torch.nonzero(column != 0, as_tuple=False).flatten()
            if members.numel() <= 1:
                continue
            order = torch.randperm(
                members.numel(),
                device=self.bounds.device,
                generator=self._generator,
            )
            shuffled = members[order]
            n_groups = min(self.new_bins_on_split + 1, members.numel())
            groups = torch.tensor_split(shuffled, n_groups)
            for group in groups[1:]:
                child = torch.zeros_like(column)
                child[group] = column[group]
                columns[source][group] = 0
                new_columns.append(child)

        return torch.stack(columns + new_columns, dim=-1)

    def project(self, Z: Tensor) -> Tensor:
        """Map target-space coordinates to the feasible original-space box."""
        if Z.shape[-1] != self.target_dim:
            raise ValueError("Z last dimension must equal target_dim.")
        normalized = (Z @ self.embedding.transpose(-2, -1)).clamp(-1.0, 1.0)
        center = self.bounds.mean(dim=0)
        half_range = 0.5 * (self.bounds[1] - self.bounds[0])
        return center + half_range * normalized

    def expand_subspace(self) -> bool:
        """Expand all splittable target bins while preserving the nested embedding."""
        if not self.state.restart_triggered:
            raise RuntimeError("Subspace expansion requires restart_triggered=True.")
        if self.target_dim >= self.input_dim:
            return False

        expanded = self._split_embedding()
        if expanded.shape[-1] == self.target_dim:
            return False
        self.embedding = expanded
        self.state = BAxUSState(
            target_dim=self.target_dim,
            length=0.8,
            length_min=self.state.length_min,
            length_max=self.state.length_max,
            success_tolerance=self.state.success_tolerance,
            failure_tolerance=self.state.failure_tolerance,
            best_value=self.state.best_value,
        )
        return True

    def update_state(
        self,
        values: Tensor,
        *,
        relative_improvement: float = 1e-3,
    ) -> BAxUSState:
        self.state = update_baxus_state(
            self.state,
            values,
            relative_improvement=relative_improvement,
        )
        return self.state

    def optimize(self, acq_function: AcquisitionFunction, *, q: int = 1) -> SearchResult:
        """Optimize an original-space acquisition in the current subspace."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        if self.state.restart_triggered:
            raise RuntimeError("BAxUS subspace expansion is required before optimization.")

        embedded_acq = _BAxUSAcquisition(acq_function, self)
        Z, _ = optimize_acqf(
            acq_function=embedded_acq,
            bounds=self.target_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )
        candidates = self.project(Z)
        with torch.no_grad():
            acquisition_value = acq_function(candidates)

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={
                "target_candidates": Z.detach(),
                "target_dim": self.target_dim,
                "trust_region_length": self.state.length,
                "embedding": self.embedding.detach().clone(),
            },
        )
