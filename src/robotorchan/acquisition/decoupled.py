"""Finite-decision-set knowledge gradient for independently observed outputs."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.sampling.list_sampler import ListSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class qDecoupledKnowledgeGradient(AcquisitionFunction):
    """Masked fantasy KG with a fixed terminal decision set and linear utility.

    The terminal value is the maximum posterior expected utility on decision_X.
    Weights select an objective or define a scalarization. Optional independent
    Gaussian upper-bound constraints use a specified infeasible/abstain utility.
    Pending evaluations are not supported.
    """

    def __init__(
        self,
        model: ModelListGP,
        decision_X: Tensor,
        weights: Tensor,
        sampler: ListSampler,
        X_evaluation_mask: Tensor,
        *,
        constraint_indices: tuple[int, ...] = (),
        constraint_thresholds: Tensor | None = None,
        infeasible_utility: float | None = None,
    ) -> None:
        super().__init__(model=model)
        if not isinstance(model, ModelListGP) or len(model.models) != model.num_outputs:
            raise ValueError("requires one independent GP per output")
        if decision_X.ndim != 2 or decision_X.shape[0] == 0:
            raise ValueError("decision_X must be a nonempty two-dimensional tensor")
        if weights.shape != (model.num_outputs,) or not torch.isfinite(weights).all():
            raise ValueError("weights must be finite with one entry per output")
        if not isinstance(sampler, ListSampler) or len(sampler.samplers) != model.num_outputs:
            raise ValueError("requires one fantasy sampler per output")
        if len(sampler.sample_shape) != 1:
            raise ValueError("requires a single fantasy sample dimension")
        if len(set(constraint_indices)) != len(constraint_indices) or any(
            i < 0 or i >= model.num_outputs for i in constraint_indices
        ):
            raise ValueError("constraint indices must be unique valid output indices")
        if constraint_indices:
            if (
                constraint_thresholds is None
                or constraint_thresholds.shape != (len(constraint_indices),)
                or not torch.isfinite(constraint_thresholds).all()
                or infeasible_utility is None
                or not torch.isfinite(weights.new_tensor(infeasible_utility))
            ):
                raise ValueError("constraints require finite thresholds and infeasible utility")
            if (weights[list(constraint_indices)] != 0).any():
                raise ValueError("constraint outputs must have zero objective weights")
        elif constraint_thresholds is not None or infeasible_utility is not None:
            raise ValueError("constraint indices are required for constrained utility")
        self.constraint_indices = constraint_indices
        self.infeasible_utility = infeasible_utility
        self.register_buffer("constraint_thresholds", constraint_thresholds)
        self.sampler = sampler
        self.register_buffer("decision_X", decision_X)
        self.register_buffer("weights", weights)
        self.register_buffer("X_evaluation_mask", X_evaluation_mask)
        self._validate_mask(X_evaluation_mask)
        with torch.no_grad():
            current = self._terminal_value(model, decision_X)
        self.register_buffer("current_value", current)

    def _terminal_value(self, model: ModelListGP, X: Tensor) -> Tensor:
        posterior = model.posterior(X)
        utility = (posterior.mean * self.weights).sum(dim=-1)
        if self.constraint_indices:
            indices = list(self.constraint_indices)
            mean = posterior.mean[..., indices]
            variance = posterior.variance[..., indices]
            std = variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt()
            z = (self.constraint_thresholds - mean) / std
            probability = torch.special.ndtr(z).prod(dim=-1)
            fallback = utility.new_tensor(self.infeasible_utility)
            utility = fallback + probability * (utility - fallback)
            return torch.maximum(utility.amax(dim=-1), fallback)
        return utility.amax(dim=-1)

    def _validate_mask(self, mask: Tensor) -> None:
        if (
            mask.ndim != 2
            or mask.shape[-1] != self.model.num_outputs
            or mask.dtype != torch.bool
            or not mask.any(dim=-1).all()
        ):
            raise ValueError("evaluation mask must select at least one output per candidate")

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None:
            raise NotImplementedError("pending evaluations are not supported")
        super().set_X_pending(None)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mask = self.X_evaluation_mask
        self._validate_mask(mask)
        if mask.shape[0] != X.shape[-2]:
            raise ValueError("evaluation mask must match candidate q")
        fantasy = self.model.fantasize(X=X, sampler=self.sampler, evaluation_mask=mask)
        decision = self.decision_X.expand(*X.shape[:-2], *self.decision_X.shape)
        return self._terminal_value(fantasy, decision).mean(dim=0) - self.current_value
