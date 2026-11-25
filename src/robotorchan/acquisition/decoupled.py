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
    Weights select an objective or define a scalarization. No pending evaluations
    or outcome constraints are supported by this acquisition.
    """

    def __init__(
        self,
        model: ModelListGP,
        decision_X: Tensor,
        weights: Tensor,
        sampler: ListSampler,
        X_evaluation_mask: Tensor,
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
        self.sampler = sampler
        self.register_buffer("decision_X", decision_X)
        self.register_buffer("weights", weights)
        self.register_buffer("X_evaluation_mask", X_evaluation_mask)
        self._validate_mask(X_evaluation_mask)
        with torch.no_grad():
            current = (model.posterior(decision_X).mean * weights).sum(dim=-1).amax(dim=-1)
        self.register_buffer("current_value", current)

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
        utility = (fantasy.posterior(decision).mean * self.weights).sum(dim=-1)
        return utility.amax(dim=-1).mean(dim=0) - self.current_value
