"""Pairwise preference GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from botorch.models import PairwiseGP as BoTorchPairwiseGP
from botorch.models.likelihoods.pairwise import PairwiseLikelihood
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import ScaleKernel
from torch import Tensor

from robotorchan.models.base import ModelTrainingMixin, RawDataMixin


class PairwiseGP(RawDataMixin, ModelTrainingMixin, BoTorchPairwiseGP):
    """BoTorch ``PairwiseGP`` with robotorchan preference-model conventions.

    Predictive behavior, Laplace approximation, duplicate consolidation,
    conditioning, and transforms are delegated to BoTorch. robotorchan only
    adds retention of caller-supplied preference data and ``make_mll()``.
    """

    supports_mll = True

    def __init__(
        self,
        datapoints: Tensor | None,
        comparisons: Tensor | None,
        likelihood: PairwiseLikelihood | None = None,
        covar_module: ScaleKernel | None = None,
        input_transform: InputTransform | None = None,
        *,
        jitter: float = 1e-6,
        xtol: float | None = None,
        consolidate_rtol: float = 0.0,
        consolidate_atol: float = 1e-4,
        maxfev: int | None = None,
    ) -> None:
        """Initialize the preference model while retaining caller raw tensors."""
        raw_datapoints = None if datapoints is None else datapoints.detach().clone()
        raw_comparisons = None if comparisons is None else comparisons.detach().clone()

        super().__init__(
            datapoints=datapoints,
            comparisons=comparisons,
            likelihood=likelihood,
            covar_module=covar_module,
            input_transform=input_transform,
            jitter=jitter,
            xtol=xtol,
            consolidate_rtol=consolidate_rtol,
            consolidate_atol=consolidate_atol,
            maxfev=maxfev,
        )
        self._store_raw_tensor("datapoints", raw_datapoints)
        self._store_raw_tensor("comparisons", raw_comparisons)

    @property
    def raw_datapoints(self) -> Tensor | None:
        """Caller-supplied preference datapoints before preprocessing."""
        return self._get_raw_tensor("datapoints")

    @property
    def raw_comparisons(self) -> Tensor | None:
        """Caller-supplied pairwise comparisons before consolidation."""
        return self._get_raw_tensor("comparisons")

    def make_mll(self) -> PairwiseLaplaceMarginalLogLikelihood:
        """Construct the Laplace-approximated marginal log likelihood."""
        return PairwiseLaplaceMarginalLogLikelihood(self.likelihood, self)
