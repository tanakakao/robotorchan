"""Binary GP classification with continuous candidate-input uncertainty."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.uncertain.base import (
    ClassificationInputUncertaintyType,
    ClassificationUncertaintyIntegration,
    ClassificationUncertaintyTarget,
    UncertainBinaryClassificationMixin,
)


def _validate_candidate_uncertainty(
    X: Tensor,
    *,
    input_std: Tensor | None,
    input_covar: Tensor | None,
) -> Tensor:
    """Validate Gaussian candidate uncertainty and return covariance matrices."""
    if (input_std is None) == (input_covar is None):
        raise ValueError("Exactly one of input_std and input_covar must be supplied.")
    d = X.shape[-1]
    if input_std is not None:
        if input_std.shape != X.shape:
            raise ValueError("input_std must have the same shape as X.")
        if input_std.device != X.device or input_std.dtype != X.dtype:
            raise ValueError("input_std must share dtype and device with X.")
        if not torch.isfinite(input_std).all() or torch.any(input_std < 0):
            raise ValueError("input_std must be finite and nonnegative.")
        return torch.diag_embed(input_std.square())

    expected_shape = (*X.shape[:-1], d, d)
    if input_covar is None or input_covar.shape != expected_shape:
        raise ValueError("input_covar must have shape X.shape[:-1] + (d, d).")
    if input_covar.device != X.device or input_covar.dtype != X.dtype:
        raise ValueError("input_covar must share dtype and device with X.")
    if not torch.isfinite(input_covar).all():
        raise ValueError("input_covar must be finite.")
    if not torch.allclose(input_covar, input_covar.transpose(-1, -2)):
        raise ValueError("input_covar must be symmetric.")
    if torch.any(torch.linalg.eigvalsh(input_covar) < -1e-10):
        raise ValueError("input_covar must be positive semidefinite.")
    return input_covar


class ContinuousUncertainInputBinarySingleTaskGPClassifier(
    UncertainBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Binary GP classifier marginalizing Gaussian uncertainty at candidates."""

    @property
    def classification_input_uncertainty(
        self,
    ) -> frozenset[ClassificationInputUncertaintyType]:
        """Return continuous input uncertainty metadata."""
        return frozenset({ClassificationInputUncertaintyType.CONTINUOUS})

    @property
    def classification_uncertainty_target(self) -> ClassificationUncertaintyTarget:
        """Return candidate-input uncertainty semantics."""
        return ClassificationUncertaintyTarget.CANDIDATE_INPUTS

    @property
    def classification_uncertainty_integration(
        self,
    ) -> ClassificationUncertaintyIntegration:
        """Return Monte Carlo probability-space integration semantics."""
        return ClassificationUncertaintyIntegration.MONTE_CARLO

    def sample_uncertain_inputs(
        self,
        X: Tensor,
        *,
        input_std: Tensor | None = None,
        input_covar: Tensor | None = None,
        num_samples: int = 128,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw Gaussian perturbations around candidate inputs."""
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        covariance = _validate_candidate_uncertainty(
            X,
            input_std=input_std,
            input_covar=input_covar,
        )
        d = X.shape[-1]
        jitter = torch.finfo(X.dtype).eps
        identity = torch.eye(d, dtype=X.dtype, device=X.device)
        chol = torch.linalg.cholesky(covariance + jitter * identity)
        noise = torch.randn(
            (num_samples, *X.shape),
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        )
        perturbation = torch.matmul(chol.unsqueeze(0), noise.unsqueeze(-1)).squeeze(-1)
        return X.unsqueeze(0) + perturbation

    def predict_proba_with_input_uncertainty(
        self,
        X: Tensor,
        *,
        input_std: Tensor | None = None,
        input_covar: Tensor | None = None,
        num_samples: int = 128,
        generator: torch.Generator | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Return E_epsilon[P(y | X + epsilon, D)] by Monte Carlo integration."""
        perturbed = self.sample_uncertain_inputs(
            X,
            input_std=input_std,
            input_covar=input_covar,
            num_samples=num_samples,
            generator=generator,
        )
        probabilities = super().predict_proba(perturbed, **kwargs)
        return probabilities.mean(dim=0)
