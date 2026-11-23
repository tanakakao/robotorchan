"""Sparse-ARD priors for high-dimensional multiclass and ordinal GP classifiers.

These classes reuse the binary-classification kernel factories. Training uses
the existing variational fit methods; no MCMC inference is performed.
"""

from __future__ import annotations

from torch import Tensor

from robotorchan.models.classification.binary.high_dimensional.map_saas import (
    MapSaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.high_dimensional.saas import (
    SaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)


class SaasMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass variational GP with independent class-wise SAAS-style ARD priors."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        global_scale: float = 0.1,
    ) -> None:
        if global_scale <= 0.0:
            raise ValueError("global_scale must be positive.")
        covar_module = SaasBinarySingleTaskGPClassifier._make_saas_covar_module(
            train_X, global_scale
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=covar_module)
        self.global_scale = float(global_scale)


class SaasOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal-probit variational GP with a SAAS-style sparse ARD prior."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        global_scale: float = 0.1,
    ) -> None:
        if global_scale <= 0.0:
            raise ValueError("global_scale must be positive.")
        covar_module = SaasBinarySingleTaskGPClassifier._make_saas_covar_module(
            train_X, global_scale
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=covar_module)
        self.global_scale = float(global_scale)


class MapSaasMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass variational GP using the existing MAP-SAAS-style kernel factory."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        tau: float = 0.1,
    ) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        covar_module = MapSaasBinarySingleTaskGPClassifier._make_sparse_covar_module(
            train_X, tau
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=covar_module)
        self.tau = float(tau)


class MapSaasOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal-probit variational GP using the MAP-SAAS-style kernel factory."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        tau: float = 0.1,
    ) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        covar_module = MapSaasBinarySingleTaskGPClassifier._make_sparse_covar_module(
            train_X, tau
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=covar_module)
        self.tau = float(tau)
