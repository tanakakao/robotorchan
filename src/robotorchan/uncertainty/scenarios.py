"""Scenario generators for input and environmental uncertainty."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ScenarioGenerator(ABC):
    """Generate uncertain realizations around a batch of design points."""

    @abstractmethod
    def sample(self, X: Tensor, n_w: int) -> Tensor:
        """Return scenarios with a scenario axis before the feature axis."""


class GaussianPerturbation(ScenarioGenerator):
    """Independent Gaussian perturbations on selected continuous dimensions."""

    def __init__(self, std: Tensor | float, dims: list[int] | None = None) -> None:
        self.std = std
        self.dims = dims

    def sample(self, X: Tensor, n_w: int) -> Tensor:
        """Sample Gaussian perturbations in the original input space."""
        _validate_n_w(n_w)
        dims = _normalize_dims(self.dims, X.shape[-1])
        std = torch.as_tensor(self.std, dtype=X.dtype, device=X.device)
        if std.numel() not in (1, len(dims)):
            raise ValueError("std must be scalar or match the number of perturbed dims.")
        scenarios = _expand_scenarios(X, n_w)
        shape = (*X.shape[:-1], n_w, len(dims))
        scenarios[..., dims] += torch.randn(shape, dtype=X.dtype, device=X.device) * std
        return scenarios


class UniformPerturbation(ScenarioGenerator):
    """Independent bounded perturbations on selected continuous dimensions."""

    def __init__(self, half_width: Tensor | float, dims: list[int] | None = None) -> None:
        self.half_width = half_width
        self.dims = dims

    def sample(self, X: Tensor, n_w: int) -> Tensor:
        """Sample symmetric uniform perturbations in the original input space."""
        _validate_n_w(n_w)
        dims = _normalize_dims(self.dims, X.shape[-1])
        width = torch.as_tensor(self.half_width, dtype=X.dtype, device=X.device)
        if width.numel() not in (1, len(dims)):
            raise ValueError("half_width must be scalar or match the perturbed dims.")
        scenarios = _expand_scenarios(X, n_w)
        shape = (*X.shape[:-1], n_w, len(dims))
        noise = 2 * torch.rand(shape, dtype=X.dtype, device=X.device) - 1
        scenarios[..., dims] += noise * width
        return scenarios


class CorrelatedGaussianPerturbation(ScenarioGenerator):
    """Correlated Gaussian perturbations on selected continuous dimensions."""

    def __init__(self, covariance: Tensor, dims: list[int] | None = None) -> None:
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix.")
        self.covariance = covariance.detach().clone()
        self.dims = dims

    def sample(self, X: Tensor, n_w: int) -> Tensor:
        """Sample correlated Gaussian perturbations."""
        _validate_n_w(n_w)
        dims = _normalize_dims(self.dims, X.shape[-1])
        if self.covariance.shape[0] != len(dims):
            raise ValueError("covariance size must match the number of perturbed dims.")
        covariance = self.covariance.to(dtype=X.dtype, device=X.device)
        mean = torch.zeros(len(dims), dtype=X.dtype, device=X.device)
        distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)
        noise = distribution.sample((*X.shape[:-1], n_w))
        scenarios = _expand_scenarios(X, n_w)
        scenarios[..., dims] += noise
        return scenarios


class EmpiricalScenarios(ScenarioGenerator):
    """Attach empirical environmental scenarios to controllable design points."""

    def __init__(self, values: Tensor, environmental_dims: list[int]) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [n_scenarios, n_environmental_dims].")
        if values.shape[0] < 1:
            raise ValueError("values must contain at least one scenario.")
        if values.shape[1] != len(environmental_dims):
            raise ValueError("values width must match environmental_dims.")
        self.values = values.detach().clone()
        self.environmental_dims = environmental_dims

    def sample(self, X: Tensor, n_w: int) -> Tensor:
        """Sample empirical environmental rows and place them into X."""
        _validate_n_w(n_w)
        dims = _normalize_dims(self.environmental_dims, X.shape[-1])
        values = self.values.to(dtype=X.dtype, device=X.device)
        indices = torch.randint(values.shape[0], (*X.shape[:-1], n_w), device=X.device)
        scenarios = _expand_scenarios(X, n_w)
        scenarios[..., dims] = values[indices]
        return scenarios


def _expand_scenarios(X: Tensor, n_w: int) -> Tensor:
    return X.unsqueeze(-2).expand(*X.shape[:-1], n_w, X.shape[-1]).clone()


def _validate_n_w(n_w: int) -> None:
    if n_w < 1:
        raise ValueError("n_w must be at least 1.")


def _normalize_dims(dims: list[int] | None, input_dim: int) -> list[int]:
    normalized = list(range(input_dim)) if dims is None else list(dims)
    if not normalized:
        raise ValueError("At least one dimension must be selected.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("dims must not contain duplicates.")
    if any(dim < 0 or dim >= input_dim for dim in normalized):
        raise ValueError("dims contains an out-of-range dimension.")
    return normalized
