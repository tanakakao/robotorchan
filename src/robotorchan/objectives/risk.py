"""Risk and quality-engineering aggregation over scenario outcomes."""

from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch import Tensor

RiskType = Literal["expectation", "mean_variance", "worst_case", "var", "cvar", "sn_ratio"]
SNType = Literal["larger_is_better", "smaller_is_better", "nominal_is_best"]


class RiskMeasure(Protocol):
    """Aggregate a scenario axis into one robust objective value."""

    def __call__(self, values: Tensor) -> Tensor:
        """Aggregate the final tensor dimension containing scenarios."""


class Expectation:
    """Expected response across scenarios."""

    def __call__(self, values: Tensor) -> Tensor:
        return values.mean(dim=-1)


class MeanVariance:
    """Mean response penalized by scenario variance."""

    def __init__(self, risk_weight: float = 1.0) -> None:
        if risk_weight < 0:
            raise ValueError("risk_weight must be non-negative.")
        self.risk_weight = risk_weight

    def __call__(self, values: Tensor) -> Tensor:
        return values.mean(dim=-1) - self.risk_weight * values.var(dim=-1, unbiased=False)


class WorstCase:
    """Worst response for a maximization objective."""

    def __call__(self, values: Tensor) -> Tensor:
        return values.min(dim=-1).values


class VaR:
    """Lower-tail value at risk for a maximization objective."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = _validate_alpha(alpha)

    def __call__(self, values: Tensor) -> Tensor:
        return torch.quantile(values, 1.0 - self.alpha, dim=-1)


class CVaR:
    """Mean of the lower tail beyond VaR for a maximization objective."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = _validate_alpha(alpha)

    def __call__(self, values: Tensor) -> Tensor:
        threshold = torch.quantile(values, 1.0 - self.alpha, dim=-1, keepdim=True)
        mask = values <= threshold
        count = mask.sum(dim=-1).clamp_min(1)
        return (values * mask).sum(dim=-1) / count


class SNRatio:
    """Taguchi signal-to-noise ratio across environmental scenarios."""

    def __init__(
        self,
        sn_type: SNType,
        *,
        target: float | Tensor | None = None,
        eps: float = 1e-12,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if sn_type == "nominal_is_best" and target is None:
            raise ValueError("target is required for nominal_is_best.")
        if sn_type != "nominal_is_best" and target is not None:
            raise ValueError("target is only valid for nominal_is_best.")
        self.sn_type = sn_type
        self.target = target
        self.eps = eps

    def __call__(self, values: Tensor) -> Tensor:
        if self.sn_type == "larger_is_better":
            loss = values.square().clamp_min(self.eps).reciprocal().mean(dim=-1)
        elif self.sn_type == "smaller_is_better":
            loss = values.square().mean(dim=-1)
        elif self.sn_type == "nominal_is_best":
            target = torch.as_tensor(self.target, dtype=values.dtype, device=values.device)
            loss = (values - target).square().mean(dim=-1)
        else:
            raise ValueError(f"Unsupported sn_type: {self.sn_type}")
        return -10.0 * torch.log10(loss.clamp_min(self.eps))


def make_risk_measure(
    risk_type: RiskType,
    *,
    alpha: float = 0.9,
    risk_weight: float = 1.0,
    sn_type: SNType = "larger_is_better",
    target: float | Tensor | None = None,
) -> RiskMeasure:
    """Build a scenario aggregation from a compact robust-BO configuration."""
    if risk_type == "expectation":
        return Expectation()
    if risk_type == "mean_variance":
        return MeanVariance(risk_weight=risk_weight)
    if risk_type == "worst_case":
        return WorstCase()
    if risk_type == "var":
        return VaR(alpha=alpha)
    if risk_type == "cvar":
        return CVaR(alpha=alpha)
    if risk_type == "sn_ratio":
        return SNRatio(sn_type=sn_type, target=target)
    raise ValueError(f"Unsupported risk_type: {risk_type}")


def _validate_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")
    return alpha
