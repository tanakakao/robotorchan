"""BoTorch-compatible objectives for decision-time input perturbations."""

from __future__ import annotations

from botorch.acquisition.risk_measures import (\n    CVaR,\n    Expectation,\n    RiskMeasureMCObjective,\n    VaR,\n    WorstCase,\n)

from .risk import RiskType


def make_input_perturbation_objective(
    risk_type: RiskType,
    *,
    n_w: int,
    alpha: float = 0.9,
) -> RiskMeasureMCObjective:
    """Build an MC objective that groups n_w perturbed scenarios per candidate."""
    if n_w < 1:
        raise ValueError("n_w must be positive.")
    if risk_type == "expectation":
        return Expectation(n_w=n_w)
    if risk_type == "worst_case":
        return WorstCase(n_w=n_w)
    if risk_type == "var":
        return VaR(alpha=alpha, n_w=n_w)
    if risk_type == "cvar":
        return CVaR(alpha=alpha, n_w=n_w)
    if risk_type == "mean_variance":
        raise ValueError(
            "mean_variance is not yet certified for the InputPerturbation MC path."
        )
    if risk_type == "sn_ratio":
        raise ValueError("sn_ratio is not yet certified for the InputPerturbation MC path.")
    raise ValueError(f"Unsupported risk_type: {risk_type}")
