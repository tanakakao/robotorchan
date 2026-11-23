"""MC expected improvement for a fitted, known-DAG GP-SCM.

Only the acquisition function is adapted to BoTorch. GP-SCM itself is not a
BoTorch Model and its interventional samples are not a Gaussian Posterior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.causal.gp_scm import GPStructuralCausalModel
from robotorchan.causal.proposal import CausalAcquisitionProposal, optimize_causal_acqfs
from robotorchan.optim.dispatch import OptimizerName


class _GPScmMCLogEI(AcquisitionFunction):
    """Fixed-seed MC log(EI) over q=1 GP-SCM intervention candidates."""

    def __init__(
        self,
        scm: GPStructuralCausalModel,
        *,
        nodes: tuple[str, ...],
        outcome: str,
        best_f: float,
        maximize: bool,
        mc_samples: int,
        mc_seed: int,
    ) -> None:
        # The GP-SCM is not a BoTorch Model. This is a genuine acquisition
        # function over its Tensor samples, not a fake Model/Posterior adapter.
        super().__init__(model=None)
        self.scm = scm
        self.nodes = nodes
        self.outcome = outcome
        self.best_f = best_f
        self.maximize = maximize
        self.mc_samples = mc_samples
        self.mc_seed = mc_seed
        # optimize_causal_acqfs reads the acquisition's registered dtype/device.
        self.register_buffer(
            "_dtype_marker", torch.empty(0, dtype=scm.dtype, device=scm.device)
        )

    def forward(self, X: Tensor) -> Tensor:
        """Return log MC-EI per t-batch, using the same RNG stream per call."""
        if X.ndim < 2 or X.shape[-2] != 1 or X.shape[-1] != len(self.nodes):
            raise ValueError("GP-SCM MC acquisition expects X[..., 1, intervention_dim].")
        devices = [self.scm.device] if self.scm.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.mc_seed)
            samples = self.scm.sample_interventions(
                X,
                intervention_nodes=self.nodes,
                outcome=self.outcome,
                sample_shape=torch.Size([self.mc_samples]),
            )
        improvement = samples - self.best_f if self.maximize else self.best_f - samples
        estimated_ei = improvement.clamp_min(0.0).mean(dim=0).squeeze(-1)
        return estimated_ei.clamp_min(torch.finfo(estimated_ei.dtype).tiny).log()


def propose_gp_scm_intervention(
    model: GPStructuralCausalModel,
    intervention_sets: Sequence[Sequence[str]],
    bounds: Mapping[str, tuple[float, float]],
    *,
    outcome: str,
    best_f: float,
    direction: str = "maximize",
    intervention_costs: Mapping[frozenset[str], float] | None = None,
    optimizer: OptimizerName = "botorch",
    num_restarts: int = 5,
    raw_samples: int = 64,
    mc_samples: int = 128,
    mc_seed: int = 0,
) -> CausalAcquisitionProposal:
    """Optimize MC-EI per intervention set and rank by log(EI / set cost).

    The supplied best_f must come from a genuinely *interventional* outcome,
    not an observational value presumed to represent a do-intervention.
    This standalone method does not require CausalBOStudy.
    """
    if not isinstance(model, GPStructuralCausalModel):
        raise TypeError("model must be a GPStructuralCausalModel.")
    model.mechanisms  # Ensure fit() has completed, including root-only models.
    if outcome not in model.graph.nodes:
        raise ValueError("outcome must belong to the causal graph.")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'.")
    if isinstance(best_f, bool) or not isfinite(float(best_f)):
        raise ValueError("best_f must be a finite interventional outcome.")
    if mc_samples < 1 or not isinstance(mc_seed, int) or mc_seed < 0:
        raise ValueError("mc_samples must be positive and mc_seed nonnegative.")

    groups: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for raw_nodes in intervention_sets:
        if isinstance(raw_nodes, (str, bytes)):
            raise TypeError("Each intervention set must be a sequence of node names.")
        selected = tuple(raw_nodes)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Intervention sets must be nonempty without duplicates.")
        key = frozenset(selected)
        if outcome in key or not key.issubset(model.graph.nodes):
            raise ValueError("Intervention sets must contain known non-outcome nodes.")
        if key in seen:
            raise ValueError("Each intervention set must appear only once.")
        seen.add(key)
        groups.append(selected)
    if not groups:
        raise ValueError("At least one intervention set is required.")

    costs = {} if intervention_costs is None else dict(intervention_costs)
    if any(not isinstance(key, frozenset) or key not in seen for key in costs):
        raise ValueError("intervention_costs must identify configured sets.")
    if any(
        isinstance(cost, bool) or not isfinite(float(cost)) or cost <= 0.0
        for cost in costs.values()
    ):
        raise ValueError("intervention costs must be positive finite numbers.")

    acquisitions = {
        nodes: _GPScmMCLogEI(
            model,
            nodes=nodes,
            outcome=outcome,
            best_f=float(best_f),
            maximize=direction == "maximize",
            mc_samples=mc_samples,
            mc_seed=mc_seed,
        )
        for nodes in groups
    }
    return optimize_causal_acqfs(
        acquisitions,
        bounds,
        score=lambda nodes, value: value - log(float(costs.get(frozenset(nodes), 1.0))),
        optimizer=optimizer,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
