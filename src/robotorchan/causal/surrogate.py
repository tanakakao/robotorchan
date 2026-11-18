"""GP surrogate for outcomes observed under a fixed hard-intervention set."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from botorch.posteriors import Posterior
from torch import Tensor

from robotorchan.causal.data import CausalObservation
from robotorchan.causal.graph import CausalGraph
from robotorchan.models.standard.single_task import SingleTaskGP


class InterventionGPSurrogate:
    """Fit one existing BoTorch GP to outcomes from a fixed intervention set.

    Only records with actual hard interventions on exactly ``intervention_nodes``
    are used. Observational records and other intervention sets are excluded.
    This is an empirical interventional response model, not an estimator of
    causal effects from observational data.

    Columns in ``train_X`` and posterior inputs follow the graph's topological
    order restricted to the selected intervention nodes. Access ``model`` to
    use the native BoTorch model with existing acquisition functions.
    """

    def __init__(
        self,
        graph: CausalGraph,
        outcome: str,
        intervention_nodes: Sequence[str],
        observations: Iterable[CausalObservation],
        *,
        dtype: torch.dtype = torch.double,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(graph, CausalGraph):
            raise TypeError("graph must be a CausalGraph.")
        if outcome not in graph.nodes:
            raise ValueError("outcome must be a node in the causal graph.")
        if isinstance(intervention_nodes, (str, bytes)):
            raise TypeError("intervention_nodes must be a sequence of node names.")
        selected = tuple(intervention_nodes)
        if not selected:
            raise ValueError("intervention_nodes must not be empty.")
        if len(set(selected)) != len(selected):
            raise ValueError("intervention_nodes must not contain duplicates.")
        if any(node not in graph.nodes for node in selected):
            raise ValueError("intervention_nodes must belong to the causal graph.")
        if outcome in selected:
            raise ValueError("The outcome cannot be intervened upon.")

        wanted = set(selected)
        self.graph = graph
        self.outcome = outcome
        self.intervention_nodes = tuple(
            node for node in graph.topological_order if node in wanted
        )

        inputs: list[list[float]] = []
        targets: list[list[float]] = []
        for record in observations:
            if not isinstance(record, CausalObservation):
                raise TypeError("observations must contain CausalObservation records.")
            intervention = record.intervention
            if intervention is None:
                continue
            if any(node not in graph.nodes for node in intervention.values):
                raise ValueError("Intervention records contain nodes absent from the graph.")
            if set(intervention.values) != wanted:
                continue
            if outcome not in record.values:
                raise ValueError("Matching intervention records must include the outcome.")
            inputs.append([intervention.values[node] for node in self.intervention_nodes])
            targets.append([record.values[outcome]])

        if len(inputs) < 2:
            raise ValueError("At least two matching interventional observations are required.")

        train_X = torch.tensor(inputs, dtype=dtype, device=device)
        train_Y = torch.tensor(targets, dtype=dtype, device=device)
        self.model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

    @property
    def train_X(self) -> Tensor:
        """Raw intervention values in canonical column order."""
        return self.model.raw_train_X

    @property
    def train_Y(self) -> Tensor:
        """Raw outcomes from matching intervention records only."""
        return self.model.raw_train_Y

    def fit(self, *, optimizer_kwargs: dict[str, Any] | None = None) -> None:
        """Delegate GP fitting to robotorchan's existing SingleTaskGP."""
        self.model.fit(optimizer_kwargs=optimizer_kwargs)
        self.model.eval()

    def posterior(self, X: Tensor, *, observation_noise: bool = False) -> Posterior:
        """Return the native BoTorch posterior for intervention values ``[..., q, d]``."""
        if X.ndim < 2 or X.shape[-1] != len(self.intervention_nodes):
            raise ValueError(
                f"X must have shape [..., q, {len(self.intervention_nodes)}] "
                "in intervention_nodes order."
            )
        return self.model.posterior(X, observation_noise=observation_noise)
