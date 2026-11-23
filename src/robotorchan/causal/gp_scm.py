"""Known-DAG GP structural causal model built from native BoTorch mechanisms.

This is not a BoTorch Model: its propagated interventional distribution is
generally non-Gaussian and is exposed through Tensor samples, not a fabricated
Gaussian Posterior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from botorch.posteriors import Posterior
from torch import Tensor

from robotorchan.causal.data import CausalObservation
from robotorchan.causal.graph import CausalGraph
from robotorchan.causal.intervention import HardIntervention
from robotorchan.models.standard.single_task import SingleTaskGP


class GPStructuralCausalModel:
    """Learn node mechanisms on a known, causally sufficient DAG.

    Non-root mechanisms are native BoTorch SingleTaskGPs conditioned on their
    declared parents. Root nodes use empirical Normal distributions; constant
    roots remain deterministic. Each mechanism learns only from records where
    its *own node* was not intervened on, including observations under
    interventions on other nodes. Incomplete records are omitted per mechanism.

    The model assumes causal sufficiency, invariant structural mechanisms,
    suitable support for all evaluated parent values, and independent
    exogenous noise. It does not infer or validate the causal DAG.
    """

    def __init__(
        self,
        graph: CausalGraph,
        observations: Iterable[CausalObservation],
        *,
        dtype: torch.dtype = torch.double,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(graph, CausalGraph):
            raise TypeError("graph must be a CausalGraph.")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be a floating-point torch dtype.")

        records = tuple(observations)
        for record in records:
            if not isinstance(record, CausalObservation):
                raise TypeError("observations must contain CausalObservation records.")
            if set(record.values) - set(graph.nodes):
                raise ValueError("Observations contain nodes absent from the causal graph.")
            if record.intervention is not None:
                if set(record.intervention.values) - set(graph.nodes):
                    raise ValueError("Interventions contain nodes absent from the causal graph.")

        self.graph = graph
        self.observations = records
        self.dtype = dtype
        self.device = torch.device("cpu" if device is None else device)
        self._mechanisms: dict[str, SingleTaskGP] = {}
        self._roots: dict[str, tuple[Tensor, Tensor]] = {}
        self._fitted = False

    @property
    def mechanisms(self) -> Mapping[str, SingleTaskGP]:
        """Read-only registry of fitted *non-root* native BoTorch models."""
        self._require_fit()
        return MappingProxyType(self._mechanisms)

    def fit(self, *, optimizer_kwargs: dict[str, Any] | None = None) -> None:
        """Fit native GPs and estimate root marginal distributions.

        Requires at least two unmanipulated complete-case records per
        mechanism. A failed refit leaves no fitted model available.
        """
        self._fitted = False
        self._mechanisms = {}
        self._roots = {}
        fitted: dict[str, SingleTaskGP] = {}
        roots: dict[str, tuple[Tensor, Tensor]] = {}
        for node in self.graph.topological_order:
            parents = self.graph.parents_of(node)
            usable = [
                record.values
                for record in self.observations
                if (record.intervention is None or node not in record.intervention.values)
                and node in record.values
                and all(parent in record.values for parent in parents)
            ]
            if len(usable) < 2:
                raise ValueError(
                    f"Node {node!r} needs at least two complete observations "
                    "without intervention on that node."
                )

            if not parents:
                values = torch.tensor(
                    [row[node] for row in usable], dtype=self.dtype, device=self.device
                )
                mean = values.mean()
                std = values.std(unbiased=True)
                roots[node] = (mean, std)
                continue

            train_x = torch.tensor(
                [[row[parent] for parent in parents] for row in usable],
                dtype=self.dtype,
                device=self.device,
            )
            train_y = torch.tensor(
                [[row[node]] for row in usable], dtype=self.dtype, device=self.device
            )
            model = SingleTaskGP(train_X=train_x, train_Y=train_y)
            model.fit(optimizer_kwargs=optimizer_kwargs)
            model.eval()
            fitted[node] = model

        self._roots = roots
        self._mechanisms = fitted
        self._fitted = True

    def mechanism_posterior(
        self,
        node: str,
        X: Tensor,
        *,
        observation_noise: bool = False,
    ) -> Posterior:
        """Return the native GP posterior of a non-root conditional mechanism.

        X has shape [..., q, len(graph.parents_of(node))] in parent order.
        For an intervention distribution, use sample_intervention(s) instead.
        """
        self._require_fit()
        if node not in self.graph.nodes:
            raise ValueError(f"Unknown causal node: {node!r}.")
        if not self.graph.parents_of(node):
            raise ValueError("Root nodes have empirical distributions, not GP posteriors.")
        self._check_tensor(X, len(self.graph.parents_of(node)))
        return self._mechanisms[node].posterior(X, observation_noise=observation_noise)

    def sample_intervention(
        self,
        intervention: HardIntervention,
        *,
        outcome: str,
        sample_shape: torch.Size = torch.Size([128]),
        observation_noise: bool = False,
    ) -> Tensor:
        """Sample scalar outcomes under one do-intervention.

        Returns shape sample_shape (e.g. [1024]), *not* a BoTorch Posterior.
        """
        if not isinstance(intervention, HardIntervention):
            raise TypeError("intervention must be a HardIntervention.")
        nodes = tuple(intervention.values)
        candidates = torch.tensor(
            [[intervention.values[node] for node in nodes]],
            dtype=self.dtype,
            device=self.device,
        )
        samples = self.sample_interventions(
            candidates,
            intervention_nodes=nodes,
            outcome=outcome,
            sample_shape=sample_shape,
            observation_noise=observation_noise,
        )
        return samples.squeeze(-1)

    def sample_interventions(
        self,
        X: Tensor,
        *,
        intervention_nodes: Sequence[str],
        outcome: str,
        sample_shape: torch.Size = torch.Size([128]),
        observation_noise: bool = False,
    ) -> Tensor:
        """Propagate posterior samples of do(intervention_nodes = X).

        X is [..., q, d_S], with columns in intervention_nodes order. Samples
        are [*sample_shape, ..., q], preserving Tensor-based differentiation
        through X. This is MC propagation, not a Gaussian approximation or
        BoTorch Model.posterior; q>1 here is evaluation, not q-batch BO.
        """
        self._require_fit()
        if outcome not in self.graph.nodes:
            raise ValueError("outcome must belong to the causal graph.")
        if isinstance(intervention_nodes, (str, bytes)):
            raise TypeError("intervention_nodes must be an ordered sequence.")
        nodes = tuple(intervention_nodes)
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("intervention_nodes must be nonempty and distinct.")
        if any(node not in self.graph.nodes for node in nodes):
            raise ValueError("intervention_nodes must belong to the causal graph.")
        self._check_tensor(X, len(nodes))
        try:
            shape = torch.Size(sample_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_shape must be a nonempty torch.Size.") from exc
        if not shape or any(n < 1 for n in shape):
            raise ValueError("sample_shape must have positive dimensions.")

        prefix = shape + X.shape[:-1]
        fixed = {
            node: X[..., column].expand(prefix)
            for column, node in enumerate(nodes)
        }
        # Only ancestors required to predict the outcome; a hard intervention
        # blocks the edges entering its target.
        required: set[str] = set()

        def include(node: str) -> None:
            if node in required:
                return
            required.add(node)
            if node not in fixed:
                for parent in self.graph.parents_of(node):
                    include(parent)

        include(outcome)
        values: dict[str, Tensor] = {}
        for node in self.graph.topological_order:
            if node not in required:
                continue
            if node in fixed:
                values[node] = fixed[node]
            elif node in self._roots:
                mean, std = self._roots[node]
                values[node] = (
                    torch.distributions.Normal(mean, std).rsample(prefix)
                    if float(std) > 0.0
                    else mean.expand(prefix)
                )
            else:
                parent_values = torch.stack(
                    [values[parent] for parent in self.graph.parents_of(node)], dim=-1
                )
                posterior = self._mechanisms[node].posterior(
                    parent_values, observation_noise=observation_noise
                )
                values[node] = posterior.rsample().squeeze(-1)
            if not bool(torch.isfinite(values[node]).all()):
                raise ValueError(f"Non-finite propagated values for node {node!r}.")
        return values[outcome]

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the GP structural causal model before prediction.")

    def _check_tensor(self, X: Tensor, dim: int) -> None:
        if not isinstance(X, Tensor):
            raise TypeError("X must be a torch.Tensor.")
        if X.ndim < 2 or X.shape[-2] < 1 or X.shape[-1] != dim:
            raise ValueError(f"X must have shape [..., q, {dim}] with q >= 1.")
        if X.dtype != self.dtype or X.device != self.device:
            raise ValueError("X dtype and device must match the fitted model.")
        if not bool(torch.isfinite(X).all()):
            raise ValueError("X contains non-finite intervention values.")
