"""Finite-decision-set knowledge gradient for independently observed outputs."""

from collections.abc import Callable, Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import ModelListGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from robotorchan.acquisition.active_learning.epig import qJointExpectedPredictiveInformationGain
from robotorchan.models.heterogeneous import HeterogeneousModel


def _target_points(X: Tensor, target_fidelities: dict[int, float] | None) -> Tensor:
    if not target_fidelities:
        return X
    if any(
        index < 0 or index >= X.shape[-1] or not torch.isfinite(X.new_tensor(value))
        for index, value in target_fidelities.items()
    ):
        raise ValueError("target fidelities require valid dimensions and finite values")
    return project_to_target_fidelity(X, d=X.shape[-1], target_fidelities=target_fidelities)


class qDecoupledKnowledgeGradient(AcquisitionFunction):
    """Masked fantasy KG with a fixed terminal decision set and linear utility.

    The terminal value is the maximum posterior expected utility on decision_X.
    Weights select an objective or define a scalarization. Optional independent
    Gaussian upper-bound constraints use a specified infeasible/abstain utility.
    Pending evaluations are not supported.
    """

    def __init__(
        self,
        model: ModelListGP,
        decision_X: Tensor,
        weights: Tensor,
        sampler: ListSampler,
        X_evaluation_mask: Tensor,
        *,
        target_fidelities: dict[int, float] | None = None,
        constraint_indices: tuple[int, ...] = (),
        constraint_thresholds: Tensor | None = None,
        infeasible_utility: float | None = None,
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
        if len(set(constraint_indices)) != len(constraint_indices) or any(
            i < 0 or i >= model.num_outputs for i in constraint_indices
        ):
            raise ValueError("constraint indices must be unique valid output indices")
        if constraint_indices:
            if (
                constraint_thresholds is None
                or constraint_thresholds.shape != (len(constraint_indices),)
                or not torch.isfinite(constraint_thresholds).all()
                or infeasible_utility is None
                or not torch.isfinite(weights.new_tensor(infeasible_utility))
            ):
                raise ValueError("constraints require finite thresholds and infeasible utility")
            if (weights[list(constraint_indices)] != 0).any():
                raise ValueError("constraint outputs must have zero objective weights")
        elif constraint_thresholds is not None or infeasible_utility is not None:
            raise ValueError("constraint indices are required for constrained utility")
        self.constraint_indices = constraint_indices
        self.infeasible_utility = infeasible_utility
        self.register_buffer("constraint_thresholds", constraint_thresholds)
        self.sampler = sampler
        decision_X = _target_points(decision_X, target_fidelities)
        self.register_buffer("decision_X", decision_X)
        self.register_buffer("weights", weights)
        self.register_buffer("X_evaluation_mask", X_evaluation_mask)
        self._validate_mask(X_evaluation_mask)
        with torch.no_grad():
            current = self._terminal_value(model, decision_X)
        self.register_buffer("current_value", current)

    def _terminal_value(self, model: ModelListGP, X: Tensor) -> Tensor:
        posterior = model.posterior(X)
        utility = (posterior.mean * self.weights).sum(dim=-1)
        if self.constraint_indices:
            indices = list(self.constraint_indices)
            mean = posterior.mean[..., indices]
            variance = posterior.variance[..., indices]
            std = variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt()
            z = (self.constraint_thresholds - mean) / std
            probability = torch.special.ndtr(z).prod(dim=-1)
            fallback = utility.new_tensor(self.infeasible_utility)
            utility = fallback + probability * (utility - fallback)
            return torch.maximum(utility.amax(dim=-1), fallback)
        return utility.amax(dim=-1)

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
        return self._terminal_value(fantasy, decision).mean(dim=0) - self.current_value


class qDecoupledExpectedPredictiveInformationGain(AcquisitionFunction):
    """Weighted joint EPIG across independent outputs and masked candidate rows.

    Output weights define the target prediction distribution. Observation noise
    is a strictly positive variance in each output's original units.
    """

    def __init__(
        self,
        model: ModelListGP,
        target_X: Tensor,
        X_evaluation_mask: Tensor,
        observation_noise: Tensor,
        output_weights: Tensor | None = None,
        *,
        target_fidelities: dict[int, float] | None = None,
        candidate_noise_functions: Sequence[Callable[[Tensor], Tensor] | None] | None = None,
        target_noise_functions: Sequence[Callable[[Tensor], Tensor] | None] | None = None,
    ) -> None:
        super().__init__(model=model)
        if not isinstance(model, ModelListGP) or len(model.models) != model.num_outputs:
            raise ValueError("requires one independent GP per output")
        m = model.num_outputs
        valid_noise = torch.isfinite(observation_noise) & (observation_noise > 0)
        if observation_noise.shape != (m,) or not valid_noise.all():
            raise ValueError("requires positive finite observation noise per output")
        if output_weights is None:
            output_weights = observation_noise.new_ones(m)
        if (
            output_weights.shape != (m,)
            or not torch.isfinite(output_weights).all()
            or (output_weights < 0).any()
            or output_weights.sum() <= 0
        ):
            raise ValueError("requires nonnegative finite output weights with positive sum")
        self.register_buffer("output_weights", output_weights / output_weights.sum())
        self.register_buffer("X_evaluation_mask", X_evaluation_mask)
        target_X = _target_points(target_X, target_fidelities)
        if candidate_noise_functions is None:
            candidate_noise_functions = [None] * m
        if target_noise_functions is None:
            target_noise_functions = [None] * m
        if len(candidate_noise_functions) != m or len(target_noise_functions) != m:
            raise ValueError("noise functions require one entry per output")
        self.components = torch.nn.ModuleList(
            qJointExpectedPredictiveInformationGain(
                child,
                target_X,
                observation_noise=observation_noise[j],
                candidate_noise_function=candidate_noise_functions[j],
                target_noise_function=target_noise_functions[j],
            )
            for j, child in enumerate(model.models)
        )

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None:
            raise NotImplementedError("pending evaluations are not supported")
        super().set_X_pending(None)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mask = self.X_evaluation_mask
        if (
            mask.shape != (X.shape[-2], self.model.num_outputs)
            or mask.dtype != torch.bool
            or not mask.any(dim=-1).all()
        ):
            raise ValueError("evaluation mask must select outputs for each candidate")
        result = X.sum(dim=(-2, -1)) * 0
        for j, component in enumerate(self.components):
            if mask[:, j].any():
                result = result + self.output_weights[j] * component(X[..., mask[:, j], :])
        return result


class qDecoupledActiveLearning(AcquisitionFunction):
    """Explicit weighted sum of per-output acquisitions on selected candidate rows.

    Weights must express a common utility scale. Components retain their own
    batch semantics (joint information or marginal mean, for example).
    This is not a joint information gain for dependent outputs.
    """

    def __init__(
        self,
        model: HeterogeneousModel,
        components: Sequence[AcquisitionFunction],
        X_evaluation_mask: Tensor,
        output_weights: Tensor,
    ) -> None:
        super().__init__(model=model)
        if not isinstance(model, HeterogeneousModel):
            raise TypeError("requires HeterogeneousModel")
        if any(count != 1 for count in model.entry_num_outputs):
            raise ValueError("requires one semantic output per model entry")
        if len(components) != model.num_outputs:
            raise ValueError("requires one acquisition per semantic output")
        for child, component in zip(model.entries, components, strict=True):
            if component.model is not child:
                raise ValueError("component must use its corresponding child model")
            if getattr(component, "X_pending", None) is not None:
                raise ValueError("component pending evaluations are not supported")
        if (
            output_weights.shape != (model.num_outputs,)
            or not torch.isfinite(output_weights).all()
            or (output_weights < 0).any()
            or not (output_weights > 0).any()
        ):
            raise ValueError("requires finite nonnegative weights with positive sum")
        self.components = torch.nn.ModuleList(components)
        self.register_buffer("output_weights", output_weights)
        self.register_buffer("X_evaluation_mask", X_evaluation_mask)
        self._validate_mask(X_evaluation_mask)

    def _validate_mask(self, mask: Tensor) -> None:
        if (
            mask.ndim != 2
            or mask.shape[-1] != self.model.num_outputs
            or mask.dtype != torch.bool
            or not mask.any(dim=-1).all()
        ):
            raise ValueError("evaluation mask must select outputs for each candidate")

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None:
            raise NotImplementedError("pending evaluations are not supported")
        super().set_X_pending(None)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mask = self.X_evaluation_mask
        self._validate_mask(mask)
        if mask.shape[0] != X.shape[-2] or mask.device != X.device:
            raise ValueError("evaluation mask must match candidate q and device")
        result = X.sum(dim=(-2, -1)) * 0
        for j, component in enumerate(self.components):
            if mask[:, j].any():
                value = component(X[..., mask[:, j], :])
                if value.shape != result.shape:
                    raise ValueError("component must return one scalar per t-batch")
                result = result + self.output_weights[j] * value
        return result


class qDecoupledParticleKnowledgeGradient(AcquisitionFunction):
    """Finite-set approximate KG using likelihood-weighted posterior particles.

    Each child supplies joint latent regression or class-probability samples.
    The utility receives decision-set samples in semantic output order.
    Gaussian observations use explicit noise variances; categorical observations
    use the sampled class probabilities. Use derivative-free optimization:
    sampled categorical labels are discrete, not pathwise differentiable.
    """

    requires_derivative_free = True

    def __init__(
        self,
        model: HeterogeneousModel,
        decision_X: Tensor,
        utility: Callable[[Sequence[Tensor]], Tensor],
        X_evaluation_mask: Tensor,
        observation_noise: Sequence[float | None],
        *,
        num_particles: int = 128,
        num_fantasies: int = 16,
        seed: int = 0,
        abstain_utility: float | None = None,
    ) -> None:
        super().__init__(model=model)
        if not isinstance(model, HeterogeneousModel):
            raise TypeError("requires HeterogeneousModel")
        if any(count != 1 for count in model.entry_num_outputs):
            raise ValueError("requires one semantic output per entry")
        if decision_X.ndim != 2 or not len(decision_X) or not torch.isfinite(decision_X).all():
            raise ValueError("decision_X must be a finite nonempty matrix")
        if len(observation_noise) != model.num_outputs or min(num_particles, num_fantasies) < 2:
            raise ValueError("requires per-output noise and at least two particles and fantasies")
        for child, noise in zip(model.entries, observation_noise, strict=True):
            if callable(getattr(child, "probability_posterior", None)):
                if noise is not None:
                    raise ValueError("categorical observations require noise=None")
            elif noise is None or not torch.isfinite(decision_X.new_tensor(noise)) or noise <= 0:
                raise ValueError("regression requires positive finite observation variance")
        if abstain_utility is not None and not torch.isfinite(decision_X.new_tensor(abstain_utility)):
            raise ValueError("abstain utility must be finite")
        self.utility = utility
        self.observation_noise = tuple(observation_noise)
        self.num_particles = num_particles
        self.num_fantasies = num_fantasies
        self.seed = seed
        self.abstain_utility = abstain_utility
        self.register_buffer("decision_X", decision_X)
        self.register_buffer("X_evaluation_mask", X_evaluation_mask)
        self.samplers = torch.nn.ModuleList(
            SobolQMCNormalSampler(
                sample_shape=torch.Size([num_particles + num_fantasies]), seed=seed + j
            )
            for j in range(model.num_outputs)
        )

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None:
            raise NotImplementedError("pending evaluations are not supported")
        super().set_X_pending(None)

    def _best(self, values: Tensor) -> Tensor:
        best = values.amax(dim=-1)
        if self.abstain_utility is not None:
            best = best.clamp_min(self.abstain_utility)
        return best

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mask = self.X_evaluation_mask
        q = X.shape[-2]
        if (
            mask.shape != (q, self.model.num_outputs)
            or mask.dtype != torch.bool
            or mask.device != X.device
            or not mask.any(dim=-1).all()
        ):
            raise ValueError("evaluation mask must select outputs for each candidate")
        decision = self.decision_X.expand(*X.shape[:-2], *self.decision_X.shape)
        joint = torch.cat([X, decision], dim=-2)
        n, f = self.num_particles, self.num_fantasies
        log_weights = X.new_zeros((f, n, *X.shape[:-2]))
        decision_samples = []
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for j, (child, sampler, noise) in enumerate(
            zip(self.model.entries, self.samplers, self.observation_noise, strict=True)
        ):
            posterior = (
                child.probability_posterior(joint) if noise is None else child.posterior(joint)
            )
            samples = sampler(posterior)
            decision_samples.append(samples[:n, ..., q:, :])
            if not mask[:, j].any():
                continue
            selected = samples[..., :q, :][..., mask[:, j], :]
            particles, fantasies = selected[:n], selected[n:]
            if noise is None:
                uniform = torch.rand(
                    (f, *([1] * (X.ndim - 2)), fantasies.shape[-2], 1),
                    generator=generator,
                    dtype=X.dtype,
                ).to(X)
                labels = (uniform > fantasies.cumsum(dim=-1)).sum(dim=-1)
                labels = labels.clamp_max(fantasies.shape[-1] - 1)
                indices = labels.unsqueeze(1).unsqueeze(-1).expand(f, n, *particles.shape[1:-1], 1)
                likelihood = particles.unsqueeze(0).expand(f, *particles.shape)
                likelihood = likelihood.gather(-1, indices).squeeze(-1)
                log_likelihood = likelihood.clamp_min(torch.finfo(X.dtype).tiny).log()
                log_weights = log_weights + log_likelihood.sum(dim=-1)
            else:
                epsilon = torch.randn(
                    (f, *([1] * (X.ndim - 2)), fantasies.shape[-2], 1),
                    generator=generator,
                    dtype=X.dtype,
                ).to(X)
                observed = fantasies + noise**0.5 * epsilon
                residual = observed.unsqueeze(1) - particles.unsqueeze(0)
                log_weights = log_weights - residual.square().sum(dim=(-2, -1)) / (2 * noise)
        values = self.utility(decision_samples)
        if values.shape != (n, *X.shape[:-2], len(self.decision_X)):
            raise ValueError("utility must return [particles, *batch, decisions]")
        if not torch.isfinite(values).all():
            raise ValueError("utility must be finite")
        updated = (log_weights.softmax(dim=1).unsqueeze(-1) * values.unsqueeze(0)).sum(dim=1)
        return self._best(updated).mean(dim=0) - self._best(values.mean(dim=0))
