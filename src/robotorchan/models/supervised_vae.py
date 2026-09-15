"""Outcome-aware variational input reduction and GP wrappers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from robotorchan.models.neural_reduction import VAEInputReducer
from robotorchan.models.reduced import ReducedGP
from robotorchan.models.reduction import InputReducer


class SupervisedVAEInputReducer(VAEInputReducer):
    """VAE reducer with an auxiliary outcome-prediction objective."""

    def __init__(
        self,
        latent_dim: int,
        *,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        if supervised_weight < 0:
            raise ValueError("supervised_weight must be non-negative.")
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.supervised_weight = float(supervised_weight)
        self.standardize_y = bool(standardize_y)
        self.supervised_head: nn.Linear | None = None
        self.register_buffer("y_mean", None)
        self.register_buffer("y_scale", None)
        self.register_buffer("supervised_loss", torch.tensor(float("nan")))
        self.register_buffer("combined_loss", torch.tensor(float("nan")))

    def _build_supervised_head(
        self,
        output_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.supervised_head = nn.Linear(self.latent_dim, output_dim).to(device=device, dtype=dtype)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        if Y is None:
            raise ValueError("SupervisedVAEInputReducer requires paired Y values.")
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.ndim != 2:
            raise ValueError("Y must have shape [n] or [n, m].")
        if self.latent_dim > X.shape[-1]:
            raise ValueError(f"latent_dim={self.latent_dim} exceeds input dimension {X.shape[-1]}.")

        if self.standardize:
            self.x_mean = X.mean(dim=0).detach().clone()
            self.x_scale = X.std(dim=0, unbiased=False).clamp_min(self.eps).detach().clone()
        else:
            self.x_mean = torch.zeros_like(X[0])
            self.x_scale = torch.ones_like(X[0])
        if self.standardize_y:
            self.y_mean = Y.mean(dim=0).detach().clone()
            self.y_scale = Y.std(dim=0, unbiased=False).clamp_min(self.eps).detach().clone()
        else:
            self.y_mean = torch.zeros_like(Y[0])
            self.y_scale = torch.ones_like(Y[0])

        cuda_devices: list[int] = []
        if X.device.type == "cuda":
            device_index = X.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices = [device_index]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.random_state)
            self._build_network(X.shape[-1], device=X.device, dtype=X.dtype)
            self._build_supervised_head(Y.shape[-1], device=X.device, dtype=X.dtype)
            assert self.encoder_body is not None
            assert self.mu_head is not None
            assert self.logvar_head is not None
            assert self.decoder is not None
            assert self.supervised_head is not None
            assert self.y_mean is not None
            assert self.y_scale is not None
            modules = [
                self.encoder_body,
                self.mu_head,
                self.logvar_head,
                self.decoder,
                self.supervised_head,
            ]
            for module in modules:
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

            optimizer = torch.optim.Adam(
                [parameter for module in modules for parameter in module.parameters()],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            training_X = self._standardize(X).detach()
            training_Y = ((Y - self.y_mean) / self.y_scale).detach()
            n_observations = training_X.shape[0]
            batch_size = (
                n_observations if self.batch_size is None else min(self.batch_size, n_observations)
            )
            final_reconstruction = torch.tensor(float("nan"), device=X.device, dtype=X.dtype)
            final_kl = final_reconstruction.clone()
            final_supervised = final_reconstruction.clone()

            for _ in range(self.epochs):
                permutation = torch.randperm(n_observations, device=X.device)
                for start in range(0, n_observations, batch_size):
                    indices = permutation[start : start + batch_size]
                    batch_X = training_X[indices]
                    batch_Y = training_Y[indices]
                    optimizer.zero_grad(set_to_none=True)
                    hidden = self.encoder_body(batch_X)
                    mu = self.mu_head(hidden)
                    logvar = self.logvar_head(hidden)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    reconstruction = self.decoder(z)
                    prediction = self.supervised_head(mu)
                    reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, batch_X)
                    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
                    supervised_loss = torch.nn.functional.mse_loss(prediction, batch_Y)
                    loss = (
                        reconstruction_loss
                        + self.beta * kl_loss
                        + self.supervised_weight * supervised_loss
                    )
                    loss.backward()
                    optimizer.step()
                    final_reconstruction = reconstruction_loss.detach()
                    final_kl = kl_loss.detach()
                    final_supervised = supervised_loss.detach()

            self.reconstruction_loss = final_reconstruction.clone()
            self.kl_loss = final_kl.clone()
            self.supervised_loss = final_supervised.clone()
            self.elbo_loss = (final_reconstruction + self.beta * final_kl).clone()
            self.combined_loss = (
                self.elbo_loss + self.supervised_weight * final_supervised
            ).clone()

        self._freeze_network()
        for module in (
            self.encoder_body,
            self.mu_head,
            self.logvar_head,
            self.supervised_head,
        ):
            assert module is not None
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return self.latent_dim

    def predict_auxiliary(self, X: Tensor) -> Tensor:
        """Predict outcomes with the frozen auxiliary head in original Y scale."""
        self._check_fitted()
        assert self.supervised_head is not None
        assert self.y_mean is not None
        assert self.y_scale is not None
        prediction = self.supervised_head(self.transform(X))
        return prediction * self.y_scale + self.y_mean

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        metadata = state_dict.get(f"{prefix}_fit_metadata")
        head_weight = state_dict.get(f"{prefix}supervised_head.weight")
        if self.encoder_body is None and metadata is not None and bool(metadata[0].item()):
            input_dim = int(metadata[1].item())
            reference = next(
                (
                    value
                    for key, value in state_dict.items()
                    if key.startswith(f"{prefix}encoder_body.") and key.endswith(".weight")
                ),
                None,
            )
            device = metadata.device if reference is None else reference.device
            dtype = torch.get_default_dtype() if reference is None else reference.dtype
            self._build_network(input_dim, device=device, dtype=dtype)
        if self.supervised_head is None and head_weight is not None:
            self._build_supervised_head(
                head_weight.shape[0], device=head_weight.device, dtype=head_weight.dtype
            )
        InputReducer._load_from_state_dict(
            self,
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        if self.is_fitted:
            self._freeze_network()
            assert self.supervised_head is not None
            self.supervised_head.eval()
            for parameter in self.supervised_head.parameters():
                parameter.requires_grad_(False)


class SupervisedVAEGP(ReducedGP):
    """Single-task GP using a frozen outcome-aware VAE posterior mean."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=SupervisedVAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )
