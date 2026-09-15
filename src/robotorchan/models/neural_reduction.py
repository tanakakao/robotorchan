"""Neural dimensionality reducers for high-dimensional GP inputs."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from robotorchan.models.reduction import InputReducer

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class AutoEncoderInputReducer(InputReducer):
    """Autoencoder-based nonlinear reducer for continuous inputs."""

    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}. Choose from {sorted(_ACTIVATIONS)}."
            )
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.activation = activation
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self.random_state = int(random_state)

        self.encoder: nn.Sequential | None = None
        self.decoder: nn.Sequential | None = None
        self.register_buffer("x_mean", None)
        self.register_buffer("x_scale", None)
        self.register_buffer("reconstruction_loss", torch.tensor(float("nan")))

    def _activation_module(self) -> nn.Module:
        return _ACTIVATIONS[self.activation]()

    def _build_network(
        self,
        input_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_dims:
            encoder_layers.extend([nn.Linear(previous, width), self._activation_module()])
            previous = width
        encoder_layers.append(nn.Linear(previous, self.latent_dim))

        decoder_layers: list[nn.Module] = []
        previous = self.latent_dim
        for width in reversed(self.hidden_dims):
            decoder_layers.extend([nn.Linear(previous, width), self._activation_module()])
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))

        self.encoder = nn.Sequential(*encoder_layers).to(device=device, dtype=dtype)
        self.decoder = nn.Sequential(*decoder_layers).to(device=device, dtype=dtype)

    def _standardize(self, X: Tensor) -> Tensor:
        assert self.x_mean is not None
        assert self.x_scale is not None
        return (X - self.x_mean) / self.x_scale

    def _freeze_network(self) -> None:
        assert self.encoder is not None
        assert self.decoder is not None
        self.encoder.eval()
        self.decoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        del Y
        if self.latent_dim > X.shape[-1]:
            raise ValueError(f"latent_dim={self.latent_dim} exceeds input dimension {X.shape[-1]}.")

        if self.standardize:
            x_mean = X.mean(dim=0)
            x_scale = X.std(dim=0, unbiased=False).clamp_min(self.eps)
        else:
            x_mean = torch.zeros_like(X[0])
            x_scale = torch.ones_like(X[0])
        self.x_mean = x_mean.detach().clone()
        self.x_scale = x_scale.detach().clone()

        cuda_devices: list[int] = []
        if X.device.type == "cuda":
            device_index = X.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices = [device_index]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.random_state)
            self._build_network(X.shape[-1], device=X.device, dtype=X.dtype)
            assert self.encoder is not None
            assert self.decoder is not None

            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(True)
            self.encoder.train()
            self.decoder.train()

            optimizer = torch.optim.Adam(
                [*self.encoder.parameters(), *self.decoder.parameters()],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            training_X = self._standardize(X).detach()
            n_observations = training_X.shape[0]
            batch_size = (
                n_observations
                if self.batch_size is None
                else min(
                    self.batch_size,
                    n_observations,
                )
            )

            final_loss = torch.tensor(float("nan"), device=X.device, dtype=X.dtype)
            for _ in range(self.epochs):
                permutation = torch.randperm(n_observations, device=X.device)
                for start in range(0, n_observations, batch_size):
                    batch = training_X[permutation[start : start + batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    reconstruction = self.decoder(self.encoder(batch))
                    loss = torch.nn.functional.mse_loss(reconstruction, batch)
                    loss.backward()
                    optimizer.step()
                    final_loss = loss.detach()

            self.reconstruction_loss = final_loss.detach().clone()

        self._freeze_network()
        return self.latent_dim

    def _transform_2d(self, X: Tensor) -> Tensor:
        assert self.encoder is not None
        return self.encoder(self._standardize(X))

    def reconstruct(self, X: Tensor) -> Tensor:
        """Reconstruct original-space inputs through the frozen autoencoder."""
        self._check_fitted()
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final dimension {self.input_dim}, got {X.shape[-1]}.")
        assert self.decoder is not None
        assert self.x_mean is not None
        assert self.x_scale is not None

        original_shape = X.shape
        X_2d = X.reshape(-1, original_shape[-1])
        standardized = self._standardize(X_2d)
        reconstructed = self.decoder(self.encoder(standardized))
        restored = reconstructed * self.x_scale + self.x_mean
        return restored.reshape(original_shape)

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
        if self.encoder is None and metadata is not None and bool(metadata[0].item()):
            input_dim = int(metadata[1].item())
            reference = next(
                (
                    value
                    for key, value in state_dict.items()
                    if key.startswith(f"{prefix}encoder.") and key.endswith(".weight")
                ),
                None,
            )
            device = metadata.device if reference is None else reference.device
            dtype = torch.get_default_dtype() if reference is None else reference.dtype
            self._build_network(input_dim, device=device, dtype=dtype)

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        if self.is_fitted and self.encoder is not None and self.decoder is not None:
            self._freeze_network()


class VAEInputReducer(AutoEncoderInputReducer):
    """Variational autoencoder reducer using posterior means as GP inputs.

    The VAE is fitted without using outcomes. After fitting, ``transform``
    returns ``mu(X)`` rather than a stochastic latent sample. This keeps the
    standard ``ReducedGP`` posterior deterministic and differentiable while the
    encoder still learns a regularized probabilistic latent representation.
    """

    def __init__(self, latent_dim: int, *, beta: float = 1.0, **kwargs) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.beta = float(beta)
        self.encoder_body: nn.Sequential | None = None
        self.mu_head: nn.Linear | None = None
        self.logvar_head: nn.Linear | None = None
        self.register_buffer("kl_loss", torch.tensor(float("nan")))
        self.register_buffer("elbo_loss", torch.tensor(float("nan")))

    def _build_network(
        self,
        input_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        body_layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_dims:
            body_layers.extend([nn.Linear(previous, width), self._activation_module()])
            previous = width
        self.encoder_body = nn.Sequential(*body_layers).to(device=device, dtype=dtype)
        self.mu_head = nn.Linear(previous, self.latent_dim).to(device=device, dtype=dtype)
        self.logvar_head = nn.Linear(previous, self.latent_dim).to(device=device, dtype=dtype)
        self.encoder = nn.Sequential(self.encoder_body, self.mu_head)

        decoder_layers: list[nn.Module] = []
        previous = self.latent_dim
        for width in reversed(self.hidden_dims):
            decoder_layers.extend([nn.Linear(previous, width), self._activation_module()])
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers).to(device=device, dtype=dtype)

    def _encode_2d(self, X: Tensor) -> tuple[Tensor, Tensor]:
        assert self.encoder_body is not None
        assert self.mu_head is not None
        assert self.logvar_head is not None
        hidden = self.encoder_body(self._standardize(X))
        return self.mu_head(hidden), self.logvar_head(hidden)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        del Y
        if self.latent_dim > X.shape[-1]:
            raise ValueError(f"latent_dim={self.latent_dim} exceeds input dimension {X.shape[-1]}.")

        if self.standardize:
            self.x_mean = X.mean(dim=0).detach().clone()
            self.x_scale = X.std(dim=0, unbiased=False).clamp_min(self.eps).detach().clone()
        else:
            self.x_mean = torch.zeros_like(X[0])
            self.x_scale = torch.ones_like(X[0])

        cuda_devices: list[int] = []
        if X.device.type == "cuda":
            device_index = X.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices = [device_index]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.random_state)
            self._build_network(X.shape[-1], device=X.device, dtype=X.dtype)
            assert self.encoder_body is not None
            assert self.mu_head is not None
            assert self.logvar_head is not None
            assert self.decoder is not None
            modules = [self.encoder_body, self.mu_head, self.logvar_head, self.decoder]
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
            n_observations = training_X.shape[0]
            batch_size = (
                n_observations
                if self.batch_size is None
                else min(self.batch_size, n_observations)
            )
            final_reconstruction = torch.tensor(
                float("nan"), device=X.device, dtype=X.dtype
            )
            final_kl = final_reconstruction.clone()

            for _ in range(self.epochs):
                permutation = torch.randperm(n_observations, device=X.device)
                for start in range(0, n_observations, batch_size):
                    batch = training_X[permutation[start : start + batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    hidden = self.encoder_body(batch)
                    mu = self.mu_head(hidden)
                    logvar = self.logvar_head(hidden)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    reconstruction = self.decoder(z)
                    reconstruction_loss = torch.nn.functional.mse_loss(
                        reconstruction, batch
                    )
                    kl_loss = -0.5 * torch.mean(
                        1.0 + logvar - mu.square() - logvar.exp()
                    )
                    loss = reconstruction_loss + self.beta * kl_loss
                    loss.backward()
                    optimizer.step()
                    final_reconstruction = reconstruction_loss.detach()
                    final_kl = kl_loss.detach()

            self.reconstruction_loss = final_reconstruction.clone()
            self.kl_loss = final_kl.clone()
            self.elbo_loss = (final_reconstruction + self.beta * final_kl).clone()

        self._freeze_network()
        assert self.encoder_body is not None
        assert self.mu_head is not None
        assert self.logvar_head is not None
        for module in (self.encoder_body, self.mu_head, self.logvar_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return self.latent_dim

    def _transform_2d(self, X: Tensor) -> Tensor:
        mu, _ = self._encode_2d(X)
        return mu

    def encode_distribution(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior mean and log-variance for original-space inputs."""
        self._check_fitted()
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final dimension {self.input_dim}, got {X.shape[-1]}.")
        original_shape = X.shape[:-1]
        mu, logvar = self._encode_2d(X.reshape(-1, X.shape[-1]))
        latent_shape = (*original_shape, self.latent_dim)
        return mu.reshape(latent_shape), logvar.reshape(latent_shape)

    def sample_latent(self, X: Tensor, n_samples: int) -> Tensor:
        """Draw reparameterized latent samples for diagnostics and future MC GP use."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        mu, logvar = self.encode_distribution(X)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((n_samples, *std.shape), device=std.device, dtype=std.dtype)
        return mu.unsqueeze(0) + std.unsqueeze(0) * eps

    def reconstruct(self, X: Tensor) -> Tensor:
        """Reconstruct inputs using the deterministic posterior mean latent code."""
        self._check_fitted()
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final dimension {self.input_dim}, got {X.shape[-1]}.")
        assert self.decoder is not None
        assert self.x_mean is not None
        assert self.x_scale is not None
        original_shape = X.shape
        mu, _ = self._encode_2d(X.reshape(-1, X.shape[-1]))
        reconstructed = self.decoder(mu)
        restored = reconstructed * self.x_scale + self.x_mean
        return restored.reshape(original_shape)

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
        if self.encoder_body is None and metadata is not None and bool(metadata[0].item()):
            input_dim = int(metadata[1].item())
            reference = next(
                (
                    value
                    for key, value in state_dict.items()
                    if key.startswith(f"{prefix}mu_head.") and key.endswith(".weight")
                ),
                None,
            )
            device = metadata.device if reference is None else reference.device
            dtype = torch.get_default_dtype() if reference is None else reference.dtype
            self._build_network(input_dim, device=device, dtype=dtype)

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
            assert self.encoder_body is not None
            assert self.mu_head is not None
            assert self.logvar_head is not None
            for module in (self.encoder_body, self.mu_head, self.logvar_head):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
