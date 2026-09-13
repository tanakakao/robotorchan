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
    """Autoencoder-based nonlinear reducer for continuous inputs.

    The autoencoder is trained once when ``fit`` is called. After training, the
    encoder and decoder parameters are frozen. ``transform`` remains
    differentiable with respect to its input tensor, so BoTorch acquisition
    gradients can flow from the latent GP back to the original input space.

    Args:
        latent_dim: Number of latent dimensions produced by the encoder.
        hidden_dims: Hidden-layer widths used by the encoder. The decoder uses
            the reversed sequence.
        activation: Hidden-layer activation name. Supported values are
            ``"gelu"``, ``"relu"``, ``"silu"``, and ``"tanh"``.
        epochs: Number of reconstruction-training epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Optional mini-batch size. ``None`` uses full-batch training.
        standardize: Whether to standardize each input feature before training
            and encoding.
        eps: Minimum feature scale used during standardization.
        random_state: Seed used to initialize and train the autoencoder.
    """

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
                f"Unsupported activation {activation!r}. "
                f"Choose from {sorted(_ACTIVATIONS)}."
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
            raise ValueError(
                f"latent_dim={self.latent_dim} exceeds input dimension {X.shape[-1]}."
            )

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
            batch_size = n_observations if self.batch_size is None else min(
                self.batch_size,
                n_observations,
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
