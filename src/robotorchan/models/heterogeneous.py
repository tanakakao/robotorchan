"""Container for models with heterogeneous observation semantics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch import nn


class HeterogeneousModel(nn.Module):
    """Compose models without merging their prediction semantics.

    The container registers each child as a PyTorch submodule so standard
    nn.Module operations such as to(), train(), eval(), and state_dict()
    propagate naturally. It deliberately does not implement a shared
    posterior() contract because regression posteriors and classification
    latent posteriors have different meanings.

    Args:
        *models: Models to compose. At least one model is required.
    """

    def __init__(self, *models: nn.Module) -> None:
        super().__init__()
        if not models:
            raise ValueError("HeterogeneousModel requires at least one model.")
        if not all(isinstance(model, nn.Module) for model in models):
            raise TypeError("All heterogeneous model entries must be torch.nn.Module instances.")
        self.models = nn.ModuleList(models)

    def __len__(self) -> int:
        """Return the number of child model entries."""
        return len(self.models)

    def __iter__(self) -> Iterator[nn.Module]:
        """Iterate over child models in insertion order."""
        return iter(self.models)

    def __getitem__(self, index: int) -> nn.Module:
        """Return a child model by entry index."""
        return self.models[index]

    @property
    def entries(self) -> Sequence[nn.Module]:
        """Return the registered child models in insertion order."""
        return self.models
