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
        names: Optional aliases for model entries. Positional indices remain
            the canonical identity and names only provide additional lookup.
    """

    def __init__(self, *models: nn.Module, names: Sequence[str | None] | None = None) -> None:
        super().__init__()
        if not models:
            raise ValueError("HeterogeneousModel requires at least one model.")
        if not all(isinstance(model, nn.Module) for model in models):
            raise TypeError("All heterogeneous model entries must be torch.nn.Module instances.")
        self.models = nn.ModuleList(models)
        if names is None:
            self._names = (None,) * len(models)
        else:
            if len(names) != len(models):
                raise ValueError("names must contain one entry for each model.")
            if any(name is not None and not isinstance(name, str) for name in names):
                raise TypeError("Each model name must be a string or None.")
            named = [name for name in names if name is not None]
            if len(named) != len(set(named)):
                raise ValueError("Model names must be unique.")
            self._names = tuple(names)
        self._name_to_index = {
            name: index for index, name in enumerate(self._names) if name is not None
        }

    def __len__(self) -> int:
        """Return the number of child model entries."""
        return len(self.models)

    def __iter__(self) -> Iterator[nn.Module]:
        """Iterate over child models in insertion order."""
        return iter(self.models)

    def __getitem__(self, key: int | str) -> nn.Module:
        """Return a child model by canonical index or optional name alias."""
        if isinstance(key, str):
            try:
                key = self._name_to_index[key]
            except KeyError as error:
                raise KeyError(f"Unknown heterogeneous model name: {key!r}.") from error
        return self.models[key]

    @property
    def names(self) -> tuple[str | None, ...]:
        """Return optional model aliases in insertion order."""
        return self._names

    @property
    def entries(self) -> Sequence[nn.Module]:
        """Return the registered child models in insertion order."""
        return self.models
