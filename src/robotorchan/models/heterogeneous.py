"""Container for models with heterogeneous observation semantics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch import Tensor, nn

from robotorchan.models.classification.base import (
    ClassificationMetadata,
    ClassificationModelMixin,
)


def _num_outputs(model: nn.Module) -> int:
    """Return a model's declared number of outputs."""
    num_outputs = getattr(model, "num_outputs", 1)
    if not isinstance(num_outputs, int) or isinstance(num_outputs, bool) or num_outputs < 1:
        raise ValueError("num_outputs must be a positive integer when declared.")
    return num_outputs


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
        self._entry_num_outputs = tuple(_num_outputs(model) for model in models)
        offsets = [0]
        for num_outputs in self._entry_num_outputs:
            offsets.append(offsets[-1] + num_outputs)
        self._output_offsets = tuple(offsets)
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

    @property
    def entry_num_outputs(self) -> tuple[int, ...]:
        """Return each entry's declared output count in insertion order."""
        return self._entry_num_outputs

    @property
    def num_outputs(self) -> int:
        """Return the total number of outputs owned by all entries."""
        return self._output_offsets[-1]

    @property
    def output_owners(self) -> tuple[tuple[int, int], ...]:
        """Return ownership for every global output in deterministic order."""
        return tuple(self.output_owner(index) for index in range(self.num_outputs))

    @property
    def classification_output_indices(self) -> tuple[int, ...]:
        """Return global indices owned by classification model entries."""
        indices: list[int] = []
        for output_index, (entry_index, _) in enumerate(self.output_owners):
            if isinstance(self.models[entry_index], ClassificationModelMixin):
                indices.append(output_index)
        return tuple(indices)

    @property
    def regression_output_indices(self) -> tuple[int, ...]:
        """Return global indices not owned by classification model entries."""
        classification = set(self.classification_output_indices)
        return tuple(index for index in range(self.num_outputs) if index not in classification)

    def entry_posterior(
        self,
        key: int | str,
        X: Tensor,
        **kwargs: object,
    ) -> object:
        """Return one entry's native posterior without merging semantics."""
        model = self[key]
        posterior = getattr(model, "posterior", None)
        if not callable(posterior):
            raise TypeError(f"Heterogeneous model entry {key!r} does not provide posterior(X).")
        return posterior(X, **kwargs)

    def entry_predict_proba(
        self,
        key: int | str,
        X: Tensor,
        **kwargs: object,
    ) -> Tensor:
        """Return one classification entry's native predictive probabilities."""
        model = self[key]
        if not isinstance(model, ClassificationModelMixin):
            raise TypeError(
                f"Heterogeneous model entry {key!r} is not a classification model."
            )
        return model.predict_proba(X, **kwargs)

    def output_classification_metadata(
        self,
        output_index: int,
    ) -> ClassificationMetadata | None:
        """Return classification metadata when the selected output is categorical."""
        entry_index, local_index = self.output_owner(output_index)
        model = self.models[entry_index]
        if not isinstance(model, ClassificationModelMixin):
            return None
        if model.num_outputs != 1 or local_index != 0:
            raise ValueError(
                "Classification metadata requires an unambiguous single-output "
                "classification entry."
            )
        return model.classification_metadata

    def output_owner(self, output_index: int) -> tuple[int, int]:
        """Map a global output index to its entry and entry-local output index."""
        if not isinstance(output_index, int) or isinstance(output_index, bool):
            raise TypeError("output_index must be an integer.")
        if output_index < 0:
            output_index += self.num_outputs
        if output_index < 0 or output_index >= self.num_outputs:
            raise IndexError("heterogeneous output index out of range")
        for entry_index, stop in enumerate(self._output_offsets[1:]):
            if output_index < stop:
                local_index = output_index - self._output_offsets[entry_index]
                return entry_index, local_index
        raise RuntimeError("Failed to resolve heterogeneous output ownership.")
