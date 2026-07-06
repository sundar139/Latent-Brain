from __future__ import annotations

from pathlib import Path
from typing import Protocol

from latentbrain.data.schemas import NeuralDataset


class NeuralDataAdapter(Protocol):
    """Common interface for local neural dataset adapters."""

    def can_load(self, dataset_root: Path) -> bool:
        """Return whether this adapter recognizes loadable local files."""
        ...

    def load(self, dataset_root: Path) -> NeuralDataset:
        """Load a local dataset into the LatentBrain neural dataset contract."""
        ...
