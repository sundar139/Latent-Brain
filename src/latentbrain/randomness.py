from __future__ import annotations

import importlib
import importlib.util
import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, int | bool]:
    """Seed supported random number generators."""
    if seed < 0:
        msg = "seed must be a non-negative integer"
        raise ValueError(msg)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    if importlib.util.find_spec("torch") is not None:
        torch: Any = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
        torch_seeded = True

    return {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch": torch_seeded,
        "deterministic": deterministic,
    }
