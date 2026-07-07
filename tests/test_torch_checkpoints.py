from __future__ import annotations

from pathlib import Path

import torch

from latentbrain.torch.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.ones(2, 2)
    model(x).sum().backward()
    optimizer.step()
    saved_weight = model.weight.detach().clone()

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={"validation_loss": 1.25},
        config={"name": "unit"},
    )
    with torch.no_grad():
        model.weight.add_(10.0)

    loaded = load_checkpoint(path, model=model, optimizer=optimizer)

    assert loaded["epoch"] == 3
    assert loaded["metrics"] == {"validation_loss": 1.25}
    assert loaded["config"] == {"name": "unit"}
    torch.testing.assert_close(model.weight, saved_weight)
    assert optimizer.state_dict()["state"]
