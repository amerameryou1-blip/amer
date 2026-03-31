from __future__ import annotations

from typing import Any

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: str = "cpu") -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")

        self.decay = decay
        self.device = torch.device(device)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().to(self.device).clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(param.detach().to(self.device), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow = {
            name: tensor.to(self.device)
            for name, tensor in state_dict["shadow"].items()
        }
