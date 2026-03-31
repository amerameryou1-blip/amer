from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer, alpha: float = 0.5, k: int = 5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if k < 1:
            raise ValueError("k must be >= 1")

        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self._step = 0
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state: defaultdict[Any, dict[str, torch.Tensor]] = defaultdict(dict)

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["slow_param"] = param.detach().clone()

    def zero_grad(self, set_to_none: bool | None = None) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1

        if self._step % self.k == 0:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    slow = self.state[param]["slow_param"]
                    slow.add_(param.data - slow, alpha=self.alpha)
                    param.data.copy_(slow)
        return loss

    def state_dict(self) -> dict[str, Any]:
        slow_tensors: list[torch.Tensor] = []
        for group in self.param_groups:
            for param in group["params"]:
                slow_tensors.append(self.state[param]["slow_param"].cpu())
        return {
            "optimizer": self.optimizer.state_dict(),
            "alpha": self.alpha,
            "k": self.k,
            "step": self._step,
            "slow_state": slow_tensors,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.alpha = state_dict["alpha"]
        self.k = state_dict["k"]
        self._step = state_dict["step"]
        self.optimizer.load_state_dict(state_dict["optimizer"])

        slow_state = iter(state_dict["slow_state"])
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["slow_param"] = next(slow_state).to(param.device, dtype=param.dtype)
