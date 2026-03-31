from __future__ import annotations

import torch
from torch import nn


class ClippedReLU(nn.Module):
    def __init__(self, max_value: float = 1.0) -> None:
        super().__init__()
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, self.max_value)


class NnueTpuModel(nn.Module):
    INPUT_SIZE = 768
    L1_SIZE = 256
    L2_SIZE = 32
    L3_SIZE = 32
    OUTPUT_SCALE = 400.0

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(self.INPUT_SIZE, self.L1_SIZE)
        self.fc2 = nn.Linear(self.L1_SIZE, self.L2_SIZE)
        self.fc3 = nn.Linear(self.L2_SIZE, self.L3_SIZE)
        self.fc4 = nn.Linear(self.L3_SIZE, 1)
        self.activation = ClippedReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x * self.OUTPUT_SCALE
