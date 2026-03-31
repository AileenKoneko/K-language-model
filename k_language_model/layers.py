import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale


class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(d, d * mult)
        self.down = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.gelu(self.up(x))))


class K1Layer(nn.Module):
    def __init__(self, d: int, mlp_dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.feature_mix = nn.Linear(d, d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        out = self.feature_mix(self.norm1(h))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h


class K0Layer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (self.norm(h) * self.gain + self.bias)
