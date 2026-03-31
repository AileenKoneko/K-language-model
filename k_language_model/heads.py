from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


_HEAD_REGISTRY: Dict[str, Type["HeadImplementation"]] = {}


def _move_state_key(state: Dict[str, torch.Tensor], old_key: str, new_key: str) -> None:
    tensor = state.pop(old_key, None)
    if isinstance(tensor, torch.Tensor) and new_key not in state:
        state[new_key] = tensor


def register_head(cls: Type["HeadImplementation"]) -> Type["HeadImplementation"]:
    if not cls.name:
        raise ValueError("HeadImplementation subclasses must define a non-empty name.")
    _HEAD_REGISTRY[cls.name] = cls
    return cls


def build_head(
    *,
    head_mode: str,
    d_model: int,
    emb_dim: int,
    vocab_size: int,
    head_mult: int,
    head_dropout: float,
    adaptive_cutoffs: List[int] | None,
    adaptive_div_value: float,
    embedding: nn.Embedding,
) -> "HeadImplementation":
    try:
        head_cls = _HEAD_REGISTRY[head_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown head_mode: {head_mode}") from exc
    return head_cls(
        d_model=d_model,
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        head_mult=head_mult,
        head_dropout=head_dropout,
        adaptive_cutoffs=adaptive_cutoffs,
        adaptive_div_value=adaptive_div_value,
        embedding=embedding,
    )


class HeadImplementation(nn.Module, ABC):
    name = ""

    def __init__(
        self,
        *,
        d_model: int,
        emb_dim: int,
        vocab_size: int,
        head_mult: int,
        head_dropout: float,
        adaptive_cutoffs: List[int] | None,
        adaptive_div_value: float,
        embedding: nn.Embedding,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.emb_dim = int(emb_dim)
        self.vocab_size = int(vocab_size)
        self.head_mult = int(head_mult)
        self.head_dropout = float(head_dropout)
        self.adaptive_cutoffs = [] if adaptive_cutoffs is None else list(adaptive_cutoffs)
        self.adaptive_div_value = float(adaptive_div_value)
        self.tie_weights = False
        self.output_projection = nn.Identity()

    @abstractmethod
    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss(self, hidden: torch.Tensor, targets: torch.Tensor, reduction: str) -> torch.Tensor:
        raise NotImplementedError

    def adapt_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return state


@register_head
class LinearHead(HeadImplementation):
    name = "linear"

    def __init__(self, *, embedding: nn.Embedding, **kwargs):
        super().__init__(embedding=embedding, **kwargs)
        if self.emb_dim != self.d_model:
            self.output_projection = nn.Linear(self.d_model, self.emb_dim, bias=False)
        self.head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)
        self.head.weight = embedding.weight
        self.tie_weights = True

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.output_projection(hidden))

    def loss(self, hidden: torch.Tensor, targets: torch.Tensor, reduction: str) -> torch.Tensor:
        scores = self.scores(hidden)
        return F.cross_entropy(scores.reshape(-1, self.vocab_size), targets.reshape(-1), reduction=reduction)

    def adapt_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted = dict(state)
        _move_state_key(adapted, "head.weight", "head.head.weight")
        _move_state_key(adapted, "head_to_emb.weight", "head.output_projection.weight")
        adapted.pop("head_to_emb.bias", None)
        return adapted


@register_head
class GeluHead(HeadImplementation):
    name = "gelu"

    def __init__(self, *, embedding: nn.Embedding, **kwargs):
        super().__init__(embedding=embedding, **kwargs)
        hidden = max(self.d_model, self.head_mult * self.d_model)
        self.norm = RMSNorm(self.d_model)
        self.up = nn.Linear(self.d_model, hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.head_dropout) if self.head_dropout > 0 else nn.Identity()
        self.down = nn.Linear(hidden, self.vocab_size, bias=False)

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden)
        hidden = self.up(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.down(hidden)

    def loss(self, hidden: torch.Tensor, targets: torch.Tensor, reduction: str) -> torch.Tensor:
        scores = self.scores(hidden)
        return F.cross_entropy(scores.reshape(-1, self.vocab_size), targets.reshape(-1), reduction=reduction)

    def adapt_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted = dict(state)
        legacy_map = {
            "head.0.scale": "head.norm.scale",
            "head.1.weight": "head.up.weight",
            "head.1.bias": "head.up.bias",
            "head.3.weight": "head.down.weight",
        }
        for old_key, new_key in legacy_map.items():
            _move_state_key(adapted, old_key, new_key)
        return adapted


@register_head
class AdaptiveHead(HeadImplementation):
    name = "adaptive"

    def __init__(self, *, embedding: nn.Embedding, **kwargs):
        super().__init__(embedding=embedding, **kwargs)
        self.dropout = nn.Dropout(self.head_dropout) if self.head_dropout > 0 else nn.Identity()
        self.head = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=self.d_model,
            n_classes=self.vocab_size,
            cutoffs=self.adaptive_cutoffs,
            div_value=self.adaptive_div_value,
            head_bias=False,
        )

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        head_in = self.dropout(hidden).reshape(-1, hidden.size(-1))
        return self.head.log_prob(head_in).view(hidden.size(0), hidden.size(1), self.vocab_size)

    def loss(self, hidden: torch.Tensor, targets: torch.Tensor, reduction: str) -> torch.Tensor:
        head_in = self.dropout(hidden).reshape(-1, hidden.size(-1))
        flat_targets = targets.reshape(-1)
        out = self.head(head_in, flat_targets)
        if reduction == "mean":
            return out.loss
        return out.loss * flat_targets.numel()

    def adapt_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted = dict(state)
        for key, tensor in list(adapted.items()):
            if not key.startswith("head.") or key.startswith("head.head.") or not isinstance(tensor, torch.Tensor):
                continue
            new_key = f"head.head.{key[len('head.'):]}"
            if new_key not in adapted:
                adapted[new_key] = tensor
            adapted.pop(key, None)
        return adapted
