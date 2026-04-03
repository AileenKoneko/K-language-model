import types

import torch

from k_language_model.model import K2Layer, KStackModel


def _record_rosa_routes(model: KStackModel, rosa_h: torch.Tensor) -> list[tuple[int, bool]]:
    seen: list[tuple[int, bool]] = []
    k2_layers = [layer for layer in model.k_stack.layers if isinstance(layer, K2Layer)]

    for idx, layer in enumerate(k2_layers):
        def _forward(self, h, shared_k_base=None, shared_decay_basis=None, rosa_h=None, *, _idx=idx):
            seen.append((_idx, rosa_h is not None))
            return h

        layer.forward = types.MethodType(_forward, layer)

    model.k_stack(torch.randn_like(rosa_h), rosa_h=rosa_h, rosa_layer_mask=model.rosa_k2_layer_mask)
    return seen


def test_kstack_model_routes_rosa_signal_through_all_k2_layers_by_default() -> None:
    model = KStackModel(
        vocab_size=32,
        window=8,
        d=6,
        emb_dim=4,
        rank=3,
        n_k2=2,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        decay_impl="mask",
    )
    token_ids = torch.tensor(
        [
            [1, 2, 1, 2, 3, 1, 2, 4],
            [5, 5, 5, 5, 1, 5, 1, 5],
        ],
        dtype=torch.int64,
    )

    logits = model(token_ids)
    rosa_h = torch.randn(2, 8, 6)

    assert logits.shape == (2, 8, 32)
    k2_layers = [layer for layer in model.k_stack.layers if isinstance(layer, K2Layer)]
    assert len(k2_layers) == 2
    assert all(layer.rho_logit.shape == () for layer in k2_layers)
    assert model.describe_rosa_layers() == "all"
    assert _record_rosa_routes(model, rosa_h) == [(0, True), (1, True)]


def test_kstack_model_can_limit_rosa_to_final_k2_layer() -> None:
    model = KStackModel(
        vocab_size=32,
        window=8,
        d=6,
        emb_dim=4,
        rank=3,
        n_k2=3,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        decay_impl="mask",
        rosa_layers="final",
    )

    routes = _record_rosa_routes(model, torch.randn(2, 8, 6))

    assert model.describe_rosa_layers() == "final"
    assert routes == [(0, False), (1, False), (2, True)]


def test_kstack_model_supports_ngram_cache_rosa_backend() -> None:
    model = KStackModel(
        vocab_size=48,
        window=8,
        d=6,
        emb_dim=4,
        rank=3,
        n_k2=2,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        decay_impl="mask",
        rosa_impl="ngram_cache",
        rosa_layers="all",
    )
    token_ids = torch.tensor(
        [
            [1, 2, 1, 2, 3, 1, 2, 4],
            [5, 5, 5, 5, 1, 5, 1, 5],
        ],
        dtype=torch.int64,
    )

    logits = model(token_ids)
    assert logits.shape == (2, 8, 48)


def test_kstack_model_supports_copy_prior_rosa_backend() -> None:
    model = KStackModel(
        vocab_size=48,
        window=8,
        d=6,
        emb_dim=4,
        rank=3,
        n_k2=2,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        decay_impl="mask",
        rosa_impl="copy_prior",
        rosa_layers="all",
    )
    token_ids = torch.tensor(
        [
            [1, 2, 1, 2, 3, 1, 2, 4],
            [5, 5, 5, 5, 1, 5, 1, 5],
        ],
        dtype=torch.int64,
    )

    logits = model(token_ids)
    assert logits.shape == (2, 8, 48)
