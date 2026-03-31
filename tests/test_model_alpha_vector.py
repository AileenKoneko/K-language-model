import torch

from k_language_model.model import K2Layer, KStackModel


def test_k2_alpha_vector_forward_and_state_adaptation() -> None:
    layer = K2Layer(window=8, d=4, rank=3, k_base_rank=0, decay_impl="mask")
    inputs = torch.randn(2, 8, 4)
    outputs = layer(inputs)

    assert layer.alpha_logit.shape == (3,)
    assert outputs.shape == inputs.shape

    model = KStackModel(
        vocab_size=32,
        window=8,
        d=4,
        emb_dim=None,
        rank=3,
        n_k2=1,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        decay_impl="mask",
    )
    alpha_key = next(name for name in model.state_dict().keys() if name.endswith("alpha_logit"))

    adapted_scalar = model.prepare_state_dict_for_load({alpha_key: torch.tensor(0.0)})
    adapted_singleton = model.prepare_state_dict_for_load({alpha_key: torch.tensor([0.25])})

    assert adapted_scalar[alpha_key].shape == (3,)
    assert adapted_singleton[alpha_key].shape == (3,)
    assert torch.allclose(adapted_scalar[alpha_key], torch.zeros(3))
    assert torch.allclose(adapted_singleton[alpha_key], torch.full((3,), 0.25))
