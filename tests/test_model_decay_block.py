import unittest

import torch

from k_language_model.model import K2Layer


class BlockDecayTests(unittest.TestCase):
    def test_block_decay_matches_mask_and_stays_finite_for_small_gamma(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(2, 64, 8)

        layer_mask = K2Layer(
            window=64,
            d=8,
            rank=4,
            k_base_rank=0,
            alpha_cap=0.95,
            gamma_min=0.05,
            gamma_max=0.9995,
            decay_impl="mask",
        )
        layer_block = K2Layer(
            window=64,
            d=8,
            rank=4,
            k_base_rank=0,
            alpha_cap=0.95,
            gamma_min=0.05,
            gamma_max=0.9995,
            decay_impl="block",
        )
        layer_block.load_state_dict(layer_mask.state_dict())
        layer_mask.decay_logit.data.fill_(-100.0)
        layer_block.decay_logit.data.fill_(-100.0)

        out_mask = layer_mask(inputs)
        out_block = layer_block(inputs)

        self.assertTrue(torch.isfinite(out_block).all().item())
        torch.testing.assert_close(out_block, out_mask, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
