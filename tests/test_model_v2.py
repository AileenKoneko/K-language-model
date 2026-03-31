import unittest

import torch

from k_language_model.model import K2Layer, KStackModel
from k_language_model.train_app import _build_model, build_parser


class ModelV2Tests(unittest.TestCase):
    def test_k2_layer_v2_block_decay_matches_mask(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(2, 64, 8)

        layer_mask = K2Layer(
            window=64,
            d=8,
            rank=4,
            k_base_kernel_size=5,
            alpha_cap=0.95,
            gamma_min=0.05,
            gamma_max=0.9995,
            decay_impl="mask",
        )
        layer_block = K2Layer(
            window=64,
            d=8,
            rank=4,
            k_base_kernel_size=5,
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

    def test_kstack_model_v2_forward_and_loss_are_finite(self) -> None:
        model = KStackModel(
            vocab_size=32,
            window=16,
            d=8,
            emb_dim=8,
            rank=3,
            n_k2=2,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
            k_base_kernel_size=5,
            share_k_base=True,
            rosa_impl="off",
        )
        token_ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))

        logits = model(token_ids)
        loss = model(token_ids, targets=targets)

        self.assertEqual(logits.shape, (2, 16, 32))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(loss).item())

    def test_prepare_state_dict_projects_v1_shared_dense_k_base_to_kernel(self) -> None:
        model = KStackModel(
            vocab_size=16,
            window=8,
            d=4,
            emb_dim=4,
            rank=2,
            n_k2=2,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
            k_base_kernel_size=4,
            share_k_base=True,
            rosa_impl="off",
        )
        kernel = torch.tensor([1.0, 2.0])
        adapted = model.prepare_state_dict_for_load({"k_stack.shared_k_base_kernel": kernel})

        self.assertIn("k_stack.shared_k_base_kernel", adapted)
        torch.testing.assert_close(
            adapted["k_stack.shared_k_base_kernel"],
            torch.tensor([1.0, 2.0, 0.0, 0.0]),
        )

    def test_prepare_state_dict_adapts_legacy_v2_linear_head_keys(self) -> None:
        model = KStackModel(
            vocab_size=16,
            window=8,
            d=4,
            emb_dim=6,
            rank=2,
            n_k2=2,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
            share_k_base=True,
            k_base_kernel_size=4,
            head_mode="linear",
            rosa_impl="off",
        )
        adapted = model.prepare_state_dict_for_load(
            {
                "head.weight": torch.randn(16, 6),
                "head_to_emb.weight": torch.randn(6, 4),
                "eta_logit": torch.tensor(0.0),
            }
        )

        self.assertIn("head.head.weight", adapted)
        self.assertIn("head.output_projection.weight", adapted)
        self.assertNotIn("head.weight", adapted)
        self.assertNotIn("head_to_emb.weight", adapted)
        self.assertNotIn("eta_logit", adapted)

    def test_train_app_build_model_selects_v2(self) -> None:
        args = build_parser().parse_args(
            [
                "--window",
                "16",
                "--d-model",
                "8",
                "--rank",
                "3",
                "--n-k2",
                "2",
                "--k-base-rank",
                "0",
                "--k-base-kernel-size",
                "7",
                "--decay-impl",
                "block",
            ]
        )

        model = _build_model(
            args,
            vocab_size=32,
            adaptive_cutoffs=None,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
        )

        self.assertIsInstance(model, KStackModel)
        self.assertEqual(model.k_base_kernel_size, 7)
        self.assertEqual(model.decay_impl, "block")
        self.assertEqual(model.k_stack.layers[1].decay_impl, "block")


if __name__ == "__main__":
    unittest.main()
