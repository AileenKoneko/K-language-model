import unittest
from unittest import mock

import torch

from k_language_model.checkpoint import _restore_rng_state


class CheckpointRngRestoreTests(unittest.TestCase):
    def test_restore_rng_state_coerces_cpu_rng_tensor_to_byte(self) -> None:
        state = {"torch_cpu": torch.tensor([1, 2, 3, 255], dtype=torch.int64)}

        with mock.patch("torch.random.set_rng_state") as set_rng_state:
            restored = _restore_rng_state(state)

        self.assertTrue(restored)
        set_rng_state.assert_called_once()
        restored_tensor = set_rng_state.call_args.args[0]
        self.assertIsInstance(restored_tensor, torch.Tensor)
        self.assertEqual(restored_tensor.dtype, torch.uint8)
        self.assertEqual(restored_tensor.device.type, "cpu")
        self.assertTrue(restored_tensor.is_contiguous())
        self.assertEqual(restored_tensor.tolist(), [1, 2, 3, 255])

    def test_restore_rng_state_coerces_cuda_rng_tensor_list_to_byte(self) -> None:
        state = {"torch_cuda": [torch.tensor([7, 8, 9], dtype=torch.int16)]}

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_rng_state_all") as set_rng_state_all,
        ):
            restored = _restore_rng_state(state)

        self.assertTrue(restored)
        set_rng_state_all.assert_called_once()
        restored_tensors = set_rng_state_all.call_args.args[0]
        self.assertEqual(len(restored_tensors), 1)
        self.assertEqual(restored_tensors[0].dtype, torch.uint8)
        self.assertEqual(restored_tensors[0].device.type, "cpu")
        self.assertEqual(restored_tensors[0].tolist(), [7, 8, 9])


if __name__ == "__main__":
    unittest.main()
