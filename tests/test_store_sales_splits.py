from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from kaggle.store_sales.data import build_train_val_indices


class StoreSalesSplitTests(unittest.TestCase):
    def test_build_train_val_indices_reserves_multiple_rolling_windows(self) -> None:
        bundle = SimpleNamespace(history=16, horizon=4, train_date_count=80, series_count=2)

        train_indices, val_indices = build_train_val_indices(bundle=bundle, train_stride=4, val_windows=6)

        train_target_starts = np.unique(train_indices[1]).tolist()
        val_target_starts = np.unique(val_indices[1]).tolist()

        self.assertEqual(train_target_starts, [16, 20, 24, 28, 32, 36, 40, 44, 48, 52])
        self.assertEqual(val_target_starts, [56, 60, 64, 68, 72, 76])
        self.assertEqual(train_indices[0].shape, train_indices[1].shape)
        self.assertEqual(val_indices[0].shape, val_indices[1].shape)
        self.assertEqual(len(train_indices[0]), len(train_target_starts) * bundle.series_count)
        self.assertEqual(len(val_indices[0]), len(val_target_starts) * bundle.series_count)


if __name__ == "__main__":
    unittest.main()
