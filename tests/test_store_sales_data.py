from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from kaggle.store_sales.data import load_store_sales_bundle


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class StoreSalesDataTests(unittest.TestCase):
    def _write_competition_files(self, root: Path, test_dates: list[str]) -> None:
        _write_csv(
            root / "stores.csv",
            ["store_nbr", "city", "state", "type", "cluster"],
            [{"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 1}],
        )
        _write_csv(
            root / "train.csv",
            ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
            [
                {"id": 1, "date": "2020-01-01", "store_nbr": 1, "family": "GROCERY I", "sales": 2.0, "onpromotion": 0},
                {"id": 2, "date": "2020-01-03", "store_nbr": 1, "family": "GROCERY I", "sales": 4.0, "onpromotion": 2},
            ],
        )
        _write_csv(
            root / "test.csv",
            ["id", "date", "store_nbr", "family", "onpromotion"],
            [
                {"id": idx + 3, "date": current_date, "store_nbr": 1, "family": "GROCERY I", "onpromotion": idx}
                for idx, current_date in enumerate(test_dates)
            ],
        )
        _write_csv(
            root / "oil.csv",
            ["date", "dcoilwtico"],
            [
                {"date": "2020-01-01", "dcoilwtico": "50.0"},
                {"date": "2020-01-03", "dcoilwtico": "52.0"},
                {"date": test_dates[0], "dcoilwtico": "53.0"},
            ],
        )
        _write_csv(
            root / "holidays_events.csv",
            ["date", "type", "locale", "locale_name", "description", "transferred"],
            [],
        )

    def test_load_store_sales_bundle_fills_missing_train_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_competition_files(root, ["2020-01-04"])

            bundle = load_store_sales_bundle(root, history=2, horizon=None)

        self.assertEqual(bundle.train_dates, [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)])
        self.assertEqual(bundle.all_dates, [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)])
        self.assertEqual(bundle.horizon, 1)
        self.assertEqual(bundle.sales_log.shape, (1, 3))
        self.assertEqual(bundle.sales_mask.shape, (1, 3))
        self.assertEqual(float(bundle.sales_log[0, 1]), 0.0)
        self.assertEqual(float(bundle.sales_mask[0, 1]), 1.0)
        self.assertEqual(float(bundle.onpromotion[0, 1]), 0.0)

    def test_load_store_sales_bundle_rejects_gap_before_test_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_competition_files(root, ["2020-01-05"])

            with self.assertRaisesRegex(
                ValueError,
                "The test horizon must start on the day immediately after the train range ends.",
            ):
                load_store_sales_bundle(root, history=2, horizon=None)


if __name__ == "__main__":
    unittest.main()
