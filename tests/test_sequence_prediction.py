from pathlib import Path

import numpy as np
import torch

from k_language_model.sequence_prediction import SequencePredictionConfig, train_sequence_predictor


def test_sequence_prediction_logs_shapes_and_save(tmp_path: Path) -> None:
    output_path = tmp_path / "sequence_prediction_logs.npz"
    cfg = SequencePredictionConfig(
        seq_len=32,
        batch_size=4,
        steps=6,
        log_interval=2,
        d_model=16,
        rank=4,
        n_k2=2,
        device="cpu",
        output_path=output_path,
    )

    logs = train_sequence_predictor(cfg)

    assert logs["gamma_vec"].shape == (3, 2, 4)
    assert logs["alpha"].shape == (3, 2, 4)
    assert logs["k_base_gate"].shape == (3, 2)
    assert logs["training_loss"].shape == (3,)
    assert logs["training_rmse"].shape == (3,)
    assert logs["training_mae"].shape == (3,)
    assert logs["baseline_mse"].shape == (3,)
    assert logs["baseline_rmse"].shape == (3,)
    assert logs["logged_steps"].tolist() == [2, 4, 6]
    assert output_path.exists()

    saved = np.load(output_path)
    assert saved["gamma_vec"].shape == (3, 2, 4)
    assert saved["alpha"].shape == (3, 2, 4)
    assert saved["training_rmse"].shape == (3,)
