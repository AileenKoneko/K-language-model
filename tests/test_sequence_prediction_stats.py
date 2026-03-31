from pathlib import Path

import numpy as np

from k_language_model.sequence_prediction_stats import format_sequence_prediction_report, load_sequence_prediction_logs


def test_sequence_prediction_stats_report(tmp_path: Path) -> None:
    path = tmp_path / "logs.npz"
    np.savez_compressed(
        path,
        gamma_vec=np.array(
            [
                [[0.7, 0.8], [0.9, 0.95]],
                [[0.71, 0.81], [0.91, 0.96]],
            ],
            dtype=np.float32,
        ),
        alpha=np.array(
            [
                [[0.10, 0.11], [0.20, 0.21]],
                [[0.11, 0.12], [0.21, 0.22]],
            ],
            dtype=np.float32,
        ),
        k_base_gate=np.array([[0.8, 0.7], [0.81, 0.71]], dtype=np.float32),
        training_loss=np.array([0.5, 0.25], dtype=np.float32),
        training_rmse=np.array([0.7, 0.5], dtype=np.float32),
        training_mae=np.array([0.6, 0.4], dtype=np.float32),
        baseline_mse=np.array([1.0, 0.9], dtype=np.float32),
        baseline_rmse=np.array([1.0, 0.95], dtype=np.float32),
        logged_steps=np.array([100, 200], dtype=np.int64),
    )

    logs = load_sequence_prediction_logs(path)
    report = format_sequence_prediction_report(logs=logs, path=path, checkpoint_index=-1, max_rows=10)

    assert "File:" in report
    assert "training_loss: shape=(2,)" in report
    assert "100 | 0.500000 | 0.700000 | 0.600000 | 1.000000" in report
    assert "200 | 0.250000 | 0.500000 | 0.400000 | 0.950000" in report
    assert "Layer stats at checkpoint index=1, step=200:" in report
    assert "0 | 0.110000 | 0.115000 | 0.120000 | 0.810000 | 0.710000 | 0.760000 | 0.810000" in report
