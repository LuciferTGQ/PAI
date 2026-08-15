import numpy as np
import torch
import yaml
from types import SimpleNamespace

from src.world_model.model import (
    MLPWorldModel,
    RecurrentWorldModel,
    TemporalTransformer,
    build_world_model,
)
from src.world_model.trainer import train_from_config
from src.evaluation.checkpoint_selection import select_by_validation_protocol
from src.evaluation.checkpoint_selection import _one_step_nrmse


def test_temporal_transformer_shape() -> None:
    model = TemporalTransformer()
    output = model(torch.zeros(4, 30, 6), torch.zeros(4, 3))
    assert output.shape == (4, 6)


def test_all_world_models_share_interface() -> None:
    history = torch.zeros(4, 30, 6)
    action = torch.zeros(4, 3)
    configs = [
        {"type": "mlp", "hidden_dim": 32, "num_layers": 2, "dropout": 0.0},
        {"type": "gru", "hidden_size": 16, "num_layers": 2, "dropout": 0.0},
        {"type": "lstm", "hidden_size": 16, "num_layers": 2, "dropout": 0.0},
        {
            "type": "transformer",
            "d_model": 16,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
    ]
    outputs = [build_world_model(config)(history, action) for config in configs]
    assert all(output.shape == (4, 6) for output in outputs)
    assert isinstance(build_world_model(configs[0]), MLPWorldModel)
    assert isinstance(build_world_model(configs[1]), RecurrentWorldModel)
    assert isinstance(build_world_model(configs[2]), RecurrentWorldModel)
    assert isinstance(build_world_model(configs[3]), TemporalTransformer)


def _write_tiny_ib_dataset(path, transition_count: int = 16) -> None:
    generator = np.random.default_rng(7)
    obs = generator.normal(size=(transition_count, 180)).astype(np.float32)
    next_obs = np.empty_like(obs)
    next_obs[:, 6:] = obs[:, :-6]
    next_obs[:, :6] = generator.normal(size=(transition_count, 6))
    action = generator.normal(size=(transition_count, 3)).astype(np.float32)
    reward = -(3.0 * next_obs[:, 4] + next_obs[:, 5])
    np.savez(
        path,
        obs=obs,
        next_obs=next_obs,
        action=action,
        reward=reward,
        done=np.zeros(transition_count, dtype=bool),
        index=np.array([transition_count]),
    )


def test_training_checkpoint_can_resume(tmp_path) -> None:
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _write_tiny_ib_dataset(train_path)
    _write_tiny_ib_dataset(val_path)
    checkpoint_path = tmp_path / "tiny_mlp_best.pt"
    config = {
        "seed": 42,
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "history_len": 30,
            "frame_dim": 6,
            "action_dim": 3,
        },
        "model": {
            "type": "mlp",
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "training": {
            "device": "cpu",
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "mixed_precision": False,
            "patience": 10,
            "checkpoint_interval": 1,
        },
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "last_checkpoint": str(tmp_path / "tiny_mlp_last.pt"),
            "training_history_csv": str(tmp_path / "training.csv"),
            "training_curve": str(tmp_path / "training.png"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train_from_config(config_path)
    epoch_one = tmp_path / "tiny_mlp_epoch_001.pt"
    first_checkpoint = torch.load(epoch_one, map_location="cpu", weights_only=False)
    _, history = train_from_config(config_path, resume_from=epoch_one, target_epochs=2)
    final_checkpoint = torch.load(
        tmp_path / "tiny_mlp_epoch_002.pt", map_location="cpu", weights_only=False
    )

    assert len(history) == 2
    assert final_checkpoint["epoch"] == 2
    assert final_checkpoint["history"][0] == first_checkpoint["history"][0]
    assert "optimizer_state" in final_checkpoint
    assert "scaler_state" in final_checkpoint
    assert "rng_state" in final_checkpoint


def test_validation_only_selection_protocol() -> None:
    records = [
        {
            "epoch": 5,
            "one_step_NRMSE": 1.00,
            "mean_NRMSE_H5_H10_H20": 2.0,
        },
        {
            "epoch": 10,
            "one_step_NRMSE": 1.09,
            "mean_NRMSE_H5_H10_H20": 1.5,
        },
        {
            "epoch": 15,
            "one_step_NRMSE": 1.11,
            "mean_NRMSE_H5_H10_H20": 1.0,
        },
    ]
    selection = select_by_validation_protocol(records)
    assert selection["eligible_epochs"] == [5, 10]
    assert selection["selected"]["epoch"] == 10


def test_latest_first_history_update() -> None:
    history = np.arange(180, dtype=np.float32)
    next_frame = np.arange(6, dtype=np.float32) + 1000
    updated = np.concatenate((next_frame, history[:-6]))
    assert np.array_equal(updated[:6], next_frame)
    assert np.array_equal(updated[6:], history[:-6])


def test_common_nrmse_denominator_can_be_fixed_across_models() -> None:
    class FakeWorldModel:
        frame_dim = 6
        stats = SimpleNamespace(target_std=np.full(6, 2.0, dtype=np.float32))

        def predict_next_frame(self, observations, actions):
            return np.zeros((len(observations), 6), dtype=np.float32)

    validation = {
        "obs": np.zeros((4, 180), dtype=np.float32),
        "action": np.zeros((4, 3), dtype=np.float32),
        "next_obs": np.ones((4, 180), dtype=np.float32),
    }
    model_scaled = _one_step_nrmse(FakeWorldModel(), validation, batch_size=2)
    fixed_scaled = _one_step_nrmse(
        FakeWorldModel(),
        validation,
        batch_size=2,
        nrmse_std=np.ones(6, dtype=np.float32),
    )
    assert model_scaled == 0.5
    assert fixed_scaled == 1.0
