from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ib_dataset import (
    IBTransitionDataset,
    compute_normalization,
    load_ib_npz,
    validate_ib_semantics,
)
from src.utils.config import load_config, resolve_path
from src.utils.seed import seed_everything
from src.world_model.model import TemporalTransformer


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for history, action, target in loader:
            history = history.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(history, action)
                loss = loss_fn(prediction, target)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            batch_size = len(history)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def _write_history(history: list[Dict[str, float]], csv_path: Path, figure_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_mse", "val_mse"))
        writer.writeheader()
        writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_mse"] for row in history], marker="o", label="train")
    plt.plot(epochs, [row["val_mse"] for row in history], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("IB Temporal Transformer training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def train_from_config(config_path: str | Path) -> Tuple[Path, list[Dict[str, float]]]:
    config, project_root = load_config(config_path)
    seed_everything(int(config["seed"]))
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    output_cfg = config["outputs"]

    train_data = load_ib_npz(resolve_path(project_root, data_cfg["train_path"]))
    val_data = load_ib_npz(resolve_path(project_root, data_cfg["val_path"]))
    train_audit = validate_ib_semantics(train_data, data_cfg["history_len"], data_cfg["frame_dim"])
    val_audit = validate_ib_semantics(val_data, data_cfg["history_len"], data_cfg["frame_dim"])
    print("train_audit", train_audit)
    print("val_audit", val_audit)

    stats = compute_normalization(train_data, data_cfg["history_len"], data_cfg["frame_dim"])
    train_dataset = IBTransitionDataset(
        train_data, stats, data_cfg["history_len"], data_cfg["frame_dim"]
    )
    val_dataset = IBTransitionDataset(val_data, stats, data_cfg["history_len"], data_cfg["frame_dim"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    full_model_cfg = {
        "frame_dim": data_cfg["frame_dim"],
        "action_dim": data_cfg["action_dim"],
        "history_len": data_cfg["history_len"],
        **model_cfg,
    }
    device = _select_device(train_cfg["device"])
    model = TemporalTransformer(**full_model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    loss_fn = nn.MSELoss()
    use_amp = bool(train_cfg["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    checkpoint_path = resolve_path(project_root, output_cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    epochs_without_improvement = 0
    history: list[Dict[str, float]] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            scaler,
            use_amp,
            train_cfg["gradient_clip"],
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            None,
            scaler,
            use_amp,
            train_cfg["gradient_clip"],
        )
        row = {"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss}
        history.append(row)
        print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": full_model_cfg,
                    "normalization": stats.to_dict(),
                    "seed": int(config["seed"]),
                    "epoch": epoch,
                    "best_val_mse": best_val,
                    "source_config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_cfg["patience"]):
                break

    _write_history(
        history,
        resolve_path(project_root, output_cfg["training_history_csv"]),
        resolve_path(project_root, output_cfg["training_curve"]),
    )
    print(f"best_checkpoint={checkpoint_path} best_val_mse={best_val:.6f}")
    return checkpoint_path, history
