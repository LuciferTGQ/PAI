from __future__ import annotations

import copy
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ib_dataset import (
    IBTransitionDataset,
    NormalizationStats,
    compute_normalization,
    load_ib_npz,
    validate_ib_semantics,
)
from src.utils.config import load_config, resolve_path
from src.utils.seed import seed_everything
from src.world_model.model import build_world_model


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
    batch_losses: list[torch.Tensor] = []
    batch_sizes: list[int] = []
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
            batch_losses.append(loss.detach())
            batch_sizes.append(len(history))
    loss_values = torch.stack(batch_losses).cpu().tolist()
    return sum(value * size for value, size in zip(loss_values, batch_sizes)) / sum(batch_sizes)


def _write_history_csv(history: list[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_history_figure(
    history: list[Dict[str, float]], figure_path: Path, model_name: str
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_mse"] for row in history], label="train")
    plt.plot(epochs, [row["val_mse"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title(f"IB {model_name} training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def _build_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any] | None, epochs: int
):
    if not scheduler_config:
        return None
    config = dict(scheduler_config)
    scheduler_type = str(config.pop("type")).lower()
    if scheduler_type in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config)
    if scheduler_type in {"cosine", "cosine_annealing"}:
        config.setdefault("T_max", epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config)
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def _periodic_checkpoint_path(best_path: Path, epoch: int) -> Path:
    stem = best_path.stem
    if stem.endswith("_best"):
        stem = stem[:-5]
    return best_path.with_name(f"{stem}_epoch_{epoch:03d}{best_path.suffix}")


def _capture_rng_state(generator: torch.Generator) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator": generator.get_state(),
    }


def _restore_rng_state(state: Dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["data_loader_generator"])


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    model_config: Dict[str, Any],
    stats: NormalizationStats,
    config: Dict[str, Any],
    generator: torch.Generator,
    epoch: int,
    val_mse: float,
    best_val_mse: float,
    best_epoch: int,
    epochs_without_improvement: int,
    history: list[Dict[str, float]],
    elapsed_seconds: float,
    parameter_count: int,
) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "model_config": model_config,
        "normalization": stats.to_dict(),
        "seed": int(config["seed"]),
        "epoch": epoch,
        "validation_mse": val_mse,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "elapsed_training_seconds": elapsed_seconds,
        "parameter_count": parameter_count,
        "rng_state": _capture_rng_state(generator),
        "source_config": config,
    }


def train_from_config(
    config_path: str | Path,
    resume_from: str | Path | None = None,
    target_epochs: int | None = None,
) -> Tuple[Path, list[Dict[str, float]]]:
    config, project_root = load_config(config_path)
    config = copy.deepcopy(config)
    train_cfg = config["training"]
    if target_epochs is not None:
        train_cfg["epochs"] = int(target_epochs)
    max_epochs = int(train_cfg["epochs"])
    resume_value = resume_from or train_cfg.get("resume_from")
    resume_path = resolve_path(project_root, resume_value) if resume_value else None

    seed_everything(int(config["seed"]))
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["outputs"]
    train_data = load_ib_npz(resolve_path(project_root, data_cfg["train_path"]))
    val_data = load_ib_npz(resolve_path(project_root, data_cfg["val_path"]))
    print("train_audit", validate_ib_semantics(train_data, data_cfg["history_len"], data_cfg["frame_dim"]))
    print("val_audit", validate_ib_semantics(val_data, data_cfg["history_len"], data_cfg["frame_dim"]))

    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        required = {"optimizer_state", "epoch", "normalization", "model_config", "history"}
        missing = required - set(resume_checkpoint)
        if missing:
            raise ValueError(f"Checkpoint cannot resume; missing keys: {sorted(missing)}")
        stats = NormalizationStats.from_dict(resume_checkpoint["normalization"])
    else:
        stats = compute_normalization(train_data, data_cfg["history_len"], data_cfg["frame_dim"])

    train_dataset = IBTransitionDataset(train_data, stats, data_cfg["history_len"], data_cfg["frame_dim"])
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
    if resume_checkpoint is not None and resume_checkpoint["model_config"] != full_model_cfg:
        raise ValueError("Resume checkpoint model_config does not match the requested config")
    device = _select_device(train_cfg["device"])
    model = build_world_model(full_model_cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = _build_scheduler(optimizer, train_cfg.get("scheduler"), max_epochs)
    use_amp = bool(train_cfg["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"model_type={model_cfg.get('type', 'transformer')} parameters={parameter_count}")

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[Dict[str, float]] = []
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if resume_checkpoint.get("scaler_state"):
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val = float(resume_checkpoint.get("best_val_mse", float("inf")))
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(resume_checkpoint["history"])
        elapsed_before = float(resume_checkpoint.get("elapsed_training_seconds", 0.0))
        if resume_checkpoint.get("rng_state"):
            _restore_rng_state(resume_checkpoint["rng_state"], generator)
        print(f"resumed_from={resume_path} start_epoch={start_epoch}")
    if start_epoch > max_epochs:
        raise ValueError(f"Resume epoch {start_epoch - 1} is not below target {max_epochs}")

    checkpoint_path = resolve_path(project_root, output_cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv = resolve_path(project_root, output_cfg["training_history_csv"])
    interval = int(train_cfg.get("checkpoint_interval", 0))
    loss_fn = nn.MSELoss()
    session_start = time.perf_counter()
    last_payload = None
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _run_epoch(
            model, train_loader, loss_fn, device, optimizer, scaler, use_amp,
            train_cfg["gradient_clip"],
        )
        val_loss = _run_epoch(
            model, val_loader, loss_fn, device, None, scaler, use_amp,
            train_cfg["gradient_clip"],
        )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        elapsed = elapsed_before + time.perf_counter() - session_start
        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        _write_history_csv(history, history_csv)
        print(
            f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
            f"lr={row['learning_rate']:.3g} seconds={row['epoch_seconds']:.2f}"
        )
        last_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=full_model_cfg,
            stats=stats,
            config=config,
            generator=generator,
            epoch=epoch,
            val_mse=val_loss,
            best_val_mse=best_val,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            elapsed_seconds=elapsed,
            parameter_count=parameter_count,
        )
        if improved:
            torch.save(last_payload, checkpoint_path)
        if interval and epoch % interval == 0:
            periodic_path = _periodic_checkpoint_path(checkpoint_path, epoch)
            torch.save(last_payload, periodic_path)
            print(f"periodic_checkpoint={periodic_path}")
        if epochs_without_improvement >= int(train_cfg["patience"]):
            print(f"early_stop_epoch={epoch}")
            break

    if last_payload is None:
        raise RuntimeError("Training produced no epochs")
    _write_history_figure(
        history,
        resolve_path(project_root, output_cfg["training_curve"]),
        str(model_cfg.get("type", "transformer")),
    )
    if output_cfg.get("last_checkpoint"):
        last_checkpoint_path = resolve_path(project_root, output_cfg["last_checkpoint"])
        last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(last_payload, last_checkpoint_path)
        print(f"last_checkpoint={last_checkpoint_path}")
    print(
        f"best_checkpoint={checkpoint_path} best_epoch={best_epoch} "
        f"best_val_mse={best_val:.6f} total_seconds={last_payload['elapsed_training_seconds']:.2f}"
    )
    return checkpoint_path, history
