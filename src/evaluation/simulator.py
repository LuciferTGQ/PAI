from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.data.ib_dataset import load_ib_npz, trajectory_spans
from src.strategy.cem_mpc import CEMMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _summarize(values: Iterable[float]) -> Dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "episodes": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def empirical_behavior_returns(dataset_path: str | Path) -> np.ndarray:
    data = load_ib_npz(dataset_path)
    spans = trajectory_spans(data["index"], len(data["reward"]))
    rewards = data["reward"].reshape(-1)
    return np.asarray([rewards[start:stop].sum() for start, stop in spans])


def _make_official_ib(seed: int, root: Path):
    source_root = root / "external" / "NeoRL"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    from neorl.neorl_envs.ib.industrial_benchmark_python.IBGym import IBGym

    env = IBGym(
        setpoint=70,
        reward_type="classic",
        action_type="continuous",
        observation_type="include_past",
        reset_after_timesteps=1000,
        init_seed=seed,
        n_past_timesteps=30,
    )
    env.init_seed = seed
    return env


def evaluate_online_policy(
    policy_name: str,
    policy_factory: Callable[[int], Any],
    seeds: Iterable[int],
    episode_horizon: int,
    root: Path,
) -> Dict[str, Any]:
    episode_returns: list[float] = []
    reward_traces: list[list[float]] = []
    clipped_scalars = 0
    action_scalars = 0
    for seed in seeds:
        env = _make_official_ib(int(seed), root)
        observation = np.asarray(env.reset(), dtype=np.float32)
        policy = policy_factory(int(seed))
        rewards: list[float] = []
        for _ in range(episode_horizon):
            raw_action = np.asarray(policy.act(observation), dtype=np.float32)
            action = np.clip(raw_action, -1.0, 1.0)
            clipped_scalars += int(np.count_nonzero(action != raw_action))
            action_scalars += int(action.size)
            observation, reward, done, _ = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            rewards.append(float(reward))
            if done:
                break
        episode_return = float(np.sum(rewards))
        episode_returns.append(episode_return)
        reward_traces.append(rewards)
        print(f"policy={policy_name} seed={seed} return={episode_return:.3f}")
    return {
        "returns": episode_returns,
        "summary": _summarize(episode_returns),
        "reward_traces": reward_traces,
        "clipped_action_scalars": clipped_scalars,
        "action_scalars": action_scalars,
        "clipped_action_fraction": clipped_scalars / max(action_scalars, 1),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(results: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Behavior data\n(offline)", "Official BC\n(online)", "WM-CEM\n(online)"]
    values = [
        results["original_behavior_reference"]["returns"],
        results["official_bc"]["returns"],
        results["world_model_cem"]["returns"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot(values, labels=labels, showmeans=True)
    axes[0].set_ylabel("Episode return (higher is better)")
    axes[0].set_title("IB policy comparison")
    axes[0].grid(axis="y", alpha=0.25)
    for key, label in (("official_bc", "Official BC"), ("world_model_cem", "WM-CEM")):
        traces = np.asarray(results[key]["reward_traces"], dtype=np.float32)
        axes[1].plot(traces.mean(axis=0), label=label)
    axes[1].set(xlabel="Simulator step", ylabel="Mean reward", title="Matched-seed online traces")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_strategy_from_config(
    config_path: str | Path,
    world_model_checkpoint_override: str | Path | None = None,
    output_suffix: str = "",
) -> Dict[str, Any]:
    config, root = load_config(config_path)
    data_config = config["data"]
    policy_config = config["policies"]
    evaluation_config = config["evaluation"]
    output_config = config["outputs"]
    seeds = [int(seed) for seed in evaluation_config["seeds"]]
    episode_horizon = int(evaluation_config["episode_horizon"])

    bc_checkpoint = resolve_path(root, policy_config["official_bc_checkpoint"])
    wm_checkpoint = resolve_path(
        root, world_model_checkpoint_override or policy_config["world_model_checkpoint"]
    )
    device = str(policy_config["device"])
    behavior_returns = empirical_behavior_returns(
        resolve_path(root, data_config["behavior_reference_path"])
    )
    bc_policy = OfficialBCPolicy.from_checkpoint(bc_checkpoint, device=device)
    world_model = FrozenWorldModel(wm_checkpoint, device=device)
    if any(parameter.requires_grad for parameter in world_model.model.parameters()):
        raise RuntimeError("World model must remain frozen during strategy evaluation")

    results: Dict[str, Any] = {
        "metadata": {
            "seeds": seeds,
            "episode_horizon": episode_horizon,
            "official_bc_checkpoint": str(bc_checkpoint),
            "world_model_checkpoint": str(wm_checkpoint),
            "behavior_reference": "offline released behavior trajectories; not matched-seed online rollouts",
            "online_environment": "fixed official NeoRL IBGym, setpoint=70, classic reward",
            "environment_action_clipping": [-1.0, 1.0],
            "cem": config["cem"],
        },
        "original_behavior_reference": {
            "returns": behavior_returns.tolist(),
            "summary": _summarize(behavior_returns),
        },
    }
    results["official_bc"] = evaluate_online_policy(
        "official_bc", lambda _: bc_policy, seeds, episode_horizon, root
    )
    results["world_model_cem"] = evaluate_online_policy(
        "world_model_cem",
        lambda seed: CEMMPCPolicy(world_model, bc_policy, config["cem"], seed=seed),
        seeds,
        episode_horizon,
        root,
    )

    def output_path(key: str) -> Path:
        path = resolve_path(root, output_config[key])
        if output_suffix:
            path = path.with_name(f"{path.stem}{output_suffix}{path.suffix}")
        return path

    metrics_path = output_path("metrics_json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: ({inner: value for inner, value in section.items() if inner != "reward_traces"}
              if isinstance(section, dict) else section)
        for key, section in results.items()
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    episode_rows: list[Dict[str, Any]] = []
    for policy_name in ("official_bc", "world_model_cem"):
        for seed, episode_return in zip(seeds, results[policy_name]["returns"]):
            episode_rows.append({"policy": policy_name, "seed": seed, "return": episode_return})
    for index, episode_return in enumerate(behavior_returns):
        episode_rows.append(
            {"policy": "original_behavior_reference_offline", "seed": "", "return": episode_return}
        )
    _write_csv(
        output_path("episode_csv"),
        ["policy", "seed", "return"],
        episode_rows,
    )
    trace_rows: list[Dict[str, Any]] = []
    for policy_name in ("official_bc", "world_model_cem"):
        for seed, trace in zip(seeds, results[policy_name]["reward_traces"]):
            trace_rows.extend(
                {"policy": policy_name, "seed": seed, "step": step, "reward": reward}
                for step, reward in enumerate(trace, start=1)
            )
    _write_csv(
        output_path("reward_trace_csv"),
        ["policy", "seed", "step", "reward"],
        trace_rows,
    )
    _plot_comparison(results, output_path("comparison_figure"))
    return serializable
