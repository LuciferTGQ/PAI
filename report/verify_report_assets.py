"""Lightweight consistency checks for the final report artifact.

The verifier checks report-facing assets and exported data only. It never
reruns training, the simulator, or the historical data audit.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
GENERATED = REPORT / "generated"
SOURCE_DATA = REPORT / "source_data"
PDF = ROOT / "output" / "pdf" / "pai_industrial_world_model_report.pdf"
LOG = ROOT / "output" / "pdf" / "pai_industrial_world_model_report.log"


def row_count(filename: str) -> int:
    with (SOURCE_DATA / filename).open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def report_text() -> str:
    paths = [REPORT / "main.tex"]
    paths.extend(sorted((REPORT / "sections").glob("*.tex")))
    paths.extend(sorted(GENERATED.glob("*.tex")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def main() -> None:
    failures: list[str] = []

    for index in range(1, 10):
        stem = next(FIGURES.glob(f"fig{index}_*.pdf"), None)
        if stem is None:
            failures.append(f"missing PDF for Figure {index}")
            continue
        for suffix in (".pdf", ".svg", ".png"):
            candidate = stem.with_suffix(suffix)
            if not candidate.exists() or candidate.stat().st_size == 0:
                failures.append(f"missing or empty figure asset: {candidate.name}")

    expected_rows = {
        "fig2_fig3_world_model_metrics.csv": 15,
        "fig5_strategy_episodes.csv": 65,
        "fig6_world_model_strategy_episodes.csv": 50,
        "fig7_final_evaluation.csv": 1040,
        "fig8_kl_ablation_returns.csv": 10,
        "fig9_training_history.csv": 50,
        "table_horizon_ablation.csv": 3,
        "table_horizon_sensitivity.csv": 3,
        "table_training_sufficiency.csv": 3,
        "dataset_scale_audit.csv": 3,
    }
    observed_rows = {name: row_count(name) for name in expected_rows}
    for name, expected in expected_rows.items():
        if observed_rows[name] != expected:
            failures.append(f"{name}: expected {expected} rows, got {observed_rows[name]}")

    with (SOURCE_DATA / "fig2_fig3_world_model_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        world_models = list(csv.DictReader(handle))
    matrix = {(row["scale"], row["architecture"]) for row in world_models}
    if len(matrix) != 15:
        failures.append(f"World Model matrix has {len(matrix)} unique pairs")
    selected = {
        row["scale"]: row["architecture"]
        for row in world_models
        if row["selected_best_world_model"].lower() == "true"
    }
    expected_selected = {
        "M100": "GRU",
        "M1000": "Transformer-2L",
        "M10000": "Transformer-2L",
    }
    if selected != expected_selected:
        failures.append(f"selected World Models differ: {selected}")

    visible_text = report_text()
    forbidden = [
        r"reference\s+MPPI",
        r"custom\s+MPPI",
        r"corrected\s+MPPI",
        r"reference\s+iCEM",
        r"fair\s+Transformer",
        r"BestWM@",
        r"offline empirical reference",
        r"common-validation",
        r"population\s+SD",
        r"win\s*rate",
        r"untouched\s+seeds?",
        r"seeds?\s+\d",
        r"42--46",
        r"100--109",
        r"本次考核",
        r"task\s+coverage",
        r"Task\s+[123]",
        r"PAI Industrial World Model Assessment",
        r"连接线仅",
        r"连接线只",
        r"不包含插值",
        r"不用于反向",
        r"确定性优化测试",
        r"M99(?:9|99)?\b",
    ]
    for pattern in forbidden:
        if re.search(pattern, visible_text, flags=re.IGNORECASE):
            failures.append(f"forbidden report-facing term found: {pattern}")

    required_phrases = [
        "面向工业过程控制的世界模型",
        "原始行为数据参考值（Original Behavior）",
        "仿真奖励只用于策略优化和最终控制评价",
        "称为统一验证",
        "归一化均方根误差（Normalized Root Mean Squared Error, NRMSE）",
        "H5 表示连续预测5个时间步后的误差",
        "并不是前1至 $k$ 步预测误差的算术累加",
        "单步 NRMSE 遍历全部验证转移",
        "规划长度消融使用 CEM 作为代表性规划器",
        "H=10也是唯一合格候选",
        "arXiv:2102.00714v2",
        "架构内选定轮次",
        "100、1000和10000条完整工业轨迹",
        "补充实验结果",
        "工业应用与部署思考",
        "相对 BC 分别提高14,229、68,282和64,535",
    ]
    for phrase in required_phrases:
        if phrase not in visible_text:
            failures.append(f"required report phrase missing: {phrase}")

    pages = len(PdfReader(str(PDF)).pages) if PDF.exists() else 0
    if pages < 10:
        failures.append(f"final PDF missing or unexpectedly short: {pages} pages")

    log_text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    for marker in ("Overfull \\hbox", "Undefined control sequence", "LaTeX Error"):
        if marker in log_text:
            failures.append(f"LaTeX log contains: {marker}")

    snapshot = json.loads((GENERATED / "report_metrics_snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("world_model_rows") != 15:
        failures.append("metrics snapshot does not contain 15 World Model rows")
    if snapshot.get("final_online_episode_rows") != 40:
        failures.append("metrics snapshot does not contain 40 final online episodes")

    with (SOURCE_DATA / "dataset_scale_audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        scale_rows = list(csv.DictReader(handle))
    observed_scales = {
        row["scale"]: (
            int(row["training_trajectories"]),
            int(row["transitions_per_trajectory"]),
            int(row["total_usable_transitions"]),
            int(row["formal_training_epochs"]),
        )
        for row in scale_rows
    }
    expected_scales = {
        "M100": (100, 1000, 100_000, 50),
        "M1000": (1000, 1000, 1_000_000, 50),
        "M10000": (10000, 1000, 10_000_000, 10),
    }
    if observed_scales != expected_scales:
        failures.append(f"dataset scale audit differs: {observed_scales}")

    with (SOURCE_DATA / "table_horizon_ablation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        horizon_rows = list(csv.DictReader(handle))
    observed_horizons = {
        int(row["horizon"]): (
            int(row["episodes"]),
            round(float(row["mean_return"]), 6),
            round(float(row["std_return"]), 6),
            round(float(row["median_return"]), 6),
            row["selected"].lower() == "true",
        )
        for row in horizon_rows
    }
    expected_horizons = {
        5: (5, -280283.901494, 1052.485097, -280282.875719, False),
        10: (5, -269373.945350, 486.880849, -269071.808946, True),
        20: (5, -272618.281878, 489.727079, -272624.757813, False),
    }
    if observed_horizons != expected_horizons:
        failures.append(f"Horizon ablation differs: {observed_horizons}")

    with (SOURCE_DATA / "table_training_sufficiency.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        training_rows = list(csv.DictReader(handle))
    observed_training = {
        row["scale"]: (int(row["total_epochs"]), int(row["selected_epoch"]))
        for row in training_rows
    }
    expected_training = {
        "M100": (50, 35),
        "M1000": (50, 20),
        "M10000": (10, 4),
    }
    if observed_training != expected_training:
        failures.append(f"training sufficiency differs: {observed_training}")

    for filename in (
        "table_all_world_models.tex",
        "table_data_scales.tex",
        "table_horizon_ablation.tex",
        "table_horizon_sensitivity.tex",
        "table_training_sufficiency.tex",
    ):
        if not (GENERATED / filename).exists():
            failures.append(f"missing appendix table: {filename}")

    audit = {
        "status": "pass" if not failures else "fail",
        "pdf_pages": pages,
        "figure_count": len(list(FIGURES.glob("fig*.pdf"))),
        "source_data_rows": observed_rows,
        "selected_world_models": selected,
        "failures": failures,
    }
    (GENERATED / "report_asset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
