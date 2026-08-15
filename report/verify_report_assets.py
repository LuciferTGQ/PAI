"""Lightweight consistency checks for the final report artifact.

This verifier deliberately checks only report-facing assets and exported data.
It does not rerun training, the simulator, or the historical data audit.
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

    for index in range(1, 9):
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
        failures.append(f"World Model matrix has {len(matrix)} unique scale-architecture pairs")
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
    ]
    for pattern in forbidden:
        if re.search(pattern, visible_text, flags=re.IGNORECASE):
            failures.append(f"forbidden report-facing term found: {pattern}")

    required_phrases = [
        "原始行为数据参考值（Original Behavior）",
        "simulator reward",
        "不参与World Model",
        "behavior KL",
        "Industrial Applications and Deployment Considerations",
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
