# Main system and final fresh-seed results

## Development main matrix (seeds 42–46)

Each dataset scale uses its validation-only BestWM. Returns are mean ± population standard deviation; higher is better.

| Dataset + BestWM | CEM | iCEM | MPPI | MB-PPO + KL |
|---|---:|---:|---:|---:|
| M-100 + GRU e35 | -274,238 ± 588 | -2,646,910 ± 4,581 | -348,211 ± 5,687 | -281,009 ± 131 |
| M-1000 + Transformer-2L e20 | -267,826 ± 407 | -334,597 ± 4,503 | **-221,072 ± 1,475** | -285,326 ± 158 |
| M-10000 + Transformer-2L e4 | -269,218 ± 411 | -321,264 ± 36,310 | **-223,774 ± 578** | -285,217 ± 283 |

The fixed official BC baseline has mean return -288,175 on the same five development seeds. CEM is selected at M-100; reference MPPI is selected at M-1000 and M-10000. The iCEM toy objective passes, so its simulator failures are recorded as World Model/planner exploitation interactions rather than evidence that iCEM is generally unsuitable for IB.

## M-1000 World Model → control cross experiment

| World Model | dynamics mean(H5,H10,H20) | MPPI | MB-PPO + KL |
|---|---:|---:|---:|
| MLP | 0.385195 | -338,535 ± 5,384 | -287,698 ± 174 |
| GRU | 0.331771 | -303,188 ± 5,791 | -284,715 ± 174 |
| LSTM | 0.402662 | -309,541 ± 5,830 | **-277,127 ± 132** |
| Transformer-2L | **0.317736** | -221,072 ± 1,475 | -285,326 ± 158 |
| Transformer-4L | 0.337813 | **-216,331 ± 2,810** | -283,859 ± 244 |

Validation dynamics ranking is only partially associated with downstream control. Transformer-4L gives the best MPPI return, while LSTM gives the best learned-policy return. These simulator results do not alter the validation-only BestWM selection: BestWM@M1000 remains Transformer-2L e20.

## Final untouched seeds 100–109

| Selected system | mean ± std | median | Δ vs BC | win rate vs BC |
|---|---:|---:|---:|---:|
| M-100 + GRU e35 + CEM | -274,181 ± 396 | -274,214 | +14,229 | 100% |
| M-1000 + Transformer-2L e20 + MPPI | **-220,129 ± 1,699** | -220,755 | +68,282 | 100% |
| M-10000 + Transformer-2L e4 + MPPI | -223,876 ± 1,033 | -223,938 | +64,535 | 100% |

The fixed BC mean on fresh seeds is -288,411. Seeds 100–109 were first used only after World Models, planners, strategy settings, and selected systems were frozen.

## Source-of-truth records

- World Models: `outputs/metrics/world_model_5x3_common_validation*.csv/json`
- Planner audit: `outputs/metrics/icem_*`, `outputs/metrics/mppi_reference_*`, and `outputs/metrics/reference_planner_freeze.json`
- Development main matrix: `outputs/metrics/main_system_matrix_development*.csv/json`
- World Model → control cross: `outputs/metrics/world_model_control_cross_m1000*.csv/json`
- Final fresh seeds: `outputs/metrics/final_selected_systems_fresh_seeds*.csv/json`
- MB-PPO histories: `outputs/metrics/mbppo_main_*_training.csv` and `outputs/metrics/mbppo_cross_*_training.csv`
- Machine-checkable record audit: `outputs/metrics/experiment_record_audit.json`

CSV/JSON are the source of truth; figures can be regenerated without training or simulator reruns.
