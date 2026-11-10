# Phase 3 Feature Dictionary

All distances are metres, time is seconds, speed is metres per second, and acceleration is metres per second squared. Positions refer to front bumpers along the one-dimensional positive travel axis. These definitions apply only to the same-lane longitudinal prototype.

## Source trajectory fields

| Field | Definition | Availability |
|---|---|---|
| `episode_id`, `scenario_id`, `seed`, `step` | Episode identity, scenario family, episode seed, and sample index. | Metadata; do not use identifiers as model features. |
| `time_s` | Elapsed simulation time. | Current sample. |
| `ego_front_x_m`, `lead_front_x_m` | Front-bumper coordinates. | Current sample. |
| `ego_speed_mps`, `lead_speed_mps` | Vehicle speeds. | Current sample. |
| `ego_acceleration_mps2`, `lead_acceleration_mps2` | Scripted accelerations recorded for the interval ending at the sample; zero at the initial sample. | Current/past simulation state. |
| `lead_length_m` | Lead vehicle length used for gap geometry. | Scenario metadata. |

## Canonical derived features

| Feature | Formula | Validity rule | Interpretation |
|---|---|---|---|
| `gap_m` | `lead_front_x_m - lead_length_m - ego_front_x_m` | Always computed from positions; `gap_m <= 0` denotes sampled contact/overlap. | Bumper-to-bumper longitudinal gap. |
| `relative_speed_mps` | `ego_speed_mps - lead_speed_mps` | Always defined. | Positive values indicate the ego vehicle is closing on the lead vehicle. |
| `ttc_s` | `gap_m / relative_speed_mps` | Only when `gap_m > 0` and `relative_speed_mps > 0`; otherwise blank and `ttc_valid=0`. | Constant-relative-speed estimate; it ignores acceleration and is not a collision probability. |
| `thw_s` | `gap_m / ego_speed_mps` | Only when `gap_m > 0` and `ego_speed_mps > 0`; otherwise blank and `thw_valid=0`. | Current distance expressed as ego travel time. |
| `drac_mps2` | `relative_speed_mps^2 / (2 * gap_m)` | Only when `gap_m > 0` and `relative_speed_mps > 0`; otherwise blank and `drac_valid=0`. | Simplified deceleration indicator under constant-speed lead assumptions. |
| `ttc_valid`, `thw_valid`, `drac_valid` | 1 if the corresponding value is defined, otherwise 0. | Always defined. | Prevents undefined values from being confused with zero. |
| `contact` | 1 when `gap_m <= 0`, otherwise 0. | Always defined. | First sampled contact ends the episode. This is not a continuous-time collision label. |

The pipeline recomputes the derived features from positions and speeds rather than trusting the Phase 2 cached values. It audits those values before writing canonical output. It uses only the current row; no future trajectory samples are read to calculate features. Phase 3 does not add hazard labels, fit data-dependent transforms, or create train/test splits.
