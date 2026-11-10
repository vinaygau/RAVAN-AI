# Phase 2 Synthetic Dataset Quality Report

Dataset: `synthetic_longitudinal_pilot_v1`

## Generation summary

- Episodes: 140
- Trajectory samples: 11524
- Scenario families: S1 (20), S2 (20), S3 (20), S4 (20), S5 (20), S6 (20), S7 (20)
- Contact episodes: 47
- Recorded episode duration: 0.8–12.0 s
- Risk labels: none (intentionally not assigned in Phase 2)

## Feature coverage

| Feature | Minimum | Maximum | Non-null samples |
|---|---:|---:|---:|
| `ego_speed_mps` | 0.000000 | 29.971689 | 11524 |
| `lead_speed_mps` | 0.000000 | 29.527094 | 11524 |
| `gap_m` | -1.970870 | 221.505848 | 11524 |
| `relative_speed_mps` | -16.007863 | 24.629669 | 11524 |
| `ttc_s` | 0.001850 | 6494.894725 | 2715 |
| `thw_s` | 0.000594 | 6878.692403 | 9566 |
| `drac_mps2` | 0.000000 | 2424.945117 | 2715 |

## Missing and validity handling

- TTC is undefined in 8809 samples and stored as a blank value with `ttc_valid=0`.
- THW is undefined in 1958 samples and stored as a blank value with `thw_valid=0`.
- DRAC is undefined in 8809 samples and stored as a blank value with `drac_valid=0`.
- Contact is detected at the first 0.1-second sample with a non-positive gap. The vehicles may overlap slightly at that sample because contact is not interpolated between frames; the gap is a discrete-time indicator, not a continuous-time impact estimate.
- No rows are emitted after the first contact sample; post-contact motion is outside this toy simulator.

## Data dictionary

| Column | Meaning |
|---|---|
| `episode_id`, `scenario_id`, `seed`, `step` | Stable episode identity, scenario family, episode seed, and sample index. |
| `time_s` | Elapsed simulation time in seconds. |
| `ego_front_x_m`, `lead_front_x_m` | Front-bumper positions in metres along the positive travel axis. |
| `ego_speed_mps`, `lead_speed_mps` | Vehicle speeds in metres per second. |
| `ego_acceleration_mps2`, `lead_acceleration_mps2` | Applied acceleration over the interval ending at this sample; zero at the initial sample. |
| `lead_length_m` | Lead vehicle length used in the bumper-gap calculation. |
| `gap_m` | Lead front position minus lead length minus ego front position; may be non-positive at the first sampled contact. |
| `relative_speed_mps` | Ego speed minus lead speed; positive means closing in this same-lane setup. |
| `ttc_s`, `thw_s`, `drac_mps2` | Simplified derived indicators; blank when their validity conditions do not hold. |
| `ttc_valid`, `thw_valid`, `drac_valid` | 1 when the corresponding value is defined, otherwise 0. |
| `contact` | 1 at the first sampled non-positive gap; the episode ends at that row. |

## Limitations

This is a synthetic one-dimensional, same-lane dataset generated from scripted kinematics. It is not collected from CARLA or real vehicles, does not represent sensor noise, and does not support claims about real-world collision probability or safety. Scenario categories describe generation setups, not hazard labels. The configured ranges are provisional and should be reviewed before modeling.

## Reproducibility

Run `python scripts/generate_phase2.py` from the project root. The base seed, per-episode seed policy, generation configuration, episode parameters, and quality summary are recorded in `metadata.json`.
