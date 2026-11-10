# Phase 2 Synthetic Dataset Quality Report

Dataset: `synthetic_contact_prediction_v2`

## Generation summary

- Episodes: 700
- Trajectory samples: 58310
- Scenario families: S1 (100), S2 (100), S3 (100), S4 (100), S5 (100), S6 (100), S7 (100)
- Contact episodes: 221
- Recorded episode duration: 0.7–12.0 s
- Risk labels: none (intentionally not assigned in Phase 2)

## Feature coverage

| Feature | Minimum | Maximum | Non-null samples |
|---|---:|---:|---:|
| `ego_speed_mps` | 0.000000 | 29.971689 | 58310 |
| `lead_speed_mps` | 0.000000 | 29.971689 | 58310 |
| `gap_m` | 0.000000 | 187.695067 | 58310 |
| `relative_speed_mps` | -18.232269 | 24.369203 | 58310 |
| `ttc_s` | 0.000152 | 7563.818079 | 14619 |
| `thw_s` | 0.000054 | 50518.503677 | 48329 |
| `drac_mps2` | 0.000000 | 23168.068752 | 14619 |

## Missing and validity handling

- TTC is undefined in 43691 samples and stored as a blank value with `ttc_valid=0`.
- THW is undefined in 9981 samples and stored as a blank value with `thw_valid=0`.
- DRAC is undefined in 43691 samples and stored as a blank value with `drac_valid=0`.
- Contact is interpolated within a time step; the final recorded contact frame has a tiny non-positive bumper gap to remain stable after CSV rounding.
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
| `gap_m` | Lead front position minus lead length minus ego front position; at interpolated contact the recorded value is approximately zero (tiny negative rounding guard). |
| `relative_speed_mps` | Ego speed minus lead speed; positive means closing in this same-lane setup. |
| `ttc_s`, `thw_s`, `drac_mps2` | Simplified derived indicators; blank when their validity conditions do not hold. |
| `ttc_valid`, `thw_valid`, `drac_valid` | 1 when the corresponding value is defined, otherwise 0. |
| `contact` | 1 at first contact; v2 interpolates within a time step and the episode ends at that row. |

## Limitations

This is a synthetic one-dimensional, same-lane dataset generated from scripted kinematics. It is not collected from CARLA or real vehicles, does not represent sensor noise, and does not support claims about real-world collision probability or safety. Scenario categories describe generation setups, not hazard labels. The configured ranges are provisional and should be reviewed before modeling.

## Reproducibility

Run `python scripts/generate_phase2.py` from the project root. The base seed, per-episode seed policy, generation configuration, episode parameters, and quality summary are recorded in `metadata.json`.
