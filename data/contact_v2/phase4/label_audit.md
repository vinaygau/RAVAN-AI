# Phase 4 Label Audit

- Protocol: `phase4.1`; future horizon: 3.0 s.
- Episodes: 700; labeled samples: 58310.
- Current risk categories: {'critical': 3428, 'high': 2362, 'low': 50466, 'moderate': 2054}.
- Primary future-conflict target: {'0': 34710, '1': 9507}; eligible=44217, censored=14093.
- Secondary future-contact target: {'0': 37904, '1': 5815}; eligible=43719, censored=14591.
- Current contact samples: 221.

## Primary target by scenario

| Scenario | Eligible samples | Positive samples |
|---|---:|---:|
| S1 | 6970 | 0 |
| S2 | 5432 | 3126 |
| S3 | 5556 | 4007 |
| S4 | 5279 | 2374 |
| S5 | 7040 | 0 |
| S6 | 6930 | 0 |
| S7 | 7010 | 0 |

## TTC boundary review

| TTC boundary (s) | Exact samples | Within ±0.05 s |
|---:|---:|---:|
| 1.5 | 0 | 183 |
| 3.0 | 0 | 130 |
| 5.0 | 0 | 76 |

## Audit interpretation

The primary label uses only later samples in the defined future window. A positive event is retained even if contact truncates the episode; a negative requires full horizon coverage. Rows without sufficient future observation and no observed event are censored (blank target, eligible=0). Current risk categories and future targets are distinct.

## Limitations

TTC thresholds are provisional surrogate-conflict criteria. The primary target is not a real-world collision probability; the secondary contact target is a severe outcome in this synthetic simulator.
The primary target contains a future TTC threshold event, while TTC is also a candidate current feature. Model results must be compared with a transparent TTC/rule baseline and described as prediction of this operational surrogate event, not independent real-world crash prediction. Thresholds require sensitivity analysis before model evaluation.
