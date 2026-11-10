# Phase 4 Label Audit

- Protocol: `phase4.1`; future horizon: 3.0 s.
- Episodes: 140; labeled samples: 11524.
- Current risk categories: {'critical': 731, 'high': 518, 'low': 9792, 'moderate': 483}.
- Primary future-conflict target: {'0': 6781, '1': 2051}; eligible=8832, censored=2692.
- Secondary future-contact target: {'0': 7444, '1': 1243}; eligible=8687, censored=2837.
- Current contact samples: 47.

## Primary target by scenario

| Scenario | Eligible samples | Positive samples |
|---|---:|---:|
| S1 | 1340 | 0 |
| S2 | 875 | 722 |
| S3 | 1371 | 888 |
| S4 | 1026 | 441 |
| S5 | 1360 | 0 |
| S6 | 1430 | 0 |
| S7 | 1430 | 0 |

## TTC boundary review

| TTC boundary (s) | Exact samples | Within ±0.05 s |
|---:|---:|---:|
| 1.5 | 0 | 39 |
| 3.0 | 0 | 24 |
| 5.0 | 0 | 16 |

## Audit interpretation

The primary label uses only later samples in the defined future window. A positive event is retained even if contact truncates the episode; a negative requires full horizon coverage. Rows without sufficient future observation and no observed event are censored (blank target, eligible=0). Current risk categories and future targets are distinct.

## Limitations

TTC thresholds are provisional surrogate-conflict criteria. The primary target is not a real-world collision probability; the secondary contact target is a severe outcome in this synthetic simulator.
The primary target contains a future TTC threshold event, while TTC is also a candidate current feature. Model results must be compared with a transparent TTC/rule baseline and described as prediction of this operational surrogate event, not independent real-world crash prediction. Thresholds require sensitivity analysis before model evaluation.
