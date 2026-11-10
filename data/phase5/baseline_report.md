# Phase 5 Physics Baseline Results

## Primary future-conflict target

- Eligible frames: 8832
- Confusion matrix: {'tp': 1747, 'fp': 42, 'fn': 304, 'tn': 6739}
- Precision / recall / F1: 0.9765231973169368 / 0.8517796196977084 / 0.9098958333333333
- False-positive rate: 0.006193776729096004

## Secondary future-contact target

- Eligible frames: 8687
- Confusion matrix: {'tp': 1132, 'fp': 2, 'fn': 111, 'tn': 7442}
- Precision / recall / F1: 0.9982363315696648 / 0.910699919549477 / 0.9524610854017669
- False-positive rate: 0.00026867275658248256

## Trace and interpretation

- Reason codes: {'CURRENT_CONTACT': 47, 'TTC_GT_5S': 1030, 'TTC_LE_1_5S': 684, 'TTC_LE_3S': 518, 'TTC_LE_5S': 483, 'TTC_UNDEFINED_OR_NOT_CLOSING': 8762}.
- Each row in `physics_baseline.csv` includes the current TTC basis, chosen category, rule reason, thresholds, and predictions.
- Predictions were computed from current-row fields only; Phase 4 labels were used only for the metrics above.
- Metrics are frame-level. Nearby frames from one episode are correlated and should not be treated as independent samples.

## Limitations

The rules assume current relative speed persists and ignore acceleration, changing gaps, and other traffic. TTC cutoffs are provisional surrogate criteria.
The primary target contains a future TTC threshold event. This rule is a transparent constant-relative-speed comparator, not a calibrated model or a real-world safety claim.
