# Phase 5 Physics Baseline Protocol

## Purpose

This baseline provides a transparent, hand-checkable reference before ML training. It reads only current-row trajectory fields and does not use Phase 4 target values to make predictions. Phase 4 labels are joined only after predictions are generated to calculate evaluation metrics.

## Current-state category and rule trace

The current category uses the Phase 4 TTC bands: low for undefined TTC or TTC > 5 s; moderate for 3 < TTC <= 5 s; high for 1.5 < TTC <= 3 s; critical for TTC <= 1.5 s or current sampled contact. Each row carries a reason code and rule basis so a result can be traced to the observed state.

## Primary future-conflict rule

The Phase 4 primary target is a future valid TTC <= 3 s or sampled contact within 3 s. Under a constant-relative-speed assumption, TTC decreases by elapsed time while closing. The physics baseline therefore predicts a future TTC threshold event when current TTC <= 3 s + 3 s = 6 s. If TTC is undefined because the vehicles are not closing or the gap is invalid, the rule predicts no future conflict. This is an intentionally simple baseline, not a learned or calibrated probability.

## Secondary future-contact rule

For the separate contact target, constant-relative-speed time to contact is current TTC. Predict contact within 3 s when valid current TTC <= 3 s. This rule ignores lead braking and acceleration and will miss contacts caused by later dynamic changes.

At a current contact row, both future predictions are blank: the episode has ended and the future is unobserved. Such rows are not included in metrics when the Phase 4 target is ineligible.

## Evaluation

Report frame-level confusion counts, precision, recall, F1, and false-positive rate on rows whose corresponding Phase 4 target is eligible. Rows in the same episode are correlated, so frame-level metrics are descriptive and can overstate the amount of independent evidence. Compare later models using the same eligible rows and report episode/event-level metrics as well.

The rules inherit provisional TTC thresholds and synthetic-distribution limitations. Performance is not evidence of real-world safety or collision probability calibration.
