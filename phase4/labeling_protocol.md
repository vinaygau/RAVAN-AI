# Phase 4 Labeling Protocol

## Targets

Phase 4 creates a current-state category and two future targets. They are separate columns with separate meanings.

### Current risk category

Assign `current_risk_category` from the current sample's TTC:

- `low`: TTC is undefined or TTC > 5 seconds, unless contact is already present.
- `moderate`: 3 < TTC <= 5 seconds.
- `high`: 1.5 < TTC <= 3 seconds.
- `critical`: TTC <= 1.5 seconds or current contact (`gap_m <= 0`).

Undefined TTC is categorized as low only for this provisional longitudinal TTC-only baseline. It does not assert that a non-closing or unobserved interaction is safe. THW and DRAC do not alter this category; avoiding undocumented tie-break rules keeps the initial labels deterministic.

### Primary future target

`future_conflict_within_3s` is 1 when at least one **later** sample in the interval `(t, t + 3 seconds]` has valid TTC <= 3 seconds, or sampled contact occurs in that interval. It is 0 when the full three-second horizon is observed and neither event occurs. The current sample is excluded from the future window.

This is a supervised target for predicting the future presence of a TTC threshold event in the synthetic scenario distribution. Because TTC is also an input feature candidate, the task may mainly learn a threshold-presence proxy; it is not evidence of independent crash prediction.

### Secondary severe-outcome target

`future_contact_within_3s` is 1 when sampled contact occurs in `(t, t + 3 seconds]`, and 0 when the full horizon is observed with no such contact. Report it separately from the primary surrogate-conflict target. It represents contact in this toy simulator, not a real-world crash label.

## Eligibility and censoring

For either target, positive events observed within the horizon are labeled 1 even if contact ends the trace before the horizon completes. A negative label requires the trace to reach at least `t + 3 seconds`. If the trace ends before then without an observed positive event, set the corresponding `*_eligible` field to 0 and leave the target blank. The target at a current contact sample is censored because the target interval is strictly in the future.

This avoids labeling unobserved future time as negative. Future trajectory values are used only to construct label columns and audit summaries; they must not be passed into model features.

## Provisional thresholds and caveats

The TTC bands and three-second horizon carry forward the research document's proposed starting values. They are operational choices for this prototype, not universal safety limits. The labels are specific to a same-lane, one-dimensional synthetic simulator and its discrete 0.1-second sampling. Contact can be slightly overlapped at the first sampled contact.

Before model training, inspect examples at the TTC boundaries, review episode-level class balance, and compare results with the TTC/rule baseline. Any threshold or horizon change must increment the labeling protocol version and trigger label regeneration.
