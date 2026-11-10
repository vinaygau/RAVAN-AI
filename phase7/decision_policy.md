# Phase 7 Decision Policy

## Inputs and boundary

The decision engine combines the Phase 6 model score for `future_conflict_within_3s` with current TTC, time headway, contact status, and required-input quality. It uses no future labels or future trajectory rows to make a recommendation. Model-score thresholds are calibrated on Phase 6 validation episodes scored by a model fit on the training episodes only. The saved Phase 6 model is then used for the trajectory-wide trace.

The engine produces advice for a simulated driver-assistance display only. It does not brake, steer, or control a vehicle.

## Threshold calibration

On validation data, select a recall-weighted probability threshold by maximizing F2 and a precision-weighted threshold by maximizing F0.5. The lower threshold becomes the caution threshold and the higher becomes the prepare-to-slow threshold. The selected values and validation metrics are saved in `policy_calibration.json`. Threshold choice is explicitly a prototype trade-off, not a safety standard.

## Deterministic action order

Apply the first matching rule:

1. **Critical warning:** current sampled contact, or valid TTC <= 1.5 s. Physical evidence overrides a lower model score.
2. **Prepare to slow:** valid TTC <= 3 s, or model score >= the prepare-to-slow threshold.
3. **Increase following gap:** valid TTC <= 5 s, or valid THW <= 2 s.
4. **Caution:** model score >= the caution threshold.
5. **Monitor:** otherwise.

If required current state fields are absent or invalid, emit `caution` with an input-quality warning unless a known current-contact condition triggers the critical override.

## Conflict logging

- `physics_high_model_low`: TTC <= 3 s while score < caution threshold. Keep the physical prepare-to-slow recommendation and log the disagreement.
- `model_high_physics_low`: score >= caution threshold while TTC is undefined or > 3 s. Use the score-driven caution/prepare recommendation and log the disagreement.
- `agreement_or_partial`: neither disagreement condition holds.

Every decision trace records the action, reason code, model score, current physics category, data-quality status, and disagreement status. This policy is deterministic so decisions can be reviewed and tested.

## Limits

The model predicts a future TTC/contact surrogate label generated from the same physical indicators supplied to the policy. The scores are not real-world collision probabilities. Scenario diversity is narrow, the data is synthetic, and the recommendation policy has not been evaluated with drivers or in a traffic simulator.
