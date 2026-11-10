# Phase 7 Decision Policy Run

- Samples processed: 11524.
- Validation-calibrated thresholds: {'caution': 0.1887004484353369, 'prepare_to_slow': 0.39806028059909504}.
- Recommendation counts: {'caution': 77, 'critical_warning': 731, 'increase_following_gap': 2307, 'monitor': 6661, 'prepare_to_slow': 1748}.
- Disagreement counts: {'agreement_or_partial': 9567, 'model_high_physics_low': 1957}.
- Data quality counts: {'complete': 11524}.
- Every output row includes model probability, current TTC/THW, current risk category, recommendation, reason, disagreement, and input quality.
- The output is advisory-only and makes no vehicle-control commands.

## Limits

This is a transparent decision-support policy over synthetic same-lane data. It does not actuate vehicle controls; thresholds are provisional and do not establish safety.
High model scores and physical indicators are related because the model target is defined from future TTC/contact. Disagreement rates should be reviewed by scenario before any user-facing integration.
