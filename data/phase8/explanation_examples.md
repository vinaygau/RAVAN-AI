# Phase 8 Explanation Examples

Method: exact additive logistic-regression contributions in standardized feature space. These are not SHAP values. Positive log-odds contributions raise the score; negative contributions lower it.
Probability recomposition maximum error: 4.44e-16; difference from Phase 6 predictions: 5e-10; difference from Phase 7 traces: 5e-10.

## Global test-set contributions

- `thw_s`: mean absolute contribution 4.7972 log-odds; coefficient -6.8242.
- `ego_acceleration_mps2`: mean absolute contribution 1.7317 log-odds; coefficient 2.2118.
- `relative_speed_mps`: mean absolute contribution 1.7102 log-odds; coefficient 2.9958.
- `lead_acceleration_mps2`: mean absolute contribution 1.0431 log-odds; coefficient -1.2412.
- `gap_m`: mean absolute contribution 1.0234 log-odds; coefficient -1.5425.
- `lead_speed_mps`: mean absolute contribution 0.8681 log-odds; coefficient -1.2445.
- `thw_valid`: mean absolute contribution 0.6234 log-odds; coefficient 1.1639.
- `ego_speed_mps`: mean absolute contribution 0.4831 log-odds; coefficient 0.6765.

## Representative held-out cases

### True Positive
Episode P2-0033 (S2), step 11: target=1, model score=1.0000, Phase 7 action=`critical_warning`, disagreement=`agreement_or_partial`.
- `drac_mps2` = 621.707773276: +28.9848 log-odds (raises score).
- `relative_speed_mps` = 19.445952000: +10.0740 log-odds (raises score).
- `gap_m` = 0.304118000: +2.7155 log-odds (raises score).
- `thw_s` = 0.012228936: +1.9616 log-odds (raises score).
- `lead_speed_mps` = 5.422770: +1.3239 log-odds (raises score).

### True Negative
Episode P2-0078 (S4), step 66: target=0, model score=0.0000, Phase 7 action=`monitor`, disagreement=`agreement_or_partial`.
- Scenario metadata: lead braking began at 3.71 s, 2.89 s before this sample; the scripted onset time is not a model feature.
- `thw_s` = 35.030599698: -17.4535 log-odds (lowers score).
- `ego_acceleration_mps2` = -3.818225: -3.1160 log-odds (lowers score).
- `relative_speed_mps` = -1.823597000: -1.6726 log-odds (lowers score).
- `lead_speed_mps` = 3.102858: +1.6643 log-odds (raises score).
- `ego_speed_mps` = 1.279261: -1.1548 log-odds (lowers score).

### False Positive
Episode P2-0056 (S3), step 44: target=0, model score=0.7100, Phase 7 action=`prepare_to_slow`, disagreement=`model_high_physics_low`.
- Scenario metadata: lead braking began at 4.21 s, 0.19 s before this sample; the scripted onset time is not a model feature.
- `ttc_s` = 165.052741781: +2.2474 log-odds (raises score).
- `ego_acceleration_mps2` = 0.000000: +1.1668 log-odds (raises score).
- `lead_speed_mps` = 22.273942: -1.1488 log-odds (lowers score).
- `lead_acceleration_mps2` = -2.727057: +1.1074 log-odds (raises score).
- `thw_s` = 1.996344335: +0.8615 log-odds (raises score).

### False Negative
Episode P2-0052 (S3), step 24: target=1, model score=0.0073, Phase 7 action=`monitor`, disagreement=`agreement_or_partial`.
- Scenario metadata: lead braking is scheduled at 2.95 s, 0.55 s after this sample; the future schedule is not a model feature.
- `ego_acceleration_mps2` = 0.000000: +1.1668 log-odds (raises score).
- `gap_m` = 50.097237000: -1.0857 log-odds (lowers score).
- `lead_acceleration_mps2` = 0.000000: -0.7530 log-odds (lowers score).
- `relative_speed_mps` = 0.000000000: -0.6655 log-odds (lowers score).
- `lead_speed_mps` = 18.886470: -0.6517 log-odds (lowers score).

### Model Physics Disagreement
Episode P2-0052 (S3), step 53: target=1, model score=0.9999, Phase 7 action=`prepare_to_slow`, disagreement=`model_high_physics_low`.
- Scenario metadata: lead braking began at 2.95 s, 2.35 s before this sample; the scripted onset time is not a model feature.
- `relative_speed_mps` = 12.027877000: +5.9772 log-odds (raises score).
- `lead_acceleration_mps2` = -5.229512: +2.8147 log-odds (raises score).
- `ego_acceleration_mps2` = 0.000000: +1.1668 log-odds (raises score).
- `lead_speed_mps` = 6.858593: +1.1132 log-odds (raises score).
- `thw_s` = 1.920167083: +0.9038 log-odds (raises score).

## Limits

These contributions exactly decompose this logistic model's log-odds relative to the zero standardized reference; they do not identify causal risk factors. TTC, gap, relative speed, THW, and DRAC are correlated. The target is itself derived from future TTC/contact in a synthetic one-dimensional simulator, so these examples explain model behavior on that operational task only.
