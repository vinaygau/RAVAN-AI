# Phase 3 Preprocessing Quality Report

- Pipeline version: `phase3.1`
- Episodes: 700
- Samples: 58310
- Scenario episode counts: {'S1': 100, 'S2': 100, 'S3': 100, 'S4': 100, 'S5': 100, 'S6': 100, 'S7': 100}
- Contact episodes: 221
- Gap range (m): -1.01e-07 to 187.695067134
- Stored Phase 2 derived-feature mismatches: 0
- Negative-speed samples: 0
- Non-monotonic position transitions: 0
- Irregular sample intervals: 0
- Feature calculations use current-row information only; no future observations or labels are used.
- A shortened final interval is allowed only when Phase 2 interpolates the first contact within a time step; contact is recorded at zero bumper gap.
- Phase 2 feature values are audited with the stated rounding tolerance because source positions and speeds were serialized to six decimals.
- No risk or future-hazard labels are created.
