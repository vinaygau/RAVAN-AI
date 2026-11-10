# Phase 6 Model Evaluation

- Selected model: `LogisticRegression` (highest validation PR-AUC).
- Episode split counts: {'train': 98, 'validation': 21, 'test': 21}; eligible frame counts: {'train': 6152, 'validation': 1401, 'test': 1279}.
- Validation selection metric: validation average precision (PR-AUC).
- Final model refit on train + validation episodes, then evaluated once on held-out test episodes.
- Test confusion matrix: TN=1009, FP=1, FN=13, TP=256.
- Test precision=0.9961, recall=0.9517, F1=0.9734, PR-AUC=0.9715, Brier=0.0174, FPR=0.0010.
- Test event-level recall: 1.0; negative-episode alarm rate: 0.0.
- Physics baseline on the same test episodes: PR-AUC=0.8239, recall=0.7770, FPR=0.0000.
- Strong-braking held-out check (|lead deceleration| >= 5.0 m/s^2): 0.9323 PR-AUC, 0.9122 recall over 897 eligible frames from 14 held-out episodes; selected RandomForest using only other parameter ranges.
- Candidate validation metrics are in `validation_metrics.json`; episode assignments and class counts are in `split_manifest.json`.
- Future target columns, identifiers, time, and scenario family were excluded from model features.
- The target is defined from future TTC/contact while current TTC and related physics features are inputs, so high scores mainly show learnability of this operational target.

## Limits

Synthetic one-dimensional trajectories; labels are provisional TTC/contact surrogate events; frame-level metrics are correlated within episodes.
Frame-level test metrics remain correlated within episodes. This pilot is small and synthetic; results describe only the generated scenario distribution and operational label rule. They do not establish real-world transfer, calibrated collision probability, or safety effectiveness.
