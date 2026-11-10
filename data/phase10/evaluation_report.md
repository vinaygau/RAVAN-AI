# Phase 10 Evaluation Report

## Scope and evaluation discipline

The pilot contains 11524 labeled samples from 140 synthetic episodes. The primary held-out comparison uses 21 test episodes (1279 eligible rows). Phase 10 ablations reuse this holdout after Phase 6's primary result was already examined; they are exploratory. All preprocessing/model fitting for feature ablations uses only the existing train and validation episodes. The test episode IDs are unchanged.

## E1 Physics baseline versus ML

The selected logistic model achieved test PR-AUC 0.9715, recall 0.9517, and FPR 0.0010. The physics rule on the same episodes achieved PR-AUC 0.8239, recall 0.7770, and FPR 0.0000.

## E2 Feature ablation

Basic-state features: PR-AUC 0.9707, recall 0.9294; full safety features: PR-AUC 0.9715, recall 0.9517.
The basic-state and safety-feature models use the same Logistic Regression estimator, split, and threshold. See `experiment_results.json` for their complete metrics and feature lists.

## E3 Past-only history ablation

Snapshot-only PR-AUC 0.9715; snapshot plus 1-second lag features PR-AUC 0.9718 (absolute change +0.0002). Lag values are sampled from t−1 s within the same episode; early samples are imputed using training data. This small change on 21 test episodes is not strong evidence that history improves generalization.

## E4 Parameter-held-out check

Strong lead braking (absolute deceleration >= 5 m/s²) was excluded from fitting and selection: 14 episodes, 897 eligible rows. PR-AUC 0.9323; recall 0.9122. The small episode count limits confidence.

## E5 Observation robustness

Clean held-out PR-AUC 0.9715; with independent position noise σ=0.25 m and speed noise σ=0.5 m/s, 0.7090; with an additional 5% random feature-cell missingness, 0.9700. Total missing cells in that test matrix were 17.3%, including naturally undefined TTC/THW/DRAC. Labels remain from clean trajectories; these perturbations are sensitivity probes, not realistic sensor models.

## E6 Decision-layer review

Integrated recommendations on held-out samples: {'caution': 2, 'critical_warning': 103, 'increase_following_gap': 458, 'monitor': 947, 'prepare_to_slow': 167}. Physics-only action counts: {'critical_warning': 103, 'increase_following_gap': 523, 'monitor': 984, 'prepare_to_slow': 67}. Model/physics disagreement counts: {'agreement_or_partial': 1447, 'model_high_physics_low': 230}. These descriptive counts do not measure recommendation relevance; no human rubric or driver study was run.

## E7 Label sensitivity

Changing the TTC event threshold and horizon changes eligible coverage and positive prevalence. The complete 3×3 comparison is in `experiment_results.json`; these are label-distribution audits and do not retrain or re-evaluate models under each alternative target.

## E8 Explanation review

The exact additive Phase 8 explanation reconstructs test probabilities within 4.44e-16 absolute probability error and matches the Phase 6 prediction output within 5e-10.

## Project phase completion matrix

**Correction status:** Phase 6–10 describe the original TTC-threshold proxy experiment. The corrected, separate contact-prediction experiment is versioned as Contact Model v2 below; it does not retroactively change those Phase 6–10 artifacts.

- Phase 1: Scenario specification, parameter envelope, acceptance cases — complete. Evidence: `phase1/scenario_spec.md`, `config/phase1.json`, `tests/phase1_cases.csv`.
- Phase 2: Synthetic generator, episode dataset, metadata, quality report — complete; 140 episodes and 11,524 samples; synthetic-only. Evidence: `scripts/generate_phase2.py`, `data/phase2/trajectories.csv`, `data/phase2/metadata.json`, `data/phase2/quality_report.md`.
- Phase 3: Preprocessing, feature dictionary, hand-calculated checks — complete; 5 unit tests passed. Evidence: `scripts/preprocess_phase3.py`, `data/phase3/trajectories_features.csv`, `data/phase3/feature_dictionary.md`, `tests/test_phase3_features.py`.
- Phase 4: Versioned target protocol, labels and audit — complete; 6 unit tests passed; surrogate labels. Evidence: `phase4/labeling_protocol.md`, `scripts/label_phase4.py`, `data/phase4/trajectories_labeled.csv`, `data/phase4/label_audit.md`, `tests/test_phase4_labeling.py`.
- Phase 5: Physics baseline, rule traces and evaluation — complete; 5 unit tests passed. Evidence: `phase5/physics_baseline_protocol.md`, `scripts/physics_baseline_phase5.py`, `data/phase5/physics_baseline.csv`, `tests/test_phase5_baseline.py`.
- Phase 6: Grouped model comparison and held-out evaluation — complete; small synthetic holdouts. Evidence: `scripts/train_phase6.py`, `data/phase6/selected_model.joblib`, `data/phase6/test_metrics.json`, `data/phase6/robustness_metrics.json`.
- Phase 7: Decision policy, calibrated thresholds and traces — complete; 7 policy tests passed; advisory only. Evidence: `phase7/decision_policy.md`, `scripts/decision_engine_phase7.py`, `data/phase7/decision_trace.csv`, `tests/test_phase7_policy.py`.
- Phase 8: Faithful model explanation and examples — complete; exact logistic decomposition instead of SHAP; 2 checks passed. Evidence: `phase8/explainability_protocol.md`, `scripts/explain_phase8.py`, `data/phase8/explanation_examples.md`, `data/phase8/explanation_summary.json`, `tests/test_phase8_explanations.py`.
- Phase 9: Held-out trajectory dashboard and integration checks — complete; 4 integration checks and JS syntax passed; browser visual review stopped by Escape. Evidence: `dashboard/index.html`, `scripts/build_phase9_dashboard.py`, `data/phase9/dashboard_data.js`, `tests/test_phase9_integration.py`.
- Phase 10: Evaluation report, ablations, limitations and conclusions — complete; ablations exploratory on previously examined holdout. Evidence: `config/phase10.json`, `scripts/run_phase10.py`, `data/phase10/experiment_results.json`, `data/phase10/evaluation_report.md`.

## Corrected contact model v2

The original Phase 6 target included future TTC threshold events, while TTC is also an input feature. That made the task partly a proxy for reconstructing its own label. Contact Model v2 instead predicts independently simulated bumper contact during the next three seconds. It uses interpolated contact frames, excludes censored rows, keeps all rows from each episode in one split, compares several model families on validation data, calibrates on a separate episode set, and reports a same-test TTC-rule baseline. See `data/contact_v2/model/model_card.md`, `data/contact_v2/model/evaluation.json`, and `data/contact_v2/model/episode_splits.json`.

V2 test results are strong within this synthetic simulator (PR-AUC 0.991; recall 0.966 at a validation-selected threshold). These figures are not real-world crash metrics. The model still needs external validation with a trusted traffic simulator or observed driving data and scenario-shift testing.

## Conclusions supported by this pilot

- The end-to-end synthetic pipeline runs reproducibly from scenario generation through feature engineering, labeling, a transparent physics baseline, model scoring, recommendations, and local explanations.
- On the current synthetic future-TTC/contact label, the Logistic Regression model outperformed the current-state physics rule on the same held-out episodes. This is evidence about this operational target and scenario generator only.
- Feature and history ablations, observation sensitivity, and alternative-label prevalence are documented as exploratory diagnostics in `experiment_results.json`.

## Claims not supported

This pilot does not establish real-world collision probability, crash reduction, safety certification, driver acceptance, recommendation relevance, sensor robustness, or transfer to CARLA/road traffic. The target is generated from future TTC/contact and closely related indicators are model inputs. High scores can therefore reflect learning the target construction. The 21-episode main test and 14-episode hard-braking holdout are small, and frame-level metrics are correlated.

## Reproduction

Run `python scripts/run_phase10.py` from the project root. The held-out test episode IDs are preserved in `data/phase6/split_manifest.json` and copied into `experiment_results.json`; seeds, perturbation settings, feature sets, metrics, software versions, and caveats are recorded in the config/results files.
