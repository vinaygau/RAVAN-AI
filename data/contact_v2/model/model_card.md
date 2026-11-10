# Synthetic Contact Model v2 Evaluation

## Task and data

Predict whether the simulator reaches bumper contact during the next 3 seconds. Training uses only current measurements and one-second past changes. Future labels and episode/scenario identifiers are not model inputs. Episodes, not rows, are separated between train, validation, calibration, and test.

- Source: `data\contact_v2\phase4\trajectories_labeled.csv` (700 episodes; 43719 eligible rows).
- Contact episodes: 221.
- Selected using validation average precision: `random_forest`.
- Episode split sizes: `{'train': 424, 'validation': 107, 'calibration': 67, 'test': 102}`.

## Locked test results

- Model: PR-AUC / average precision **0.991**, ROC-AUC **0.998**, precision **0.893**, recall **0.966**, F2 **0.951**, Brier **0.009**.
- Same-test TTC rule baseline (current TTC ≤ 3 s): PR-AUC **0.881**, recall **0.903**, precision **0.961**.
- Event-level recall: **1.000** across 32 positive test episodes.

## Interpretation and limits

This is a materially cleaner ML experiment than the earlier TTC-threshold target: its outcome is separately simulated bumper contact, and episode overlap is prevented across splits. It is still a prototype trained only on synthetic toy data. Good test scores show it learned this simulator, not that it predicts real crashes. Before any real safety claim, use a validated simulator such as CARLA and/or independently collected driving data, define a realistic outcome, test across unseen scenarios, and review calibration and false alarms with domain experts.

## Run predictions

```powershell
python scripts/predict_contact_model_v2.py --model data/contact_v2/model/contact_model.joblib --input path/to/trajectory_features.csv --output data/contact_v2/model/predictions.csv
```

The input needs the same current kinematic and derived-feature columns produced by Phase 3. The output contains a calibrated probability, the validation-selected alert threshold, and an alert flag. This output is for research review only.
