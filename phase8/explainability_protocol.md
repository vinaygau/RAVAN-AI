# Phase 8 Explainability Protocol

## Method

The selected Phase 6 model is a logistic-regression pipeline with median imputation and standard scaling. The `shap` package is not available in the project runtime. For this model, use its exact additive decomposition in standardized feature space:

`logit(p) = intercept + sum(coefficient_i * standardized_feature_i)`

Each `coefficient_i * standardized_feature_i` is reported as a signed contribution to model log-odds relative to the all-zero standardized reference. Positive values increase the model score; negative values decrease it. The sigmoid of the reconstructed logit must match the model's predicted probability within the configured tolerance.

This is a faithful model explanation, but it is not a SHAP value calculation. Correlated features such as gap, TTC, THW, and DRAC can share or redistribute apparent importance; contributions describe this fitted model and reference point, not causal effects or unique feature importance.

## Scope

Explain only held-out Phase 6 test predictions for quantitative summaries and example cases. Preserve the test episode IDs from the split manifest. Local examples include representative true-positive, true-negative, false-positive, and false-negative predictions when present, plus available model/physics disagreement examples.

Future labels are shown only as the outcome being explained; they are not part of the model input or attribution formula. Feature values are the current/past-only Phase 6 inputs. Recommendation actions are contextual output from Phase 7 and are not predicted by the logistic model.

## Review checks and limits

- Recompose each test probability from intercept and additive contributions; fail if error exceeds tolerance.
- Compare explanation input row scores with the saved Phase 6 test predictions and Phase 7 probability trace.
- Review the largest absolute global contributions and signed local contributions for representative cases.
- Report the synthetic label definition and model scope beside the examples.

The target is defined from future TTC/contact while current TTC-derived features are model inputs. High performance and explanations therefore describe how this model represents a synthetic TTC/contact surrogate task; they do not explain real-world crash causes or establish safety.
