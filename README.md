# Math for ML Coursework Code

## Setup
Install the dependencies first: `pip install -r requirements.txt`

## Shared conventions
Every experiment notebook follows the same rules, so their results can be compared directly:

- **Split:** `split()` gives 60% train / 20% validation / 20% test (`train_test_split`, `random_state = SEED = 42`).
- **Scales:** n = 20, 30, 40 and data fractions 25%, 50%, 100%. A fraction takes the first part of the train and
  validation splits; the test split is always the full 20%.
- **Preprocessing:** no normalisation (the features already have mean ≈ 0 and variance ≈ 1; see
  `data_analysis.ipynb`). Most models use the sign of each feature (`binarise`).
- **Feature selection:** a greedy forward search that adds one feature or a pair of features per step, scored by a
  lookup table on validation. It runs separately for every (n, fraction).
- **Model choice:** hyperparameters are tuned on validation, and only the validation-best model is scored on test.
- **Weights:** the best model of each (n, fraction) is saved to `weights/<model>/n<n>/<pct>pct/`.

## Files

| Name | Purpose |
|---|---|
| `Datasets/` | Kryptonite datasets (labelled and hidden) |
| `PaperAndInstructions/` | Coursework paper and instructions |
| `weights/` | Best saved model for every model, n and data fraction (layout below) |
| `hiddenlabels/` | Predicted labels for the four hidden datasets (the submission) |
| `data_analysis.ipynb` | Mean/variance analysis (no normalisation needed), value ranges of positive, negative and absolute values, and correlation analysis (feature selection needed) |
| `baseline.ipynb` | Polynomial-kernel logistic regression (Poly-kernel LR) with feature selection, plus the same search without feature selection and/or without binarisation |
| `logistic_RBF.ipynb` | Logistic regression and RBF (Nyström) features + logistic regression |
| `svm.ipynb` | Linear, polynomial and RBF SVMs |
| `trees.ipynb` | Random forest, Extra Trees and XGBoost |
| `mlp.ipynb` | MLP and ResMLP (three sizes each, three learning rates) with feature selection |
| `mlp_no_fs.ipynb` | The same MLP and ResMLP search on all features, without feature selection |
| `lora.ipynb` | LoRA fine-tuning of KryptoNet-40 (n = 40 and 40-adapt), rank sweep |
| `lora_with_feature_selection.ipynb` | The same LoRA rank sweep with unselected features set to 0 |
| `lora_bin.ipynb` | The LoRA rank sweep on binarised inputs, without feature selection |
| `lora_with_fs_bin.ipynb` | The LoRA rank sweep on binarised inputs, with unselected features set to 0 |
| `comparision.ipynb` | Reloads every saved model, compares test/validation accuracy, writes the hidden-label predictions |
| `MockPaperExperiments.ipynb` | Experiments from the paper |
| `kryptonet.py` | Defines and loads KryptoNet-40 |
| `kryptonet40.pt` | Pretrained KryptoNet-40 weights |
| `check_submission.py` | Checks the format of the hidden-label files |
| `requirements.txt` | Python dependencies |

## Weights layout

| Folder | Model | File in each `n<n>/<pct>pct/` |
|---|---|---|
| `baseline_fs/` | Poly-kernel LR | `model.npz` (dual weights, support set, features, degree, C, binarise flag) |
| `baseline_fs_no_bin/`, `baseline_no_fs/`, `baseline_no_fs_no_bin/` | Poly-kernel LR without binarisation, without feature selection, without both | `model.npz` (same keys) |
| `logistic_regression/`, `rbf_logistic_regression/` | Logistic regression, RBF features + LR | `model.joblib` |
| `linear_svm/`, `poly_svm/`, `rbf_svm/` | SVMs | `model.joblib` |
| `random_forest/`, `extra_trees/`, `xgboost/` | Tree models | `model.joblib` |
| `mlp/`, `resmlp/` | MLP, ResMLP with feature selection | `model.pt` |
| `mlp_no_fs/mlp/`, `mlp_no_fs/resmlp/` | MLP, ResMLP without feature selection | `model.pt` |

LoRA exists for n = 40 only, so its folders have no `n<n>/` level: each has `<pct>pct/` and `adapt/` (40-adapt)
subfolders with `best.pt` (merged KryptoNet-40 weights), plus `features.npy` when feature selection is used. The
top-level `best.pt` is the best model across fractions.

| Folder | Model | Feature selection | Binarise |
|---|---|---|---|
| `lora/` | LoRA KryptoNet (no FS) | no | no |
| `lora_fs/` | LoRA KryptoNet | yes | no |
| `lora_bin/` | LoRA KryptoNet (no FS, bin) | no | yes |
| `lora_fs_bin/` | LoRA KryptoNet (bin) | yes | yes |

## Hidden-label predictions
`comparision.ipynb` writes the submitted files in `hiddenlabels/`. For each n it uses the 100% model with the highest
validation accuracy (ties broken by test accuracy, then model order):

| File | Model |
|---|---|
| `y_predicted_20.npy` | MLP |
| `y_predicted_30.npy` | Poly-kernel LR |
| `y_predicted_40.npy` | LoRA KryptoNet with feature selection |
| `y_predicted_40_adapt.npy` | LoRA KryptoNet without feature selection (`weights/lora/adapt/`) |

`lora.ipynb` and `lora_with_feature_selection.ipynb` also write `y_predicted_40.npy` and
`y_predicted_40_adapt.npy`, so run `comparision.ipynb` last. `lora_bin.ipynb` and `lora_with_fs_bin.ipynb` do not
write any labels. Check the format with `python check_submission.py`.
