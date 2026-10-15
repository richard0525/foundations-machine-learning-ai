# Homework 1

## Entry points

- `run.py`: Problems 4 and 5, including all data preparation, SGD fits, diagnostics,
  metrics, learning curves, split indices, and provenance.
- `models.py`: affine predictor, sigmoid/probit probability maps, explicit custom
  BCE/MSE/Student-t losses, and mini-batch SGD.
- `data_utils.py`: public ALE download, SHA-256 validation, and column selection.
- `test_models.py`: independent loss/gradient checks and recovery of a known linear model.
- `solution.tex`: full written solutions to Problems 1-5.
- `build_report.py`: refresh report tables from actual results and optionally compile.
- `config.json`: fixed experiment settings.

Run instructions are in the repository-root README. The computations use CPU
float64, one PyTorch thread, deterministic algorithms, and seed 42. Runtime is
normally a few minutes; it depends on the computer. The tested package versions
are pinned in `requirements.txt`; `requirements-lock.txt` records the full environment.

## Method choices

ALE uses its first four columns to predict `ale`; `sd_ale` is never an input. A fixed
random 80/20 split gives 85 training and 22 test rows, shared by Problems 4 and 5.
Problem 4 uses OLS, RBF SVR, a depth-limited decision tree, and a random forest.
Normality diagnostics use held-out residuals (`observed - predicted`), with
D'Agostino-Pearson, Shapiro-Wilk, Q-Q plots, and Holm-adjusted omnibus p-values.
Only the four omnibus tests form the Holm family; Shapiro-Wilk is a sensitivity check.

Breast-cancer classification uses an 80/20 stratified split (455 train, 114 test),
with labels 0=malignant and 1=benign. A further training-only stratified split is
used to select an epoch by validation BCE, independently for logistic and probit
models. Each selected model is then refit from scratch on all 455 training rows.
Scaling is learned separately on the corresponding training subset, never the test
set. Class predictions threshold the probability of label 1 at 0.5.

The sklearn logistic baseline retains its default L2 regularization (`C=1`); the
custom losses are the unpenalized BCE requested in the assignment. Their differing
regularization and early stopping are explicitly acknowledged in the report.
Log-domain formulas avoid sigmoid/CDF clipping without changing the mathematical
BCE. All four custom models use `torch.optim.SGD` with zero momentum and no weight
decay. The rate in epoch e (starting at zero) is eta0/(1+e/decay_epochs).

Regression standardizes X only. Targets and the Student-t residuals remain in
original ALE units, so the assigned df=5, scale=1 likelihood is preserved. MSE SGD
is checked against sklearn OLS; Student-t SGD is checked against an independent
training-only BFGS reference, which is used for verification and never supplies
the submitted predictions. The test set is not used to select any settings.

## Outputs

`results/solution.pdf` is the written submission. CSV files contain metrics,
held-out predictions/residuals, and optimization histories. JSON files record
model coefficients, scaling, split indices, configuration, package versions, data
checksums, and source hashes. `results/figures/` contains PDF and PNG plots.

To rerun into a separate directory:

```bash
python homework_1/run.py --output-dir /path/to/new-results
```

The report builder reads the default `homework_1/results/` directory. The source
PDF and lecture decks are not redistributed in this repository.

## Data attribution

ALE: Singh, A., Kotiyal, V., Sharma, S., Nagar, J., and Lee, C. (2020).
*Average Localization Error (ALE) in sensor node localization process in WSNs*.
UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5ZP6R.
License: CC BY 4.0. The included `data/ale.csv` is the original unmodified CSV,
renamed from `mcs_ds_edited_iter_shuffled.csv`; the program verifies its checksum.
Source: https://archive.ics.uci.edu/dataset/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns

Breast Cancer Wisconsin (Diagnostic): Wolberg, W., Mangasarian, O., Street, N.,
and Street, W. (1993). UCI Machine Learning Repository.
https://doi.org/10.24432/C5DW2B. Loaded through
`sklearn.datasets.load_breast_cancer`; no separate download is required.
