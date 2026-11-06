# Homework 2 — Ruochen Wang

Complete solutions to the corrected HW2 assignment: heteroscedastic taxi-trip
regression, categorical encodings for retail counts, and RKS versus a one-hidden-layer MLP.

## Run from a fresh clone

Git and Python 3.12 are prerequisites. No GPU, account, API key, or external data
download is needed. The five supplied datasets are included unchanged and
verified by SHA-256. Install dependencies with internet access:

```bash
git clone https://github.com/richard0525/foundations-machine-learning-ai.git
cd foundations-machine-learning-ai
python3.12 -m venv .venv
.venv/bin/python -m pip install -r homework_2/requirements-lock.txt
.venv/bin/python homework_2/run.py
.venv/bin/python -m unittest discover -s homework_2 -v
.venv/bin/python homework_2/build_report.py
```

On Windows, use `py -3.12 -m venv .venv` and replace `.venv/bin/python` with
`.venv\Scripts\python.exe`. `requirements.txt` lists the seven direct numerical
dependencies; `requirements-lock.txt` also pins their transitive dependencies.
Runs use CPU, one numerical thread, float32 neural networks, float64 ridge and
evaluation, and seed 2026. Expect several minutes, depending on hardware.
Small numerical differences across platforms may occur.

`build_report.py` refreshes tables from the numerical results. It rebuilds the
PDF if `pdflatex` and the required standard LaTeX packages are available; the
finished PDF is already included, so LaTeX is optional for code reproduction.

For one problem or a separate results directory:

```bash
.venv/bin/python homework_2/run.py --part q3 --output-dir /path/to/results
```

## Submission files

- [Written solutions](results/solution.pdf), with clickable repository links.
- [RKS test predictions](results/rks_test_predictions.csv).
- [MLP test predictions](results/mlp_test_predictions.csv).

Both prediction files have **200 rows, one numeric column, no header**, in exactly
the order of `data/test.csv`. The test labels are unavailable; no toy test MSE is
reported. `q3_test_audit.csv` includes coordinates and row numbers for auditing.

## Methods and leakage controls

**Q1.** Approximately 70/15/15 random split of exact-record groups, keeping
identical taxi records in the same train/validation/test partition. The provided
seven features are already standardized; the duration target is centered/scaled
using training statistics only. A 7–64–64–2 ReLU MLP returns a mean and
`softplus(raw_scale) + 0.001` standard deviation on this scaled target. Gaussian
NLL trains both heads; predictions and scores are returned to minutes. Adam,
batch size 2048, validation-NLL early stopping, and all settings are recorded in
`config.json`. Baselines are a constant Gaussian, linear Gaussian, and gradient
boosted mean with constant residual SD. A training-event-frequency baseline
also contextualizes probability scores for the rare event of trips lasting 45
minutes or longer. Evaluation includes Gaussian NLL, RMSE, MAE, interval coverage,
Brier score, binary log-loss, and threshold-0.5 accuracy. No tail calibration or
hyperparameter tuning is performed on the test set.

**Q2.** Approximately 70/15/15 random transaction split. Both identifiers are
categorical strings. Training rows get **five-fold out-of-fold smoothed means of
raw Quantity**, separately for each category; the prior mean also excludes the
held-out fold. Validation/test use mappings fitted only on the complete training
set, with the training mean for unseen categories. Encoding values are log1p
transformed and standardized using OOF training encodings. Embedding vocabularies
are training-only and reserve zero for unseen categories. Two 16-dimensional
embeddings are concatenated, optionally with the two target-encoded features,
then passed through 64 and 32 ReLU units. The target-only network uses the same
hidden widths. Each model minimizes MSE on `log1p(Quantity)` and uses the same
validation-stopping rule. Original-unit predictions are `expm1(predicted_log1p)`;
they are continuous and nonnegative, not rounded and not claimed to estimate the
arithmetic conditional mean. All transactions, including extreme quantities,
remain in the data. Test metrics report log1p-MSE, RMSLE, MAE, RMSE, and median
absolute error. Random transaction splits assess this sampling setting, not
future-time or entirely new-customer generalization.

**Q3.** The supplied 600/200/200 splits are preserved. RKS features are the exact
interleaved sine/cosine pairs in the assignment. Frequency draws have standard
deviation `1/sigma`; phases are uniform on `[0,2*pi]`. Separate random streams
produce nested draws across the feature-count sweep. Ridge minimizes
`mean((y - intercept - Phi @ beta)**2) + lambda * sum(beta**2)` with an unpenalized
intercept and no feature normalization. The sweep varies only total feature count
`D=2h`; sigma=1 and lambda=0.001 are fixed. The MLP sweep varies only hidden width;
tanh, Adam, batch size, learning rate, maximum epochs, patience, and weight decay
are fixed. Validation chooses the width and checkpoint. Both selected models are
refitted from scratch on **training rows only**; the MLP uses its chosen epoch
count, and both deterministic refits reproduce their selection validation scores.
The final predictions come from these refits. Validation is a model-selection
score, not an unbiased test estimate.

## Files and audit trail

- `run.py`, `q1_taxi.py`, `q2_retail.py`, `q3_features.py`: experiment entry points.
- `models.py`, `encoding.py`, `common.py`: implemented models and shared utilities.
- `dataloader.py`, `plotting.py`: provided course helpers (unchanged).
- `test_models.py`: independent likelihood/gradient, ridge, leakage, unknown-
  category, split, and dataset-contract checks.
- `config.json`: fixed experimental settings.
- `results/`: metrics, prediction audits, split indices, encoding maps, trained
  weights, sweep/learning curves, figures, and compiled report.
- `results/metadata_all.json`: package versions, input checksums, and experiment source hashes.

The assignment explicitly requests two prediction CSVs in Q3(d); the repository
links support retrieval but do not replace any required submission uploads.
The ZIP, model weights, data, and verification files need not be uploaded
separately under the course GitHub code guidelines.

The original assignment PDF, screenshot, and incomplete starter model skeleton
are retained in the user's local input folder, not republished as submitted code.
