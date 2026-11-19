# Homework 3 — Ruochen Wang

Poisson boosting theory, finite-candidate risk bounds, white-wine model selection,
and gradient-boosting experiments on the two supplied binary datasets.

- [Written solutions](results/HW3_Solutions.pdf)
- [Original assignment](data/assignment_hw3.pdf)
- [Course repository](https://github.com/richard0525/foundations-machine-learning-ai)

## Run from a fresh clone

Python 3.12 is recommended (tested: 3.12.14). A CPU is sufficient. All data are
included, so running experiments needs no account, API key, or data download.
Internet access is needed only to clone and install dependencies.

```bash
git clone https://github.com/richard0525/foundations-machine-learning-ai.git
cd foundations-machine-learning-ai
python3.12 -m venv .venv-hw3
.venv-hw3/bin/python -m pip install -r homework_3/requirements-lock.txt
.venv-hw3/bin/python homework_3/run.py --jobs 4
.venv-hw3/bin/python homework_3/verify.py
```

On Windows use `py -3.12 -m venv .venv-hw3` and replace
`.venv-hw3/bin/python` with `.venv-hw3\Scripts\python.exe`.
A separate environment avoids changing earlier homeworks' dependencies.
The full search and repeated boosting experiments take several minutes on a
laptop. `--jobs 1` reduces parallelism without changing the grids or seeds.

The default command runs **all 1,040 wine configurations**, the five required
boosting budgets for each dataset, and 20 prespecified split-sensitivity runs per
dataset. It verifies input checksums before fitting. To run just one problem:

```bash
.venv-hw3/bin/python homework_3/run.py --part q2
.venv-hw3/bin/python homework_3/run.py --part q3
```

To preserve committed results when reproducing them, add
`--output-dir /path/to/new-results`, and pass the same directory to
`verify.py --results-dir /path/to/new-results`.

## Methods

**Q2.** Official UCI white-wine data: 4,898 rows and 11 predictors. Reserve 20%
using seed 2026; reserve 1/8 of the remaining rows for validation using seed 2027,
giving 3,428/490/980 train/validation/test rows. Random forest searches 540
configurations; KNN searches 500, each more than the 490 validation rows. KNN
standardization is fitted only on training inputs. Candidates and selected
models are fitted on training data only. Selection uses minimum validation MSE,
with exact ties resolved by fixed `ParameterGrid` order. Both winners are saved
before test predictions. The test set never changes grids, seeds, or selections.

Exact repeated wine records are retained for the required random-row split;
the report and `q2_summary.json` quantify cross-split repeats. These scores do
not establish performance on entirely distinct wines. KNN's zero resubstitution
training error includes each training example as its own zero-distance neighbor;
it is not leave-one-out performance.

**Q3.** Both supplied CSVs are headerless. Column 0 is the binary target; columns
1–5 are inputs. Primary splits: stratified 350/150 train/test, seed 2026.
`GradientBoostingClassifier`: log loss, learning rate 0.1, maximum depth 3,
subsample 1, model seed 2026, no early stopping. Budgets are exactly
1, 10, 100, 300, and 1,000 trees. Independent budget fits are checked against
prefixes of the 1,000-tree path. Classification error and stable logit-based
binary NLL are recorded. Sensitivity uses split seeds 0–19 with the same model
seed; its SD is descriptive, not an independent-sample confidence interval.
All budgets are reported, with no test-guided model selection.

## Output files

- `q2_wine.py`, `q3_boosting.py`: complete experimental implementations.
- `run.py`: checksum validation, experiment entry point, environment metadata.
- `verify.py`: independent Poisson derivative checks; source labels, split
  integrity, full grids and selections, preprocessing, prediction-derived metrics,
  independent-fit agreement, and repeated-split aggregates.
- `results/q2_grid_scores.csv`: all 1,040 candidate scores and parameter sets.
- `results/q2_selection.json`, `q2_summary.json`: winners and all metrics.
- `results/q2_predictions.csv`, `q2_splits.csv`: predictions and zero-based data
  row IDs (excluding the CSV header).
- `results/q3_primary_metrics.csv`, `q3_primary_predictions.csv`: all required
  budgets and predictions; `csv_row_1based` refers to the headerless input row.
- `results/q3_learning_curves.csv`: every stage from 1 through 1,000.
- `results/q3_repeated_split_*.csv`, `q3_summary.json`: sensitivity and diagnostics.
- `results/figures/`: PNG learning curves, comparison plots, and data views.
- `results/checks_summary.json`: verification outcome.
- `data_manifest.json`: SHA-256 checksums and provenance for every input.

## Rebuild the PDF

The submitted PDF is included. To rebuild, install a TeX distribution providing
`pdflatex`, `lmodern`, and the standard packages listed in `solution.tex`, then run:

```bash
.venv-hw3/bin/python homework_3/build_report.py
```

This regenerates numerical table macros from saved JSON and compiles LaTeX twice.
Use `--values-only` to regenerate the table source without TeX. LaTeX is not
required for any experiment or numerical verification.

## Data and software sources

The two `dset*-1.csv` files and assignment PDF were provided with HW3, unchanged.
The white-wine CSV and its documentation were downloaded from the
[official UCI archive](https://archive.ics.uci.edu/static/public/186/wine+quality.zip).
P. Cortez et al. (2009), [Wine Quality](https://doi.org/10.24432/C56S3T), UCI.
The white subset, not the red subset, is used. No credentials are required.

Hyperparameters follow the scikit-learn 1.7.2 documentation:
[RF](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[KNN](https://scikit-learn.org/1.7/modules/generated/sklearn.neighbors.KNeighborsRegressor.html),
and [gradient boosting](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
