# Foundations of Machine Learning and AI

Reproducible course homework code. Each assignment lives in its own directory.

- `homework_1/`: theoretical solutions, ALE residual diagnostics, and custom PyTorch models.

## Run Homework 1

Use Python 3.12 (the tested interpreter version is recorded in the results metadata).
From a fresh clone of this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r homework_1/requirements.txt
python homework_1/run.py
python -m unittest discover -s homework_1 -p 'test_*.py' -v
python homework_1/build_report.py
```

The first three commands install the numerical environment. `run.py` produces every
empirical result and figure using the included, checksum-verified public ALE data;
if that CSV is absent, it is downloaded automatically. The breast-cancer data ship
with scikit-learn. No credentials or machine-specific data paths are needed.

`build_report.py` updates the report's tables from the current results. If `pdflatex`
is installed, it also rebuilds the PDF. Without LaTeX, the numerical results and
figures are still fully reproducible, and the submitted PDF is already included.
See `homework_1/README.md` for methods and output details.
