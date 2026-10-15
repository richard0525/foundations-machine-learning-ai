# Foundations of Machine Learning and AI

Reproducible course homework code. Each assignment lives in its own directory.

- [homework_1/](homework_1/): theoretical solutions, ALE residual diagnostics, and custom PyTorch models.
- Future assignments will use `homework_2/`, `homework_3/`, and so on when completed.

Course repository: https://github.com/richard0525/foundations-machine-learning-ai

Written submission: [Homework 1 solution PDF](homework_1/results/solution.pdf).
The PDF contains clickable repository links and the commands below.

## Run Homework 1

Use Python 3.12 (the tested interpreter version is recorded in the results metadata).
From a fresh clone of this repository:

```bash
git clone https://github.com/richard0525/foundations-machine-learning-ai.git
cd foundations-machine-learning-ai
python3.12 -m venv .venv
.venv/bin/python -m pip install -r homework_1/requirements.txt
.venv/bin/python homework_1/run.py
.venv/bin/python -m unittest discover -s homework_1 -p 'test_*.py' -v
.venv/bin/python homework_1/build_report.py
```

On Windows, use `py -3.12 -m venv .venv` and replace `.venv/bin/python` with
`.venv\Scripts\python.exe`. Git and Python 3.12 are prerequisites; no GPU is needed.
Dependency installation requires internet access. The full dependency snapshot
is in `homework_1/requirements-lock.txt`.

`run.py` produces every
empirical result and figure using the included, checksum-verified public ALE data;
if that CSV is absent, it is downloaded automatically. The breast-cancer data ship
with scikit-learn. No credentials or machine-specific data paths are needed.

`build_report.py` updates the report's tables from the current results. If `pdflatex`
is installed, it also rebuilds the PDF. Without LaTeX, the numerical results and
figures are still fully reproducible, and the submitted PDF is already included.
See `homework_1/README.md` for methods and output details.

The test suite contains nine tests, including independent loss/gradient checks
and a regression check for cancellation in correctly classified extreme logits.
