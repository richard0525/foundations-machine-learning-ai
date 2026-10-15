"""Refresh LaTeX tables from experiment results; compile if pdflatex is present."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def table(headers, rows, alignment):
    return "\n".join([
        r"\begin{tabular}{" + alignment + "}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", ""])


def scientific(x):
    coefficient, exponent = f"{x:.2e}".split("e")
    return rf"\ensuremath{{{coefficient}\times10^{{{int(exponent)}}}}}"


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    detail = json.loads((RESULTS / "regression_details.json").read_text())
    q4 = [[r["model"], f'{r["test_mae"]:.4f}', f'{r["normaltest_K2"]:.3f}',
           f'{r["normaltest_p"]:.4f}', f'{r["normaltest_p_holm"]:.4f}',
           f'{r["shapiro_p"]:.4f}'] for r in summary["q4"]]
    (RESULTS / "q4_table.tex").write_text(table(
        ["Model", "MAE (m)", "$K^2$", "Raw $p$", "Holm $p$", "Shapiro $p$"], q4, "lrrrrr"))
    names = ["sklearn logistic (L2)", "PyTorch logistic", "PyTorch probit"]
    cls = [[name, f'{r["correct"]}/{r["test_n"]}', f'{100*r["test_accuracy"]:.2f}\\%',
            "--" if r["epochs"] == 0 else str(r["epochs"])]
           for name, r in zip(names, summary["q5_classification"])]
    (RESULTS / "classification_table.tex").write_text(table(
        ["Model", "Correct / test", "Accuracy", "SGD epochs"], cls, "lrrr"))
    names = ["sklearn OLS", "PyTorch linear, MSE", "PyTorch linear, Student-t"]
    reg = [[name, f'{r["test_mae"]:.6f}', f'{r["test_rmse"]:.6f}', f'{r["train_mse"]:.6f}']
           for name, r in zip(names, summary["q5_regression"])]
    (RESULTS / "regression_table.tex").write_text(table(
        ["Model", "Test MAE (m)", "Test RMSE (m)", "Train MSE"], reg, "lrrr"))
    base, _, student = summary["q5_regression"]
    values = {
        "MSEExcess": scientific(detail["models"]["mse"]["excess_train_mse_over_ols"]),
        "PredictionGap": f'{detail["models"]["mse"]["max_test_prediction_difference_from_ols"]:.6f}',
        "TExcess": scientific(detail["models"]["student_t"]["excess_train_nll_over_reference"]),
        "StudentImprovement": f'{base["test_mae"]-student["test_mae"]:.6f}',
        "StudentPercent": f'{100*(base["test_mae"]-student["test_mae"])/base["test_mae"]:.2f}'
    }
    (RESULTS / "report_values.tex").write_text("\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()) + "\n")
    latex = shutil.which("pdflatex")
    if not latex:
        print("Tables refreshed. Install a LaTeX distribution to rebuild the PDF; the submitted PDF is included.")
        return
    command = [latex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=results", "solution.tex"]
    for _ in range(2):
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if run.returncode:
            print(run.stdout[-12000:])
            raise SystemExit("PDF compilation failed; inspect results/solution.log.")
    print(f"Built {RESULTS / 'solution.pdf'}")


if __name__ == "__main__":
    main()
