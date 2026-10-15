"""Run every numerical experiment for Homework 1, with a held-out test split."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import torch

from data_utils import load_ale, FEATURES, ALE_SHA256, ALE_URL
from models import LOSSES, fit_sgd, predict, original_coefficients


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def holm_adjust(pvalues):
    """Holm correction controlling family-wise error for these four tests."""
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(pvalues)-np.arange(len(pvalues))) * pvalues[order]))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def save_figure(fig, out, name):
    fig.savefig(out / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def problem4(x, y, train, test, config, out):
    seed = config["seed"]
    models = {
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "RBF SVR": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, gamma="scale")),
        "Decision tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=seed),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=seed, n_jobs=1)
    }
    rows, predictions = [], pd.DataFrame({"row_id": test, "ale": y[test]})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    resfig, raxes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for (name, model), ax, rax in zip(models.items(), axes.flat, raxes.flat):
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        residual = y[test] - pred
        k2, p = stats.normaltest(residual)
        sw, sp = stats.shapiro(residual)
        rows.append({"model": name, "test_n": len(test), "test_mae": mean_absolute_error(y[test], pred),
                     "test_rmse": mean_squared_error(y[test], pred) ** 0.5,
                     "normaltest_K2": k2, "normaltest_p": p,
                     "shapiro_W": sw, "shapiro_p": sp,
                     "residual_mean": residual.mean(), "residual_sd": residual.std(ddof=1),
                     "skewness": stats.skew(residual, bias=False),
                     "excess_kurtosis": stats.kurtosis(residual, bias=False)})
        predictions[f"{name}_prediction"] = pred
        predictions[f"{name}_residual"] = residual
        stats.probplot(residual, dist="norm", plot=ax)
        ax.set_title(f"{name}: K² p={p:.3g}")
        ax.set_xlabel("Theoretical normal quantile")
        ax.set_ylabel("Held-out residual (m)")
        ax.grid(alpha=.2)
        rax.scatter(pred, residual, s=26, color="#196d8a", alpha=.8)
        rax.axhline(0, color="#a75631", lw=1)
        rax.set(title=name, xlabel="Predicted ALE (m)", ylabel="Held-out residual (m)")
        rax.grid(alpha=.2)
    frame = pd.DataFrame(rows)
    frame["normaltest_p_holm"] = holm_adjust(frame.normaltest_p)
    frame["reject_raw_0_05"] = frame.normaltest_p < config["normality_alpha"]
    frame["reject_holm_0_05"] = frame.normaltest_p_holm < config["normality_alpha"]
    frame.to_csv(out / "q4_normality.csv", index=False)
    predictions.to_csv(out / "q4_predictions.csv", index=False)
    save_figure(fig, out, "q4_qqplots")
    save_figure(resfig, out, "q4_residuals")
    return frame.to_dict(orient="records")


def problem5_classification(config, out):
    data = load_breast_cancer()
    x, y = data.data, data.target
    seed, c = config["seed"], config["classification"]
    train, test = train_test_split(np.arange(len(y)), test_size=config["test_fraction"],
                                  random_state=seed, stratify=y)
    inner_train, validation = train_test_split(train, test_size=c["validation_fraction"],
                                             random_state=seed + 1, stratify=y[train])
    assert set(train).isdisjoint(test) and set(inner_train).isdisjoint(validation)
    scaler = StandardScaler().fit(x[train])
    xt, xe = scaler.transform(x[train]), scaler.transform(x[test])
    inner_scaler = StandardScaler().fit(x[inner_train])
    xi, xv = inner_scaler.transform(x[inner_train]), inner_scaler.transform(x[validation])
    baseline = LogisticRegression(C=1.0, solver="lbfgs", max_iter=10000, tol=1e-10)
    baseline.fit(xt, y[train])
    pred = baseline.predict(xe)
    rows = [{"model": "sklearn logistic (L2, C=1)", "test_accuracy": accuracy_score(y[test], pred),
             "correct": int(np.sum(pred == y[test])), "test_n": len(test), "epochs": 0}]
    preds = pd.DataFrame({"row_id": test, "target": y[test],
                          "baseline_probability": baseline.predict_proba(xe)[:, 1],
                          "baseline_prediction": pred})
    details = {"target_labels": {"0": "malignant", "1": "benign"},
               "feature_names": data.feature_names.tolist(),
               "train_indices": train.tolist(), "test_indices": test.tolist(),
               "inner_train_indices": inner_train.tolist(), "validation_indices": validation.tolist(),
               "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
               "baseline_confusion_matrix": confusion_matrix(y[test], pred).tolist(),
               "baseline_iterations": int(baseline.n_iter_[0]), "models": {}}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for kind, ax in zip(("logistic", "probit"), axes):
        print(f"Selecting {kind} epoch on training-only validation data...", flush=True)
        kwargs = {k: c[k] for k in ("learning_rate", "decay_epochs", "batch_size", "checkpoint_every")}
        _, hist, epoch, _ = fit_sgd(xi, y[inner_train], kind, epochs=c["max_epochs"],
                                   validation=(xv, y[validation]), seed=seed, **kwargs)
        history = pd.DataFrame(hist)
        history.to_csv(out / f"q5_{kind}_validation_history.csv", index=False)
        print(f"Refitting {kind} on all {len(train)} training rows for {epoch} epochs...", flush=True)
        model, train_hist, _, grad = fit_sgd(xt, y[train], kind, epochs=epoch, seed=seed, **kwargs)
        pd.DataFrame(train_hist).to_csv(out / f"q5_{kind}_training_history.csv", index=False)
        prob = predict(model, xe, kind)
        pred = (prob >= .5).astype(int)
        rows.append({"model": f"PyTorch {kind} (BCE, SGD)", "test_accuracy": accuracy_score(y[test], pred),
                     "correct": int(np.sum(pred == y[test])), "test_n": len(test), "epochs": epoch})
        preds[f"{kind}_probability"] = prob
        preds[f"{kind}_prediction"] = pred
        details["models"][kind] = {**original_coefficients(model, scaler), "selected_epoch": epoch,
                                    "gradient_norm": grad, "confusion_matrix": confusion_matrix(y[test], pred).tolist(),
                                    "final_train_loss": train_hist[-1]["train_loss"]}
        ax.plot(history.epoch, history.train_loss, label="Inner training")
        ax.plot(history.epoch, history.validation_loss, label="Validation")
        ax.axvline(epoch, color="#a75631", ls="--", label=f"Selected: {epoch}")
        ax.set(title=kind.capitalize(), xlabel="Epoch", ylabel="Mean binary cross-entropy")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save_figure(fig, out, "q5_classification_learning")
    pd.DataFrame(rows).to_csv(out / "q5_classification.csv", index=False)
    preds.to_csv(out / "q5_classification_predictions.csv", index=False)
    dump(out / "classification_details.json", details)
    return rows


def problem5_regression(x, y, train, test, config, out):
    c, seed = config["regression"], config["seed"]
    scaler = StandardScaler().fit(x[train])
    xt, xe = scaler.transform(x[train]), scaler.transform(x[test])
    ols = LinearRegression().fit(xt, y[train])
    base_pred = ols.predict(xe)
    base_mse = mean_squared_error(y[train], ols.predict(xt))
    rows = [{"model": "sklearn OLS", "test_mae": mean_absolute_error(y[test], base_pred),
             "test_rmse": mean_squared_error(y[test], base_pred)**.5, "train_mse": base_mse}]
    preds = pd.DataFrame({"row_id": test, "ale": y[test], "baseline_prediction": base_pred})
    details = {"feature_names": FEATURES, "train_indices": train.tolist(), "test_indices": test.tolist(),
               "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
               "target_transformation": "none; original ALE meters", "models": {}}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for kind, ax in zip(("mse", "student_t"), axes):
        print(f"Fitting ALE {kind} with SGD...", flush=True)
        model, hist, _, grad = fit_sgd(xt, y[train], kind, seed=seed,
            **{k: c[k] for k in ("epochs", "learning_rate", "decay_epochs", "batch_size", "checkpoint_every")})
        pred = predict(model, xe, kind)
        train_pred = predict(model, xt, kind)
        rows.append({"model": f"PyTorch linear ({kind}, SGD)", "test_mae": mean_absolute_error(y[test], pred),
                     "test_rmse": mean_squared_error(y[test], pred)**.5,
                     "train_mse": mean_squared_error(y[train], train_pred)})
        preds[f"{kind}_prediction"] = pred
        info = {**original_coefficients(model, scaler), "gradient_norm": grad,
                "final_train_loss": hist[-1]["train_loss"]}
        if kind == "mse":
            info["excess_train_mse_over_ols"] = mean_squared_error(y[train], train_pred) - base_mse
            info["max_test_prediction_difference_from_ols"] = float(np.max(np.abs(pred-base_pred)))
            assert info["excess_train_mse_over_ols"] < 1e-4, "SGD MSE is insufficiently close to OLS."
        else:
            # An independent reference optimizer checks training convergence only.
            # Submitted model parameters/predictions remain those trained by SGD.
            design = np.column_stack([xt, np.ones(len(xt))])
            def objective(beta):
                residual = y[train] - design @ beta
                value = -stats.t.logpdf(residual, df=5).mean()
                gradient = design.T @ (-6 * residual / (5 + residual**2)) / len(residual)
                return value, gradient
            reference = optimize.minimize(objective, np.r_[ols.coef_, ols.intercept_],
                                          method="BFGS", jac=True, options={"gtol": 1e-9})
            info["reference_train_nll"] = float(reference.fun)
            info["reference_gradient_norm"] = float(np.linalg.norm(reference.jac))
            info["excess_train_nll_over_reference"] = float(hist[-1]["train_loss"] - reference.fun)
            assert info["excess_train_nll_over_reference"] < 1e-4
        details["models"][kind] = info
        history = pd.DataFrame(hist)
        history.to_csv(out / f"q5_{kind}_training_history.csv", index=False)
        ax.plot(history.epoch, history.train_loss, color="#196d8a")
        ax.set(title="Squared error" if kind == "mse" else "Student-t, df=5, scale=1",
               xlabel="Epoch", ylabel="Mean training loss")
        ax.grid(alpha=.2)
    save_figure(fig, out, "q5_regression_learning")
    pd.DataFrame(rows).to_csv(out / "q5_regression.csv", index=False)
    preds.to_csv(out / "q5_regression_predictions.csv", index=False)
    dump(out / "regression_details.json", details)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text())
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x, y, frame = load_ale(ROOT / "data")
    train, test = train_test_split(np.arange(len(y)), test_size=config["test_fraction"],
                                  random_state=config["seed"])
    assert set(train).isdisjoint(test)
    print(f"ALE: {len(train)} train / {len(test)} test; only the first four features.", flush=True)
    results = {"q4": problem4(x, y, train, test, config, out),
               "q5_classification": problem5_classification(config, out),
               "q5_regression": problem5_regression(x, y, train, test, config, out)}
    dump(out / "summary.json", results)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [ROOT / "run.py", ROOT / "models.py", ROOT / "data_utils.py", ROOT / "config.json"]}
    dump(out / "metadata.json", {"config": config, "python": platform.python_version(),
        "platform": platform.platform(), "device": "CPU", "dtype": "float64", "threads": 1,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib"]},
        "source_sha256": source_hashes, "ale_url": ALE_URL, "ale_sha256": ALE_SHA256,
        "ale_rows": len(y), "ale_features": FEATURES, "ale_train_indices": train.tolist(),
        "ale_test_indices": test.tolist(), "ale_duplicate_feature_rows": int(frame.duplicated(FEATURES).sum()),
        "ale_duplicate_complete_rows": int(frame.duplicated().sum()),
        "baseline_regularization": "L2, C=1; custom classification losses are unpenalized",
        "model_selection": "Classification epochs chosen by training-only validation BCE, then refit on full training set. All other settings fixed before test evaluation."})
    print(json.dumps(results, indent=2), flush=True)
    print(f"Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
