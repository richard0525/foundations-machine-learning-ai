"""HW3 Q3: prespecified gradient-boosting comparisons and split sensitivity.

The CSVs have no header. Column zero is the observed binary label; all five
remaining columns are inputs. Test results describe the five assigned budgets;
they are never used to tune tree depth, shrinkage, or any other hyperparameter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _matplotlib_cache = Path(__file__).resolve().parent / ".cache" / "matplotlib"
    _matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_matplotlib_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.special import expit
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


TREE_COUNTS = (1, 10, 100, 300, 1000)
ROBUSTNESS_SEEDS = tuple(range(20))
MODEL_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": None,
    "n_iter_no_change": None,
    "ccp_alpha": 0.0,
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """Stable binary NLL from logits, avoiding probability-rounding artifacts."""
    scores = np.asarray(scores).reshape(-1)
    predicted = (scores > 0).astype(int)  # sklearn picks class 0 at a tied score.
    return {
        "n": int(y.size),
        "wrong": int(np.sum(predicted != y)),
        "error": float(np.mean(predicted != y)),
        "log_loss": float(np.mean(np.logaddexp(0.0, scores) - y * scores)),
    }


def _model(count: int, random_state: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=count, random_state=random_state, **MODEL_PARAMETERS
    )


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.loadtxt(path, delimiter=",")
    if array.shape != (500, 6) or not np.isfinite(array).all():
        raise ValueError(f"Unexpected data shape or nonfinite entries: {path}")
    if not np.array_equal(np.unique(array[:, 0]), [0, 1]):
        raise ValueError(f"Expected both binary labels in first column: {path}")
    return array[:, 1:], array[:, 0].astype(int)


def _fit_path(
    x: np.ndarray, y: np.ndarray, split_seed: int, model_seed: int,
) -> tuple[GradientBoostingClassifier, np.ndarray, np.ndarray, list[dict], dict]:
    train, test = train_test_split(
        np.arange(y.size), test_size=0.3, random_state=split_seed, stratify=y
    )
    model = _model(max(TREE_COUNTS), model_seed).fit(x[train], y[train])
    if model.n_estimators_ != max(TREE_COUNTS):
        raise AssertionError("Early stopping unexpectedly changed the tree budget")
    rows, snapshots = [], {}
    for count, (train_score, test_score) in enumerate(
        zip(model.staged_decision_function(x[train]),
            model.staged_decision_function(x[test])), start=1
    ):
        train_score, test_score = train_score.ravel(), test_score.ravel()
        metrics_train, metrics_test = _metrics(y[train], train_score), _metrics(y[test], test_score)
        rows.append({
            "trees": count,
            **{f"train_{k}": v for k, v in metrics_train.items()},
            **{f"test_{k}": v for k, v in metrics_test.items()},
        })
        if count in TREE_COUNTS:
            snapshots[count] = (train_score.copy(), test_score.copy())
    return model, train, test, rows, snapshots


def _data_diagnostics(x: np.ndarray, y: np.ndarray) -> dict:
    """Descriptive sample diagnostics only; never used for model selection."""
    standardized = (x - x.mean(axis=0)) / x.std(axis=0)
    neighbors = NearestNeighbors(n_neighbors=2).fit(standardized)
    _, indices = neighbors.kneighbors(standardized)
    nearest = indices[:, 1]
    xy = np.column_stack([y, x])
    _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    conflicting = sum(
        np.unique(y[inverse == group]).size > 1
        for group in range(counts.size) if counts[group] > 1
    )
    # Feasibility of y_sign*(b+w'x)>=1 establishes strict LINEAR separability of
    # this observed sample. Infeasibility is not proof of stochastic label noise.
    signed = 2*y - 1
    design = np.column_stack([np.ones(y.size), standardized])
    linear_feasibility = linprog(
        np.zeros(design.shape[1]), A_ub=-signed[:, None]*design,
        b_ub=-np.ones(y.size), bounds=[(None, None)]*design.shape[1], method="highs"
    )
    # Equal-count bins describe the nonmonotone/monotone marginal x1 pattern.
    # These use the observed sample only for explanation, never model selection.
    feature1_bins = [
        {"n": int(ids.size), "feature1_mean": float(x[ids, 0].mean()),
         "positive_fraction": float(y[ids].mean())}
        for ids in np.array_split(np.argsort(x[:, 0], kind="stable"), 10)
    ]
    result = {
        "rows": int(y.size), "features": int(x.shape[1]),
        "class_counts": {str(c): int(np.sum(y == c)) for c in (0, 1)},
        "duplicate_feature_rows_beyond_first": int(np.sum(counts - 1)),
        "conflicting_label_feature_groups": int(conflicting),
        "pearson_label_feature_correlations": np.corrcoef(xy, rowvar=False)[0, 1:].tolist(),
        "feature1_equal_count_bin_label_rates": feature1_bins,
        "nearest_neighbor_label_agreement": float(np.mean(y == y[nearest])),
        "random_pair_agreement_from_class_frequencies": float(np.mean(y)**2+(1-np.mean(y))**2),
        "linear_separability_lp_status": int(linear_feasibility.status),
        "linear_separability_lp_message": str(linear_feasibility.message),
        "strictly_linearly_separable_observed_sample": bool(linear_feasibility.success),
    }
    if linear_feasibility.success:
        result["minimum_signed_margin_of_feasible_separator"] = float(
            np.min(signed*(design@linear_feasibility.x))
        )
    return result


def _plots(output_dir: Path, datasets: dict, curves: list[dict], robust: list[dict]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.dpi": 180,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.9), constrained_layout=True)
    for col, name in enumerate(datasets):
        rows = [r for r in curves if r["dataset"] == name]
        t = np.array([r["trees"] for r in rows])
        for row, metric in enumerate(("error", "log_loss")):
            ax = axes[row, col]
            for split, label, color in (("train", "Train", "#1f4e79"), ("test", "Test", "#c55a11")):
                values = np.array([r[f"{split}_{metric}"] for r in rows])
                ax.plot(t, values, label=label, color=color, linewidth=1.6)
                ii = np.array(TREE_COUNTS)-1
                ax.scatter(t[ii], values[ii], s=24, color=color, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(TREE_COUNTS, [str(t) for t in TREE_COUNTS])
            ax.set_xlabel("Number of trees")
            ax.set_ylabel("Classification error" if metric == "error" else "Mean binary log loss")
            ax.set_title(f"{name}: {'error' if metric == 'error' else 'log loss'}")
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
            if metric == "error": ax.set_ylim(bottom=0)
    fig.savefig(figures / "q3_learning_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.4), constrained_layout=True)
    for ax, name in zip(axes, datasets):
        for split, label, color in (("train", "Train", "#1f4e79"), ("test", "Test", "#c55a11")):
            means, sds = [], []
            for count in TREE_COUNTS:
                a = np.array([r[f"{split}_error"] for r in robust if r["dataset"] == name and r["trees"] == count])
                means.append(a.mean()); sds.append(a.std(ddof=1))
            means, sds = np.array(means), np.array(sds)
            ax.plot(TREE_COUNTS, means, "o-", color=color, label=label)
            ax.fill_between(TREE_COUNTS, np.maximum(0, means-sds), means+sds, color=color, alpha=.14)
        ax.set_xscale("log")
        ax.set_xticks(TREE_COUNTS, [str(t) for t in TREE_COUNTS])
        ax.set_xlabel("Number of trees"); ax.set_ylabel("Classification error")
        ax.set_title(name); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("20 prespecified stratified splits: mean ± SD (not a confidence interval)", fontsize=11)
    fig.savefig(figures / "q3_split_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.4), constrained_layout=True)
    for row, (name, (x, y)) in enumerate(datasets.items()):
        for col, (a, b) in enumerate(((0, 1), (3, 4))):
            ax=axes[row, col]
            for cls, color, marker in ((0, "#1f4e79", "o"), (1, "#c55a11", "^")):
                ix=y==cls
                ax.scatter(x[ix,a],x[ix,b],s=15,c=color,marker=marker,alpha=.6,label=f"y={cls}",linewidths=0)
            ax.set_xlabel(f"Feature {a+1}"); ax.set_ylabel(f"Feature {b+1}")
            ax.set_title(name); ax.legend(frameon=False,markerscale=1.5)
    fig.suptitle("Supplied samples: two feature projections (not a proof of separability)",fontsize=11)
    fig.savefig(figures / "q3_data_projections.png")
    plt.close(fig)


def run(
    data_dir: str | Path, output_dir: str | Path, *, seed: int = 2026,
    robustness_seeds: tuple[int, ...] = ROBUSTNESS_SEEDS,
) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    all_curves, all_primary, predictions, splits, all_robust = [], [], [], [], []
    datasets, summaries = {}, {}
    with threadpool_limits(limits=1):
        for name in ("dset0", "dset1"):
            path = data_dir / f"{name}-1.csv"
            x, y = _load(path)
            datasets[name] = (x, y)
            model, train, test, rows, snapshots = _fit_path(x, y, seed, seed)
            for r in rows:
                r = {"dataset": name, **r}
                all_curves.append(r)
                if r["trees"] in TREE_COUNTS: all_primary.append(r)
            for split, ids in (("train", train), ("test", test)):
                for i in ids:
                    splits.append({"dataset":name,"csv_row_1based":int(i+1),"split":split,"y":int(y[i])})
            equality_checks = {}
            for count in TREE_COUNTS:
                independent = model if count == max(TREE_COUNTS) else _model(count, seed).fit(x[train], y[train])
                max_differences = {}
                for position, (split, ids) in enumerate((("train", train), ("test", test))):
                    score = independent.decision_function(x[ids]).ravel()
                    difference = float(np.max(np.abs(score-snapshots[count][position])))
                    if not np.allclose(score, snapshots[count][position], rtol=0, atol=1e-12):
                        raise AssertionError(f"Independent fit differs from prefix: {name}/{count}/{split}")
                    max_differences[split] = difference
                    predicted = independent.predict(x[ids]).astype(int)
                    if not np.array_equal(predicted, (score>0).astype(int)):
                        raise AssertionError("Classification tie convention mismatch")
                    for i, z, p, pred in zip(ids, score, expit(score), predicted):
                        predictions.append({"dataset":name,"trees":count,"split":split,
                            "csv_row_1based":int(i+1),"y":int(y[i]),"score":float(z),
                            "probability_1":float(p),"prediction":int(pred)})
                equality_checks[str(count)] = max_differences
            prior = float(y[train].mean())
            majority = int(prior > .5)
            summaries[name] = {
                "source_file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "diagnostics":_data_diagnostics(x,y),
                "train_rows":len(train),"test_rows":len(test),
                "train_positive":int(y[train].sum()),"test_positive":int(y[test].sum()),
                "baseline_train_positive_fraction":prior,
                "baseline_test_classification_error":float(np.mean(y[test] != majority)),
                "independent_fit_max_abs_score_difference":equality_checks,
                "feature_importances_at_1000":model.feature_importances_.tolist(),
            }
            print(f"Q3 {name}: primary five budgets and independent-fit checks complete", flush=True)
            for split_seed in robustness_seeds:
                _, _, _, sensitivity_rows, _ = _fit_path(x,y,split_seed,seed)
                for r in sensitivity_rows:
                    if r["trees"] in TREE_COUNTS:
                        all_robust.append({"dataset":name,"split_seed":split_seed,**r})
                if (split_seed+1) % 5 == 0:
                    print(f"Q3 {name}: {split_seed+1}/{len(robustness_seeds)} sensitivity splits",flush=True)
    aggregate=[]
    for name in datasets:
        for count in TREE_COUNTS:
            records=[r for r in all_robust if r["dataset"]==name and r["trees"]==count]
            if not records: continue
            row={"dataset":name,"trees":count,"splits":len(records)}
            for metric in ("train_error","test_error","train_log_loss","test_log_loss"):
                a=np.array([r[metric] for r in records])
                row[f"{metric}_mean"]=float(a.mean())
                row[f"{metric}_sd"]=float(a.std(ddof=1)) if a.size>1 else 0.0
            aggregate.append(row)
    _write_csv(output_dir/"q3_primary_metrics.csv",all_primary)
    _write_csv(output_dir/"q3_learning_curves.csv",all_curves)
    _write_csv(output_dir/"q3_primary_predictions.csv",predictions)
    _write_csv(output_dir/"q3_split_assignments.csv",sorted(splits,key=lambda r:(r["dataset"],r["csv_row_1based"])))
    if all_robust:
        _write_csv(output_dir/"q3_repeated_split_metrics.csv",all_robust)
        _write_csv(output_dir/"q3_repeated_split_summary.csv",aggregate)
        _plots(output_dir,datasets,all_curves,all_robust)
    result={
        "primary_split_seed":seed,"model_seed":seed,"test_fraction":.3,
        "split_method":"stratified random 70/30 train/test",
        "primary_counts":list(TREE_COUNTS),"model_parameters":MODEL_PARAMETERS,
        "robustness_split_seeds":list(robustness_seeds),
        "robustness_interpretation":"Prespecified splits; SD describes split sensitivity, not a confidence interval; observations overlap across splits.",
        "selection_policy":"All hyperparameters and tree budgets fixed before testing; no test-guided model selection.",
        "datasets":summaries,"primary_metrics":all_primary,
        "repeated_split_summary":aggregate,
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit-learn":sklearn.__version__,"matplotlib":matplotlib.__version__},
        "elapsed_seconds":time.perf_counter()-start,
    }
    (output_dir/"q3_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Q3 complete",flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=Path(__file__).parent/"data")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"results")
    parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,seed=args.seed)
