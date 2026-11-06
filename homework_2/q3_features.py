"""Q3: one-dimensional hyperparameter sweeps, refits and unlabeled predictions."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common import setup, dump, savefig, tensor, fit_network, plt
from dataloader import HW2Dataset
from models import RandomKitchenSinks, MLP


def mse(y, predicted):
    return float(np.mean((np.asarray(y, float)-np.asarray(predicted, float).ravel())**2))


def run(data, out, config):
    start = time.monotonic()
    seed, c = config["seed"], config["toy"]
    setup(seed, config["threads"])
    ds = HW2Dataset.from_csv(data/"train.csv", data/"val.csv", data/"test.csv")
    assert ds.test.y is None and len(ds.test.X) == 200
    fig, ax = plt.subplots(figsize=(6.2, 5))
    points = ax.scatter(*ds.train.X.T, c=ds.train.y, s=18, cmap="viridis")
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="$x_1$", ylabel="$x_2$", title="Training data (600 observations)")
    fig.colorbar(points, ax=ax, label="$y$")
    savefig(fig, out, "q3_training_data")

    rks_rows = []
    for D in c["rks_D"]:
        model = RandomKitchenSinks(D, c["rks_sigma"], c["rks_lambda"], seed).fit(ds.train.X, ds.train.y)
        row = {"D": D, "h": D//2, "train_mse": mse(ds.train.y, model.predict(ds.train.X)),
               "validation_mse": mse(ds.val.y, model.predict(ds.val.X))}
        rks_rows.append(row)
        print("RKS", row, flush=True)
    pd.DataFrame(rks_rows).to_csv(out/"q3_rks_sweep.csv", index=False)
    best_rks = min(rks_rows, key=lambda row: row["validation_mse"])

    x, y, xv, yv = tensor(ds.train.X), tensor(ds.train.y), tensor(ds.val.X), tensor(ds.val.y)
    mlp_rows = []
    histories = {}
    for width in c["mlp_widths"]:
        setup(seed, config["threads"])
        model = MLP(2, width, c["activation"])
        print(f"Toy MLP width {width}", flush=True)
        history, epoch = fit_network(model, [x], y, [xv], yv, F.mse_loss,
            **{k:c[k] for k in ["epochs", "patience", "batch_size", "learning_rate", "weight_decay", "checkpoint_every"]}, seed=seed)
        histories[width] = history
        pd.DataFrame(history).to_csv(out/f"q3_mlp_width_{width}_history.csv", index=False)
        row = {"hidden_width": width, "selected_epoch": epoch, "epochs_run": history[-1]["epoch"],
               "train_mse": mse(ds.train.y, model.predict(ds.train.X)),
               "validation_mse": mse(ds.val.y, model.predict(ds.val.X))}
        mlp_rows.append(row)
        print("MLP selection", row, flush=True)
    pd.DataFrame(mlp_rows).to_csv(out/"q3_mlp_sweep.csv", index=False)
    best_mlp = min(mlp_rows, key=lambda row: row["validation_mse"])

    # Requested refits use only the supplied training rows. Validation remains
    # outside optimization; its scores are selection scores, not test estimates.
    rks = RandomKitchenSinks(best_rks["D"], c["rks_sigma"], c["rks_lambda"], seed).fit(ds.train.X, ds.train.y)
    setup(seed, config["threads"])
    mlp = MLP(2, best_mlp["hidden_width"], c["activation"])
    fit_network(mlp, [x], y, None, None, F.mse_loss,
                epochs=best_mlp["selected_epoch"], patience=0,
                **{k:c[k] for k in ["batch_size", "learning_rate", "weight_decay", "checkpoint_every"]}, seed=seed)
    models = {"RKS": rks, "MLP": mlp}
    comparison = []
    for name, model in models.items():
        row = {"model": name, "train_mse": mse(ds.train.y, model.predict(ds.train.X)),
               "validation_mse": mse(ds.val.y, model.predict(ds.val.X))}
        comparison.append(row)
        test_prediction = model.predict(ds.test.X).reshape(-1)
        assert test_prediction.shape == (200,) and np.isfinite(test_prediction).all()
        np.savetxt(out/f"{name.lower()}_test_predictions.csv", test_prediction, delimiter=",", fmt="%.17g")
    np.testing.assert_allclose(comparison[0]["validation_mse"], best_rks["validation_mse"], rtol=1e-10)
    np.testing.assert_allclose(comparison[1]["validation_mse"], best_mlp["validation_mse"], rtol=1e-10)
    rks.save(out/"q3_rks_model.npz")
    torch.save(mlp.state_dict(), out/"q3_mlp_weights.pt")
    pd.DataFrame(comparison).to_csv(out/"q3_comparison.csv", index=False)
    for name, X, target in [("train", ds.train.X, ds.train.y), ("val", ds.val.X, ds.val.y), ("test", ds.test.X, None)]:
        frame = pd.DataFrame({"row_id": np.arange(len(X)), "x1": X[:,0], "x2": X[:,1]})
        if target is not None:
            frame["target"] = target
        for model_name, model in models.items():
            frame[model_name.lower()+"_prediction"] = model.predict(X).ravel()
        frame.to_csv(out/f"q3_{name}_audit.csv", index=False)

    # Separate train and validation plots satisfy Q3(b)'s two-plot requirement.
    for name, rows, key in [("rks", rks_rows, "D"), ("mlp", mlp_rows, "hidden_width")]:
        for metric, title in [("train_mse", "Training"), ("validation_mse", "Validation")]:
            fig, ax = plt.subplots(figsize=(5.8, 3.5))
            ax.plot([r[key] for r in rows], [r[metric] for r in rows], "o-", color="#176b89")
            ax.set(xscale="log", yscale="log", xlabel="Total random features $D=2h$" if name=="rks" else "Hidden-layer width",
                   ylabel="MSE", title=f"{name.upper()} {title.lower()} MSE")
            ax.set_xticks([r[key] for r in rows], [str(r[key]) for r in rows])
            ax.grid(alpha=.2)
            savefig(fig, out, f"q3_{name}_{metric}")
    grid = np.linspace(-3, 3, 151)
    a, b = np.meshgrid(grid, grid)
    coordinates = np.column_stack([a.ravel(), b.ravel()])
    surface = {name:model.predict(coordinates).reshape(a.shape) for name,model in models.items()}
    vmin = min(float(ds.train.y.min()), *(float(z.min()) for z in surface.values()))
    vmax = max(float(ds.train.y.max()), *(float(z.max()) for z in surface.values()))
    levels = np.linspace(vmin, vmax, 31)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    axes[0].scatter(*ds.train.X.T, c=ds.train.y, s=12, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Training responses")
    for ax, (name, z) in zip(axes[1:], surface.items()):
        im = ax.contourf(a, b, z, levels=levels, cmap="viridis")
        ax.set_title(name+" fitted function")
    for ax in axes:
        ax.set(xlim=(-3,3), ylim=(-3,3), xlabel="$x_1$", ylabel="$x_2$")
    fig.colorbar(im, ax=axes, label="Response", shrink=.85)
    savefig(fig, out, "q3_surface_comparison")
    summary = {"n_train":len(y), "n_validation":len(yv), "n_test":len(ds.test.X),
               "best_rks":best_rks, "best_mlp":best_mlp, "comparison":comparison,
               "refit_data":"training only; fixed selected epoch for MLP",
               "test_predictions":"200 header-free rows per model, in supplied test.csv order",
               "test_labels_available":False, "runtime_seconds":time.monotonic()-start}
    dump(out/"q3_summary.json", summary)
    return summary
