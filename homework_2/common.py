"""Shared deterministic setup, data integrity, metrics and artifact helpers."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def setup(seed=2026, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110})


def verify_data(data_dir):
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    for name, spec in manifest.items():
        data = (Path(data_dir) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"Data checksum mismatch: {name}")
    return manifest


def exact_row_groups(frame):
    """Identify exact full-row duplicates without treating the index as data."""
    return frame.groupby(list(frame.columns), sort=False, dropna=False).ngroup().to_numpy(np.int64)


def split_indices(n, seed, train_fraction=.7, validation_fraction=.15, groups=None):
    """Split rows, optionally keeping every member of a group in one split.

    GroupShuffleSplit proportions refer to groups, so grouped row proportions
    are approximate. The ungrouped branch preserves the retail split exactly.
    """
    indices = np.arange(n)
    ratio = round(validation_fraction/(1-train_fraction), 12)
    if groups is None:
        train, holdout = train_test_split(indices, train_size=train_fraction, random_state=seed)
        val, test = train_test_split(holdout, train_size=ratio, random_state=seed+1)
    else:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != n:
            raise ValueError("groups must be one-dimensional with one ID per row")
        train, holdout = next(GroupShuffleSplit(n_splits=1, train_size=train_fraction,
            random_state=seed).split(indices, groups=groups))
        val_positions, test_positions = next(GroupShuffleSplit(n_splits=1, train_size=ratio,
            random_state=seed+1).split(holdout, groups=groups[holdout]))
        val, test = holdout[val_positions], holdout[test_positions]
        for left, right in [(train, val), (train, test), (val, test)]:
            assert np.intersect1d(groups[left], groups[right]).size == 0
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert len(train)+len(val)+len(test) == n
    return train, val, test


def save_split(out, name, train, val, test):
    np.savez_compressed(out / f"{name}_split_indices.npz", train=train, val=val, test=test)


def savefig(fig, out, name):
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_csv_gz(frame, path):
    # Fixed gzip time makes rerun artifacts reproducible.
    frame.to_csv(path, index=False, float_format="%.12g", compression={"method": "gzip", "mtime": 0})


def batched_predict(model, arrays, batch_size=8192):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(*(a[i:i+batch_size] for a in arrays)).cpu().numpy()
                               for i in range(0, len(arrays[0]), batch_size)])


def fit_network(model, inputs, target, val_inputs, val_target, loss_fn, *, epochs,
                patience, batch_size, learning_rate, weight_decay, seed,
                checkpoint_every=1):
    """Adam training; validation chooses the epoch, never used for gradient updates.

    With val_inputs=None, run exactly epochs updates for a deterministic refit.
    The training metric is evaluated with dropout disabled, like validation.
    """
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        permutation = torch.randperm(len(target), generator=generator)
        for ix in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(*(a[ix] for a in inputs)), target[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        if epoch % checkpoint_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                def objective(arrays, truth):
                    total = 0.
                    for start in range(0, len(truth), 8192):
                        end = min(start+8192, len(truth))
                        total += float(loss_fn(model(*(a[start:end] for a in arrays)), truth[start:end])) * (end-start)
                    return total / len(truth)
                row = {"epoch": epoch, "train_loss": objective(inputs, target)}
                if val_inputs is not None:
                    value = objective(val_inputs, val_target)
                    row["validation_loss"] = value
                    if value < best:
                        best, best_epoch = value, epoch
                        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                history.append(row)
            if epoch % 20 == 0:
                print(f"  epoch {epoch}: {row}", flush=True)
            if val_inputs is not None and epoch-best_epoch >= patience:
                break
    if val_inputs is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    model.eval()
    return history, best_epoch


def tensor(x, integer=False):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long if integer else torch.float32)
