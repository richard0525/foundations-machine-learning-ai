"""Affine models and explicit custom losses, differentiated by PyTorch."""
import numpy as np
import torch
from torch import nn


class Affine(nn.Module):
    def __init__(self, n_features, bias=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float64))

    def forward(self, x):
        return x @ self.weight + self.bias


def logistic_probability(z):
    return torch.sigmoid(z)


def probit_probability(z):
    return torch.special.ndtr(z)


def logistic_loss(z, y):
    # Exactly BCE(sigmoid(z), y), evaluated in the log domain to avoid clipping.
    return (torch.nn.functional.softplus(z) - y * z).mean()


def probit_loss(z, y):
    # 1-Phi(z)=Phi(-z). log_ndtr avoids cancellation in either normal tail.
    return -(y * torch.special.log_ndtr(z)
             + (1.0 - y) * torch.special.log_ndtr(-z)).mean()


def mse_loss(prediction, y):
    return ((y - prediction) ** 2).mean()


def student_t_loss(prediction, y):
    # The assignment specifies df=5 and implicitly fixes location=0, scale=1
    # for residuals in ORIGINAL ALE units. Do not standardize y.
    residual_model = torch.distributions.StudentT(df=prediction.new_tensor(5.0), loc=0.0, scale=1.0)
    return -residual_model.log_prob(y - prediction).mean()


LOSSES = {"logistic": logistic_loss, "probit": probit_loss,
          "mse": mse_loss, "student_t": student_t_loss}


def fit_sgd(x, y, kind, *, epochs, learning_rate, decay_epochs, batch_size,
            seed, checkpoint_every=10, validation=None):
    """Random-reshuffling mini-batch SGD; no momentum or weight decay.

    If validation is supplied, return its best checkpoint epoch for refitting.
    Test observations are never passed to this function.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    model = Affine(x.shape[1], bias=float(y.mean()) if kind in ("mse", "student_t") else 0)
    loss_fn = LOSSES[kind]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.0, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    if validation is not None:
        xv, yv = (torch.as_tensor(v, dtype=torch.float64) for v in validation)
    history = []
    best_value, best_epoch = float("inf"), epochs
    for epoch in range(epochs + 1):
        if epoch % checkpoint_every == 0 or epoch == epochs:
            with torch.no_grad():
                train_loss = float(loss_fn(model(x), y))
                row = {"epoch": epoch, "train_loss": train_loss}
                if validation is not None:
                    value = float(loss_fn(model(xv), yv))
                    row["validation_loss"] = value
                    if epoch > 0 and value < best_value:
                        best_value, best_epoch = value, epoch
                history.append(row)
        if epoch == epochs:
            break
        rate = learning_rate / (1.0 + epoch / decay_epochs)
        optimizer.param_groups[0]["lr"] = rate
        permutation = torch.randperm(len(y), generator=generator)
        for ids in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {kind} loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    gradient_norm = float(torch.cat([p.grad.reshape(-1) for p in model.parameters()]).norm())
    return model, history, best_epoch, gradient_norm


def predict(model, x, kind):
    with torch.no_grad():
        z = model(torch.as_tensor(x, dtype=torch.float64))
        if kind == "logistic":
            z = logistic_probability(z)
        elif kind == "probit":
            z = probit_probability(z)
        return z.numpy()


def original_coefficients(model, scaler):
    w = model.weight.detach().numpy()
    return {"weight_standardized": w.tolist(),
            "bias_standardized": float(model.bias.detach()),
            "weight_original": (w / scaler.scale_).tolist(),
            "bias_original": float(model.bias.detach() - np.dot(w, scaler.mean_ / scaler.scale_))}
