"""Models for HW2. RKS uses exactly the paired sinusoidal features in Q3."""
import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.nn import functional as F


class RandomKitchenSinks:
    def __init__(self, D=256, sigma=1., ridge_lambda=1e-3, seed=2026):
        if D < 2 or D % 2 or sigma <= 0 or ridge_lambda <= 0:
            raise ValueError("D must be positive/even; sigma and lambda must be positive.")
        self.D, self.sigma, self.ridge_lambda, self.seed = D, sigma, ridge_lambda, seed

    def _features(self, X):
        phase = np.asarray(X, dtype=np.float64) @ self.weights_.T + self.phase_
        result = np.empty((len(X), self.D), dtype=np.float64)
        result[:, 0::2], result[:, 1::2] = np.sin(phase), np.cos(phase)
        return result

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        # Separate streams ensure nested draws when only D changes.
        self.weights_ = np.random.default_rng(self.seed).normal(0, 1/self.sigma, (self.D//2, X.shape[1]))
        self.phase_ = np.random.default_rng(self.seed+1).uniform(0, 2*np.pi, self.D//2)
        phi = self._features(X)
        self.feature_mean_ = phi.mean(axis=0)
        self.target_mean_ = float(y.mean())
        centered, target = phi-self.feature_mean_, y-self.target_mean_
        penalty = len(y)*self.ridge_lambda
        # Minimize mean squared error + lambda ||beta||^2; intercept unpenalized.
        if self.D <= len(y):
            gram = centered.T @ centered
            gram.flat[::len(gram)+1] += penalty
            self.coef_ = linalg.cho_solve(linalg.cho_factor(gram), centered.T @ target)
        else:
            gram = centered @ centered.T
            gram.flat[::len(gram)+1] += penalty
            self.coef_ = centered.T @ linalg.cho_solve(linalg.cho_factor(gram), target)
        self.intercept_ = self.target_mean_ - self.feature_mean_ @ self.coef_
        return self

    def predict(self, X):
        return self._features(X) @ self.coef_ + self.intercept_

    def save(self, path):
        np.savez_compressed(path, weights=self.weights_, phase=self.phase_, coef=self.coef_,
                            intercept=self.intercept_, D=self.D, sigma=self.sigma,
                            ridge_lambda=self.ridge_lambda, seed=self.seed)


class MLP(nn.Module):
    """Exactly one hidden layer; tanh is fixed across the Q3 width sweep."""
    def __init__(self, d_in=2, d_hidden=64, activation="tanh"):
        super().__init__()
        activation_layer = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), activation_layer(), nn.Linear(d_hidden, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(torch.as_tensor(X, dtype=torch.float32)).numpy()


class GaussianMLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 64), sigma_floor=.001):
        super().__init__()
        self.sigma_floor = sigma_floor
        modules, width = [], n_features
        for next_width in hidden:
            modules += [nn.Linear(width, next_width), nn.ReLU()]
            width = next_width
        modules.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*modules)
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias[:] = torch.tensor([0., np.log(np.expm1(1.-sigma_floor))])

    def forward(self, x):
        raw = self.net(x)
        return torch.stack([raw[:, 0], F.softplus(raw[:, 1])+self.sigma_floor], dim=1)


def gaussian_nll(prediction, y):
    mu, sigma = prediction[:, 0], prediction[:, 1]
    return (.5*np.log(2*np.pi) + sigma.log() + .5*((y-mu)/sigma).square()).mean()


class RetailMLP(nn.Module):
    def __init__(self, category_sizes, mode, embedding_dim=16, hidden=(64, 32), dropout=.05):
        super().__init__()
        if mode not in {"target", "embedding", "combined"}:
            raise ValueError(mode)
        self.mode = mode
        self.embeddings = nn.ModuleList()
        if mode != "target":
            self.embeddings = nn.ModuleList([nn.Embedding(n+1, embedding_dim, padding_idx=0) for n in category_sizes])
        width = (2*embedding_dim if mode != "target" else 0) + (2 if mode != "embedding" else 0)
        layers = []
        for next_width in hidden:
            layers += [nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(dropout)]
            width = next_width
        layers += [nn.Linear(width, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, categories, target_features):
        features = []
        if self.mode != "target":
            features += [layer(categories[:, j]) for j, layer in enumerate(self.embeddings)]
        if self.mode != "embedding":
            features.append(target_features)
        # Nonnegative prediction on log(1+Quantity) scale.
        return self.net(torch.cat(features, dim=1)).squeeze(-1)
