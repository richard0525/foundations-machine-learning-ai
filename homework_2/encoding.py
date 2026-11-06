"""Training-only categorical vocabularies and out-of-fold raw-target means."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class CrossFitTargetEncoder:
    def __init__(self, smoothing=20., folds=5, seed=2026):
        self.smoothing, self.folds, self.seed = smoothing, folds, seed

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(y.mean())
        self.columns_ = list(X.columns)
        self.maps_ = {}
        for name in self.columns_:
            stats = pd.DataFrame({"category":X[name].to_numpy(),"target":y}).groupby("category",sort=False).target.agg(["sum","count"])
            self.maps_[name] = (stats["sum"]+self.smoothing*self.global_mean_)/(stats["count"]+self.smoothing)
        return self

    def transform(self, X):
        return np.column_stack([X[name].map(self.maps_[name]).fillna(self.global_mean_).to_numpy(float) for name in self.columns_])

    def fit_transform(self, X, y):
        y = np.asarray(y, float)
        result = np.empty((len(y), X.shape[1]), dtype=float)
        self.fold_ids_ = np.empty(len(y), dtype=np.int8)
        for fold, (train, holdout) in enumerate(KFold(self.folds, shuffle=True, random_state=self.seed).split(X)):
            encoder = CrossFitTargetEncoder(self.smoothing, self.folds, self.seed).fit(X.iloc[train],y[train])
            result[holdout] = encoder.transform(X.iloc[holdout])
            self.fold_ids_[holdout] = fold
        self.fit(X,y)  # Full-training maps are only for validation/test/new rows.
        return result


class CategoryVocabulary:
    """Index zero is a fixed zero-vector unknown category, not a fitted label."""
    def fit(self, X):
        self.columns_ = list(X.columns)
        self.maps_ = {name:{str(v):i+1 for i,v in enumerate(sorted(X[name].unique()))} for name in self.columns_}
        self.sizes_ = [len(self.maps_[name]) for name in self.columns_]
        return self

    def transform(self, X):
        return np.column_stack([X[name].map(self.maps_[name]).fillna(0).to_numpy(np.int64) for name in self.columns_])
