import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class WeightedLeastSquares(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = None
        self.lm = LinearRegression(n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lm.fit(X, y, sample_weight=1 / np.abs(y))
        self.coef_ = self.lm.coef_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lm.predict(X)
