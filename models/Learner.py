import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from data import DataLoader


class AbstractLearner:
    @property
    def reconstructed_preds(self):
        return pd.DataFrame(
            {"y_true": self.y_true_, "y_pred": self.y_pred_},
            index=pd.MultiIndex.from_product(
                (self.pred_symbols, self.pred_timestamps), names=["symbol", "timestamp"]
            ),
        )

    def save_results(self, path: str):
        self.reconstructed_preds.to_pickle(path)


class Learner(AbstractLearner):
    def __init__(
        self,
        model: BaseEstimator,
        data_loader: DataLoader,
        show_progress: bool = True,
    ):
        self.model = model
        self.data_loader = data_loader
        self.y_pred_ = None
        self.y_true_ = None
        self.show_progress = show_progress

    def __getattr__(self, name: str):
        if name in self.data_loader.__dict__:
            return getattr(self.data_loader, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def fit_predict(self):
        y_pred, y_true = [], []
        itr = self.data_loader
        if self.show_progress:
            itr = tqdm(
                itr, total=len(itr), desc="Rolling windows estimation", unit="window"
            )
        for X_train, X_test, y_train, y_test in itr:
            self.model.fit(X_train, y_train)
            y_pred.append(self.model.predict(X_test))
            y_true.append(y_test)
        self.y_pred_ = np.concatenate(y_pred)
        self.y_true_ = np.concatenate(y_true)

        # If pooled, make sure to have all preds for same symbol contiguous
        if self.pooled:
            self.y_pred_ = np.concatenate(self.y_pred_.reshape(-1, self.n_symbols).T)
            self.y_true_ = np.concatenate(self.y_true_.reshape(-1, self.n_symbols).T)

        return self


class CVLearner(AbstractLearner):
    def __init__(
        self,
        model: BaseEstimator,
        train_loader: DataLoader,
        test_loader: DataLoader,
        hyperparams: dict,
        show_progress: bool = True,
    ):
        self.model = model
        _validate_loaders(train_loader, test_loader)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.hyperparams = hyperparams
        self.show_progress = show_progress

    def __getattr__(self, name: str):
        if name in self.test_loader.__dict__:
            return getattr(self.test_loader, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @property
    def hyperparam_grid(self):
        k, v = zip(*self.hyperparams.items())
        return (dict(zip(k, x)) for x in product(*v))

    @property
    def grid_len(self):
        return np.prod(list(map(len, self.hyperparams.values())))

    def fit(self):
        itr = self.train_loader

        if self.show_progress:
            itr = tqdm(
                itr,
                total=self.train_loader.n_symbols,
                desc="Cross-validation",
                unit="stock",
            )

        self.best_params_ = []
        for X_train, X_valid, y_train, y_valid in itr:
            best_params = None
            best_score = float("inf")

            for params in self.hyperparam_grid:
                self.model.set_params(**params)
                self.model.fit(X_train, y_train)
                score = mean_squared_error(y_valid, self.model.predict(X_valid))
                if score < best_score:
                    best_score = score
                    best_params = params

            self.best_params_.append(best_params)
        self.best_params_ = {
            k: v for k, v in zip(self.train_loader.pred_symbols, self.best_params_)
        }

        return self

    def predict(self):
        y_pred, y_true = [], []

        itr = self.test_loader

        if self.show_progress:
            itr = tqdm(
                itr,
                total=len(itr),
                desc="Prediction",
                unit="stock" if self.test_loader.window_style == "static" else "window",
            )

        symbol_idx = 0
        symbols = self.test_loader.pred_symbols
        n_timestamps = len(self.test_loader) / self.test_loader.n_symbols
        i = 0
        this_symbol = symbols[symbol_idx]
        self.model.set_params(**self.best_params_[this_symbol])
        for X_train, X_test, y_train, y_test in itr:
            self.model.fit(X_train, y_train)
            y_pred.append(self.model.predict(X_test))
            y_true.append(y_test)
            i += 1
            # Check if we need to update the model
            if i == n_timestamps:
                i = 0
                symbol_idx += 1
                if symbol_idx == self.test_loader.n_symbols:
                    break
                this_symbol = symbols[symbol_idx]
                self.model.set_params(**self.best_params_[this_symbol])

        self.y_pred_ = np.concatenate(y_pred)
        self.y_true_ = np.concatenate(y_true)

        # If pooled, make sure to have all preds for same symbol contiguous
        if self.pooled:
            self.y_pred_ = np.concatenate(self.y_pred_.reshape(-1, self.n_symbols).T)
            self.y_true_ = np.concatenate(self.y_true_.reshape(-1, self.n_symbols).T)

        return self

    def fit_predict(self):
        self.fit()
        self.predict()

        return self


def _validate_loaders(train_loader: DataLoader, test_loader: DataLoader):
    if train_loader.window_style != "static":
        raise ValueError("CVLearner only works with static windows for training")

    if any(train_loader.pred_symbols != test_loader.pred_symbols):
        raise ValueError("train_loader and test_loader must have same symbols")

    if train_loader.y_col != test_loader.y_col:
        raise ValueError("train_loader and test_loader must have same target column")
