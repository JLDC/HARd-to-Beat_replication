import numpy as np
import pandas as pd

from functools import partial

from typing import Iterator


class DataLoader:
    def __init__(
        self,
        df: pd.DataFrame,
        training_window: int,
        reestimation_frequency: int,
        window_style: str,
        pooled: bool,
        y_col: str = "rv_target",
    ):
        self.training_window = training_window
        self.reestimation_frequency = reestimation_frequency
        self.window_style = window_style
        self.pooled = pooled
        self.y_col = y_col

        # Make X and y arrays
        # X is (n_timestamps, n_symbols, n_features)
        self.X = np.stack(
            [X.to_numpy() for _, X in df.drop(columns=y_col).groupby("symbol")], axis=1
        )
        # y is (n_timestamps, n_symbols)
        self.y = []
        self.pred_symbols = []
        for symbol, y in df[y_col].groupby("symbol"):
            self.y.append(y.to_numpy())
            self.pred_symbols.append(symbol)
        self.y = np.stack(self.y, axis=1)
        self.pred_symbols = np.array(self.pred_symbols)

        # Some useful quantities to know (also accessible from Learner later on)
        self.n_timestamps, self.n_symbols = self.y.shape
        self.pred_timestamps = df.index.get_level_values("timestamp").unique()[
            self.training_window :
        ]

        # Define moving window functions
        if self.window_style == "moving" or self.window_style == "rolling":
            self.train_window_fn = partial(
                _moving_window, w=self.training_window, s=self.reestimation_frequency
            )
            self.pred_window_fn = partial(
                _moving_window,
                w=self.reestimation_frequency,
                s=self.reestimation_frequency,
            )
        elif self.window_style == "expanding":
            self.train_window_fn = partial(
                _expanding_window, w=self.training_window, s=self.reestimation_frequency
            )
            # Even if we use expanding windows, we still need to predict at the reestimation frequency
            self.pred_window_fn = partial(
                _moving_window,
                w=self.reestimation_frequency,
                s=self.reestimation_frequency,
            )
        elif self.window_style == "static":
            self.train_window_fn = _static_window
            self.pred_window_fn = _static_window
        else:
            raise ValueError(f"Unknown window style: {self.window_style}")

    def __iter__(self):
        return self.get_windows()

    def __len__(self):
        if self.window_style == "static":
            return 1
        if self.reestimation_frequency != 1:
            raise NotImplementedError("Non-unit reestimation frequency not implemented")
        else:
            if self.pooled:
                return self.n_timestamps - self.training_window
            else:
                return (self.n_timestamps - self.training_window) * self.n_symbols

    @property
    def test_start(self):
        return self.training_window

    @property
    def train_end(self):
        if self.window_style == "static":
            return self.training_window
        else:
            return -1

    def get_windows(self):
        if self.pooled:
            # Simple windowing on the 0-axis and flattening of 1-axis
            for X_train, X_pred, y_train, y_pred in zip(
                self.train_window_fn(self.X[: self.train_end]),
                self.pred_window_fn(self.X[self.test_start :]),
                self.train_window_fn(self.y[: self.train_end]),
                self.pred_window_fn(self.y[self.test_start :]),
            ):
                X_train = X_train.reshape(-1, X_train.shape[-1])
                X_pred = X_pred.reshape(-1, X_pred.shape[-1])
                y_train = y_train.flatten()
                y_pred = y_pred.flatten()
                yield X_train, X_pred, y_train, y_pred
        else:
            # Move axis and iterate over the 1-axis (symbols) to speed up
            X = np.moveaxis(self.X, 1, 0)
            y = np.moveaxis(self.y, 1, 0)
            for j in range(self.n_symbols):
                Xj, yj = X[j], y[j]
                for X_train, X_pred, y_train, y_pred in zip(
                    self.train_window_fn(Xj[: self.train_end]),
                    self.pred_window_fn(Xj[self.test_start :]),
                    self.train_window_fn(yj[: self.train_end]),
                    self.pred_window_fn(yj[self.test_start :]),
                ):
                    yield X_train, X_pred, y_train, y_pred


# This stuff is slightly tricky but faster than sliding over an axis that is not the first one
def _moving_window(
    arr: np.ndarray, w: int, s: int, all_once: bool = False
) -> Iterator[np.ndarray]:
    n = arr.shape[0]
    for i in range(0, n - w + 1, s):
        yield arr[i : i + w]

    # If we don't have a stride of 1 and a multiple of w, we need to handle the last window
    if s > 1 and n % w != 0:
        raise NotImplementedError


def _expanding_window(arr: np.ndarray, w: int, s: int) -> Iterator[np.ndarray]:
    n = arr.shape[0]
    for i in range(0, n - w + 1, s):
        yield arr[: i + w]

    if s > 1 and n % w != 0:
        raise NotImplementedError


# Just yield the whole array, no windowing
def _static_window(arr: np.ndarray) -> Iterator[np.ndarray]:
    yield arr
