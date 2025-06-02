import numpy as np


def squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return (y_true - y_pred) ** 2


def squared_error_log(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return squared_error(np.exp(y_pred), np.exp(y_true))


def squared_error_log_scaled(
    y_pred: np.ndarray, y_true: np.ndarray, scale: float = 1e4
) -> np.ndarray:
    return squared_error(np.exp(y_pred) * scale, np.exp(y_true) * scale)


def qlike(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    x = y_true / y_pred
    return x - np.log(x) - 1


def qlike_log(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return qlike(np.exp(y_pred), np.exp(y_true))


def realized_utility(
    y_pred: np.ndarray, y_true: np.ndarray, sr: float = 0.4, gamma: float = 2.0
) -> np.ndarray:
    d = np.exp(y_true - y_pred)
    return sr**2 / gamma * (np.sqrt(d) - 0.5 * d)


def realized_utility_transaction_costs(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    costs: np.ndarray,
    sr: float = 0.4,
    gamma: float = 2.0,
):
    ru = realized_utility(y_pred, y_true, sr, gamma)
    return ru - costs
