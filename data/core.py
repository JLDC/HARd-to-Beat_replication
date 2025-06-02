import json
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from utils import exists, default
from .ColumnTransformer import ColumnTransformer

# Typing helpers
from typing import Optional, Callable


def load_stocks(
    symbols: list[str] | str,
    path: str,
    show_progress: bool = True,
    expand_index: bool = True,
    **kwargs,
) -> pd.DataFrame:

    if isinstance(symbols, str):
        symbols = [symbols]

    if show_progress:
        symbols = tqdm(
            symbols, desc="Loading stocks", unit="symbol", total=len(symbols)
        )

    df = pd.concat(
        [
            prepare_data(pd.read_pickle(os.path.join(path, f"{symbol}.pkl")), **kwargs)
            for symbol in symbols
        ],
        axis=0,
        keys=symbols,
        names=["symbol", "timestamp"],
    )

    if expand_index:
        all_symbols = df.index.get_level_values("symbol").unique()
        all_timestamps = df.index.get_level_values("timestamp").unique()
        df = df.reindex(
            pd.MultiIndex.from_product(
                [all_symbols, all_timestamps], names=["symbol", "timestamp"]
            ),
            fill_value=np.nan,
        )

    return df


def prepare_data(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    rv_col: Optional[str] = None,
    lags: Optional[list[int]] = None,
    rolling: Optional[list[int]] = None,
    transform: Optional[Callable] = None,
    target_horizon: int = 1,
    add_vix: bool = False,
    zeros_as_na: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    column_transformers: Optional[list[ColumnTransformer]] = None,
) -> pd.DataFrame:

    if exists(rv_col):
        df.rename(columns={rv_col: "rv"}, inplace=True)

    if exists(column_transformers):
        if isinstance(column_transformers, ColumnTransformer):
            column_transformers = [column_transformers]
        elif not isinstance(column_transformers, list):
            raise ValueError("Invalid column transformers.")

    # Input validation
    if "rv" not in df.columns:
        raise ValueError(
            "`df` must contain a 'rv' column with the realized volatility estimator."
        )
    if df.index.name != "timestamp" or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "`df` must have a DatetimeIndex with 'timestamp' as index name."
        )

    # Make sure to keep at least the rv column
    cols = default(cols, ["rv"])
    lags = _validate_rv_lags_rolling(lags, "lags")
    rolling = _validate_rv_lags_rolling(rolling, "rolling")

    if "rv" not in cols:
        cols = ["rv"] + cols

    # Make sure there are no duplicates
    cols = list(set(cols))
    df = df[cols]

    if zeros_as_na:
        df.replace(0, np.nan, inplace=True)

    # Transform realized volatilities
    ct = ColumnTransformer(col="rv", fn=transform, lags=lags, rolling=rolling)
    df = ct.transform(df)

    # Compute target and add it at the beginning of the dataframe
    df["rv_target"] = df["rv"].shift(-target_horizon)
    df = df[["rv_target"] + df.columns.tolist()[:-1]]

    # Drop the lost targets due to shifting and lagging
    df = df.iloc[ct.maxlag : -target_horizon]

    if add_vix:
        vix = pd.read_pickle("data/vix.pkl")
        df = pd.concat([df, vix], axis=1, join="inner")

    # Apply column transformers
    if exists(column_transformers):
        for ct in column_transformers:
            df = ct.transform(df)

    # Filter by date
    if exists(start_date):
        df = df.loc[df.index >= start_date]
    if exists(end_date):
        df = df.loc[df.index <= end_date]

    return df


# Validate the rv_lags and rv_rolling inputs
def _validate_rv_lags_rolling(
    vals: list[int] | int | range | None, name: str
) -> list[int]:
    if vals is None:
        return []
    if isinstance(vals, list):
        return vals
    elif isinstance(vals, int):
        return [vals]
    elif isinstance(vals, range):
        return list(vals)
    else:
        raise ValueError(f"Invalid type for `{name}`, must be a list of ints.")


def get_symbols(group: str, full_only: bool = True) -> list[str]:
    if group not in ["dow30", "nasdaq100", "all"]:
        raise ValueError("Invalid group, must be one of 'dow30', 'nasdaq100' or 'all'.")

    with open(os.path.join(os.path.dirname(__file__), "symbols.json"), "r") as f:
        stocks = json.load(f)
        if full_only and group != "all":
            stocks = list(set(stocks[group]).intersection(stocks["all"]))
        else:
            stocks = stocks[group]
        return stocks
