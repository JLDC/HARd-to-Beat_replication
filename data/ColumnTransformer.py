import pandas as pd

from warnings import warn

from .core import default, exists

from typing import Optional


class ColumnTransformer:
    def __init__(
        self,
        col: str | list[str],
        fn: Optional[callable] = None,
        lags: Optional[list[int]] = None,
        rolling: Optional[list[int]] = None,
        groupby: Optional[str] = None,
    ):
        self.col = col
        self.fn = fn
        self.lags = default(lags, [])
        self.rolling = default(rolling, [])
        self.groupby = groupby
        if any([lag < 0 for lag in self.lags]) or any(
            [roll < 0 for roll in self.rolling]
        ):
            raise ValueError("Lags must be non-negative")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if exists(self.groupby):
            return df.groupby(self.groupby).apply(self._transform_single)
        else:
            if isinstance(df.index, pd.MultiIndex):
                warn(
                    "Index is not unique and groupby is not specified. "
                    + "This might lead to unwanted results."
                )
            return self._transform_single(df)

    def _transform_single(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.groupby:
            df = df.droplevel("ticker", axis=0)
        if exists(self.fn):
            for col in [self.col] if isinstance(self.col, str) else self.col:
                df[col] = self.fn(df[col])

        new_data = {}
        for col in [self.col] if isinstance(self.col, str) else self.col:
            for lag in self.lags:
                new_data[f"{col}_lag_{lag}"] = df[col].shift(lag)

            for roll in self.rolling:
                new_data[f"{col}_roll_{roll}"] = df[col].rolling(roll).mean()
        return pd.concat([df, pd.DataFrame(new_data, index=df.index)], axis=1)

    @property
    def maxlag(self) -> int:
        return max(max(self.rolling, default=0) - 1, max(self.lags, default=0))
