import argparse
import os
import wrds

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from datetime import datetime
from utils import get_connector, get_credentials, get_timestring

from typing import Optional


def build_taq_query(
    date: str,
    time_open: str,
    time_close: str,
    sym_root: Optional[str] = None,
    sym_suffix: str = "NULL",
) -> str:
    if sym_root is None and sym_suffix is not None:
        raise AssertionError("sym_suffix must be None if sym_root is None.")
    asc_string = " OR ".join([f"tr_scond LIKE '%%{asc}'" for asc in "BCHIKLMNOPQRUVXZ"])
    qry = " ".join(
        [
            "SELECT time_m, price, size, sym_root, sym_suffix",
            f"FROM taqmsec.ctm_{date}",
            # P1. Only entries within opening hours
            f"WHERE time_m BETWEEN '{time_open}' AND '{time_close}'",
            # P2. Only entries with positive prices
            f"AND price > 0",
            # P3. Only entries from NYSE, AMEX, and/or Nasdaq
            f"AND (ex = 'N' OR ex = 'T' OR ex = 'Q' OR ex = 'A')"
            # T1. Only entries without corrected trades
            f"AND tr_corr = '00'"
            # T2. Only entries without abnormal sale condition
            f"AND (tr_scond IS NULL OR NOT ({asc_string}))",
        ]
    )
    # Restrict to specific symbol
    if sym_root is not None:
        qry += f" AND sym_root = '{sym_root}'"
        if sym_suffix is not None:
            if sym_suffix == "NULL":
                qry += " AND sym_suffix IS NULL"
            else:
                qry += f" AND sym_suffix = '{sym_suffix}'"
    return qry


# NOTE: This only works because of numba engine, be careful if using this without!
def rolling_mad_excess(x: np.ndarray) -> bool:
    xneg = np.hstack((x[:25], x[26:]))
    med = np.median(xneg)
    mad = np.mean(np.abs(xneg - med))
    return np.abs(x[25] - med) > 10 * mad


def preprocess_taq_data(
    data: pd.DataFrame, date: str, aggregation_freq: str = "5min"
) -> pd.DataFrame:
    # Transform times
    data["timestamp"] = pd.to_datetime(
        data["time_m"].apply(lambda x: f"{date} {x}"), format="ISO8601"
    )
    data.drop(columns=["time_m"], inplace=True)
    # Replace empty suffixes (for aggregation purposes)
    data.fillna({"sym_suffix": ""}, inplace=True)
    # T3. Use the median when multiple transactions have the same timestamp
    grp_cols = ["timestamp", "sym_root", "sym_suffix"]
    data = data.groupby(grp_cols).agg({"price": "median", "size": "sum"}).reset_index()
    # T4 (Q4). Remove entries where the price deviates by more than 10 mean absolute
    # deviations from a rolling centered median around 50 observations
    if data.shape[0] > 50:
        data["excess"] = (
            data["price"]
            .rolling(51, center=True)
            .apply(rolling_mad_excess, engine="numba", raw=True)
        )
    else:
        data["excess"] = 0
    data.index = data["timestamp"]
    data = data.loc[data["excess"] != 1, ["sym_root", "sym_suffix", "price", "size"]]
    data = data.resample(aggregation_freq).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        logreturn=(
            "price",
            lambda x: np.nan if len(x) == 0 else np.log(x.iloc[-1] / x.iloc[0]),
        ),
    )
    return data


def get_available_days(
    conn: wrds.Connection, date_start: datetime, date_end: datetime, prefix: str = "ctm"
) -> list[int]:
    days = []
    # Get all days within the years
    for year in range(date_start.year, date_end.year + 1):
        days += [
            int(t[4:])
            for t in conn.list_tables(library=f"taqm_{year}")
            if t.startswith(prefix)
        ]
    # Drop before/after date range and return
    ds = int(date_start.strftime("%Y%m%d"))
    de = int(date_end.strftime("%Y%m%d"))
    return [d for d in sorted(days) if (ds <= d <= de)]


def download_full_data(
    outdir: str,
    conn: wrds.Connection,
    sym_root: str,
    sym_suffix: str,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
    bcal: xcals.ExchangeCalendar = xcals.get_calendar("NYSE"),
    verbose: bool = True,
) -> None:
    not os.path.isdir(outdir) and os.mkdir(outdir)
    schedule = bcal.schedule.loc[date_start:date_end]
    for date in schedule.index:
        time_open = get_timestring(schedule.loc[date]["open"])
        time_close = get_timestring(schedule.loc[date]["close"])
        day = date.strftime("%Y%m%d")
        # Obtain data and store to disk
        data = preprocess_taq_data(
            conn.raw_sql(
                build_taq_query(day, time_open, time_close, sym_root, sym_suffix)
            ),
            day,
        )
        data.to_pickle(f"{outdir}/{day}.pkl")
    verbose and print(f"{sym_root} done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", metavar="path", required=True, help="Output directory"
    )
    parser.add_argument(
        "-r", "--daterange", required=True, help="Date range, format YYYYMMDD-YYYYMMDD"
    )
    parser.add_argument(
        "-c",
        "--credentials",
        metavar="path",
        required=True,
        help="Path to the user credentials",
    )
    parser.add_argument(
        "-s",
        "--symbol",
        required=True,
        help="Symbol root and suffix, separated by a dot, e.g., GOOG.A",
    )
    parser.add_argument("-v", "--verbose", default=True, help="Verbose")

    args = parser.parse_args()

    # Create the symbol root and suffix
    if "." in args.symbol:
        sym_root, sym_suffix = args.symbol.split(".")
    else:
        sym_root, sym_suffix = args.symbol, "NULL"

    conn = get_connector(get_credentials(args.credentials))

    dates = [
        datetime(int(x[:4]), int(x[4:6]), int(x[6:])) for x in args.daterange.split("-")
    ]

    download_full_data(
        args.dir, conn, sym_root, sym_suffix, dates[0], dates[1], verbose=args.verbose
    )
