import argparse
import pandas as pd
import os

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from typing import Optional


def calculate_rv(
    data: pd.DataFrame, freq: str, time_range: Optional[tuple[str, str]] = None
) -> float:
    if time_range is not None:
        data = data.between_time(*time_range)
    return data["logreturn"].resample(freq).sum().pow(2).sum()


def calculate_preaveraged_rv(
    data: pd.DataFrame, freq: str, time_range: Optional[tuple[str, str]] = None
) -> float:
    if time_range is not None:
        data = data.between_time(*time_range)
    return (
        data["logreturn"]
        .rolling(5, min_periods=1)
        .mean()
        .resample(freq)
        .sum()
        .pow(2)
        .sum()
    )


def aggregate_single_df(
    data: pd.DataFrame, aggregation_freq: str = "1D", range_freq: str = "30min"
) -> pd.DataFrame:
    # For some obscure reason, we may have data on the 16:00 timestamp
    # probably some unclean data (very low volume, no returns)
    data = data.between_time("9:30", "15:55")
    agg_data = data.resample(aggregation_freq).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    # Calculating RV with different intervals and renaming columns accordingly
    agg_data["rv05"] = calculate_rv(data, "5min")
    agg_data["rv05_preavg"] = calculate_preaveraged_rv(data, "5min")
    agg_data["rv10"] = calculate_rv(data, "10min")
    agg_data["rv10_preavg"] = calculate_preaveraged_rv(data, "10min")
    agg_data["rv15"] = calculate_rv(data, "15min")
    agg_data["rv15_preavg"] = calculate_preaveraged_rv(data, "15min")
    agg_data["rv30"] = calculate_rv(data, "30min")
    agg_data["rv30_preavg"] = calculate_preaveraged_rv(data, "30min")

    # Calculating RV for specific time ranges
    if range_freq is not None:
        vol_data = data["volume"].resample(range_freq).sum()
        rv_data = (
            data["logreturn"].resample("5T").sum().pow(2).resample(range_freq).sum()
        )
        cols_time = rv_data.index.strftime("%H:%M")

        agg_data["volume_" + cols_time] = vol_data.values
        agg_data["rv_" + cols_time] = rv_data.values

    return agg_data


def aggregate_daily_files(dir: str) -> pd.DataFrame:
    dfs = []
    for f in os.listdir(dir):
        df = pd.read_pickle(f"{dir}/{f}")
        if len(df):
            dfs.append(aggregate_single_df(df))
    if len(dfs):
        return pd.concat(dfs).sort_index()
    else:
        return pd.DataFrame({})
    

def process_stock(stock: str, overwrite: bool = True) -> None:
    dir_in = "data_raw"
    dir_out = "data_agg"

    if not overwrite:
        if f"{stock}.pkl" in os.listdir(dir_out):
            return
    
    df = aggregate_daily_files(f"{dir_in}/{stock}")
    if len(df):
        df.to_pickle(f"{dir_out}/{stock}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", 
        metavar="path",
        default="data_raw",
        help="Input directory containing raw data files"
    )
    parser.add_argument(
        "-o", "--output",
        metavar="path", 
        default="data_agg",
        help="Output directory for aggregated data"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing files"
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=os.cpu_count(),
        help="Number of processes to use"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    stocks = os.listdir(args.input)
    
    ps = partial(process_stock, overwrite=args.overwrite)
    
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        list(tqdm(executor.map(ps, stocks), total=len(stocks)))