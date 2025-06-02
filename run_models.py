from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os
import yaml

from data import load_stocks, get_symbols, DataLoader
from models import CVLearner, Learner, WeightedLeastSquares

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from warnings import simplefilter

from typing import Optional


def get_rv_col(data_config: dict):
    if data_config["kernel"] == "preavg":
        return "rv05_preavg"
    elif data_config["kernel"] is None or data_config["kernel"] == "None":
        return "rv05"
    else:
        raise ValueError(f"Invalid kernel: {data_config['kernel']}")


def get_data(data: pd.DataFrame, vix: bool, up_to: Optional[str] = None):
    if up_to is not None:
        data = data.loc[data.index.get_level_values("timestamp") < up_to]
    return data if vix else data.drop(columns=["vix"])


def make_name(vix: bool, pooled: bool):
    return f"{'vix' if vix else 'novix'}{'_pooled' if pooled else ''}"


def run_har_model(estimator: BaseEstimator, loader: DataLoader, filepath: str):
    learner = Learner(estimator, loader)
    learner.fit_predict()
    learner.save_results(filepath)


def run_ml_model(
    estimator: BaseEstimator,
    train_loader: DataLoader,
    test_loader: DataLoader,
    filepath: str,
    cfg: dict,
):
    learner = CVLearner(
        model=estimator,
        train_loader=train_loader,
        test_loader=test_loader,
        hyperparams=cfg["hyperparams"],
    )
    learner.fit_predict()
    learner.save_results(filepath)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (yaml)",
    )
    parser.add_argument(
        "-v", "--verbose", help="Whether to print verbose output", default=True
    )
    parser.add_argument(
        "-o", "--overwrite", help="Whether to overwrite existing results", default=False
    )
    parser.add_argument(
        "-w", "--warnings", help="Whether to filter convergence warnings", default=True
    )
    args = parser.parse_args()

    if args.warnings:
        simplefilter("ignore", category=ConvergenceWarning)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cfg_data = config["data"]
    cfg_har = config["models"]["har"]
    cfg_gbt = config["models"]["gbt"]
    cfg_rf = config["models"]["rf"]
    cfg_ffnn = config["models"]["ffnn"]
    cfg_lasso = config["models"]["lasso"]
    # For lasso we transform the alpha hyperparam to a range of values
    cfg_lasso["hyperparams"]["lasso__alpha"] = np.logspace(
        cfg_lasso["hyperparams"]["alpha"]["from"],
        cfg_lasso["hyperparams"]["alpha"]["to"],
        cfg_lasso["hyperparams"]["alpha"]["total"],
    )
    del cfg_lasso["hyperparams"]["alpha"]

    for index in cfg_data["index"]:
        # %% =============================================================================
        # Run HAR models
        # ================================================================================
        # We don't have any train / test separation for HAR models as we don't validate
        # any type of hyperparameters. We just run the model on the whole sample.
        if args.verbose:
            print(f"Running HAR models for index: {index}")
            print("-" * 80)

        # Load stock data and create data loaders
        data = load_stocks(
            symbols=get_symbols(index),
            path=cfg_data["path"],
            rv_col=get_rv_col(cfg_data),
            rolling=[5, 22],
            add_vix=True,  # Always add VIX, we can drop later if needed
            start_date=cfg_data["start_date"],
            transform=np.log,
        )

        for vix in [True, False]:
            for pooled in [True, False]:
                loader = DataLoader(
                    df=get_data(data, vix),
                    training_window=cfg_har["training_window"],
                    reestimation_frequency=cfg_har["reestimation_frequency"],
                    window_style="rolling",
                    pooled=pooled,
                    y_col="rv_target",
                )

                name = make_name(vix, pooled)
                # Make folder for results if it doesn't exist
                path = os.path.join("results", get_rv_col(cfg_data), index)
                os.makedirs(path, exist_ok=True)

                # OLS
                filename = f"har_ols_{name}.pkl"
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath) and not args.overwrite:
                    print(
                        f"Skipping {filename} as it already exists (specify --overwrite to overwrite)"
                    )
                else:
                    run_har_model(LinearRegression(n_jobs=-1), loader, filepath)

                # WLS
                filename = f"har_wls_{name}.pkl"
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath) and not args.overwrite:
                    print(
                        f"Skipping {filename} as it already exists (specify --overwrite to overwrite)"
                    )
                else:
                    run_har_model(WeightedLeastSquares(), loader, filepath)

        # %% =============================================================================
        # Run ML models (static)
        # ================================================================================
        # We must split the data into training and testing sets.
        if args.verbose:
            print("=" * 80, "\n\n")
            print(f"Running Machine Learning models for index: {index}")
            print("-" * 80)

        data = load_stocks(
            symbols=get_symbols(index),
            path=cfg_data["path"],
            rv_col=get_rv_col(cfg_data),
            rolling=[5, 22],
            lags=range(1, 101),
            add_vix=True,  # Always add VIX, we can drop later if needed
            start_date=cfg_data["start_date"],
            transform=np.log,
        )

        train_window = (
            data.index.get_level_values("timestamp").unique() < cfg_data["start_valid"]
        ).sum()
        test_window = (
            data.index.get_level_values("timestamp").unique() < cfg_data["start_test"]
        ).sum()

        for vix in [True, False]:
            for pooled in [True, False]:
                # Create train and test loaders (train contains validation set)
                train_loader = DataLoader(
                    df=get_data(data, vix, cfg_data["start_test"]),
                    training_window=train_window,
                    reestimation_frequency=None,
                    window_style="static",
                    pooled=pooled,
                    y_col="rv_target",
                )

                test_loader = DataLoader(
                    df=get_data(data, vix),
                    training_window=test_window,
                    reestimation_frequency=None,
                    window_style="static",
                    pooled=pooled,
                    y_col="rv_target",
                )

                suffix = make_name(vix, pooled)
                # Make folder for results if it doesn't exist
                path = os.path.join("results", get_rv_col(cfg_data), index)
                os.makedirs(path, exist_ok=True)

                # Define ML models and matching configs and names
                models = (
                    # Lasso
                    Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())]),
                    # Gradient Boosting
                    HistGradientBoostingRegressor(),
                    # Random Forest
                    RandomForestRegressor(n_jobs=-1),
                    # Feedforward Neural Network
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("ffnn", MLPRegressor(max_iter=500, batch_size=32)),
                        ]
                    ),
                )
                cfgs = [cfg_lasso, cfg_gbt, cfg_rf, cfg_ffnn]
                names = ["lasso", "gbt", "rf", "ffnn"]

                for model, cfg, name in zip(models, cfgs, names):
                    filename = f"{name}_{suffix}.pkl"
                    filepath = os.path.join(path, filename)
                    if os.path.exists(filepath) and not args.overwrite:
                        print(
                            f"Skipping {filename} as it already exists (specify --overwrite to overwrite)"
                        )
                    else:
                        run_ml_model(model, train_loader, test_loader, filepath, cfg)
