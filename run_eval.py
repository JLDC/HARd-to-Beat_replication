from argparse import ArgumentParser
import yaml

import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from losses import qlike_log, realized_utility, squared_error_log_scaled

from typing import Callable

from model_confidence_set import ModelConfidenceSet as MCS


class Loss:
    def __init__(self, name: str, fn: Callable, short_name: str):
        self.name = name
        self.fn = fn
        self.short_name = short_name


# Make dataframe of all forecasts
def get_forecasts(universe: str, kernel: str) -> pd.DataFrame:
    if universe not in ["all", "dow30", "nasdaq100"]:
        raise ValueError("Invalid universe")
    if kernel not in ["rv05", "rv05_preavg"]:
        raise ValueError("Invalid kernel")

    folder = os.path.join("results", kernel, universe)
    files = os.listdir(folder)
    dfs = []
    for f in files:
        df = pd.read_pickle(os.path.join(folder, f))[["y_pred"]]
        df.rename(columns={"y_pred": f.split(".")[0]}, inplace=True)
        dfs.append(df)

    # Add true values as well
    dfs = [pd.read_pickle(os.path.join(folder, f))[["y_true"]]] + dfs
    return pd.concat(dfs, axis=1, join="inner")


# Helper for model names
def rename_model(model: str, vix: bool) -> str:
    vix_suffix = "_vix" if vix else "_novix"
    if model.endswith("_pooled"):
        model = model.replace("_pooled", vix_suffix + "_pooled")
    else:
        model = model + vix_suffix
    return model


# Helper for loss (quantile) table
def make_loss_table(
    df: pd.DataFrame, models: dict, loss: Loss, universe: str, kernel: str
) -> pd.DataFrame:
    # Make model index
    models_novix = {rename_model(k, False): v for k, v in models.items()}
    models_vix = {rename_model(k, True): v for k, v in models.items()}
    models_index = list(models_novix.keys()) + list(models_vix.keys())

    # Compute prediction errors, do this differently for rutc!
    if loss.short_name != "rutc":
        errors = df.drop(columns=["y_true"]).apply(lambda s: loss.fn(s, df["y_true"]))
        errors = errors.groupby(level="symbol").mean()
    else:
        sr = 0.4
        gamma = 2.0
        xstar_t = sr / (gamma * np.sqrt(252 * np.exp(df.drop(columns=["y_true"]))))
        xdiff = xstar_t.groupby("symbol").diff().dropna().abs()
        spread = get_spread()
        # Fill missing spread with median spread (by stock first, total then)
        spread = pd.merge(
            xdiff[["har_wls_novix"]],
            spread,
            left_index=True,
            right_index=True,
            how="left",
        )[["spread"]]
        spread = spread.groupby("symbol").transform(lambda x: x.fillna(x.median()))
        spread.fillna(spread.median(), inplace=True)
        # Ensure indices align
        spread.sort_index(inplace=True)
        xdiff.sort_index(inplace=True)
        # Drop duplicates
        spread = spread.loc[~spread.index.duplicated()]
        xdiff = xdiff.loc[~xdiff.index.duplicated()]
        costs = xdiff.apply(lambda s: s * spread["spread"].values, axis=0)
        errors = df.drop(columns=["y_true"]).apply(lambda s: loss.fn(s, df["y_true"]))
        errors = errors.groupby("symbol").mean() - costs.groupby("symbol").mean()

    # Collect results for printing
    results = {
        "mean": errors.mean(axis=0).reindex(models_index).to_numpy(),
        "q05": errors.quantile(0.05, axis=0).reindex(models_index).to_numpy(),
        "q25": errors.quantile(0.25, axis=0).reindex(models_index).to_numpy(),
        "q50": errors.quantile(0.50, axis=0).reindex(models_index).to_numpy(),
        "q75": errors.quantile(0.75, axis=0).reindex(models_index).to_numpy(),
        "q95": errors.quantile(0.95, axis=0).reindex(models_index).to_numpy(),
    }

    # Make LaTeX table
    table = r"""\begin{table}[h!]
        \begin{center}
        \resizebox{\columnwidth}{!}{
        \begin{tabular}{l ccccccc}
        \toprule
        & & & \multicolumn{5}{c}{\textbf{Quantiles}} \\
        \cmidrule(lr){4-8}
        \textbf{No VIX} & \textbf{Mean} & & \textbf{5%} & \textbf{25%} & \textbf{50%} & \textbf{75%} & \textbf{95%} \\
        \midrule
    """

    # No VIX models
    for i, (k, v) in enumerate(models.items()):
        table += v + " & "
        for j, s in enumerate(results.keys()):
            if loss.short_name in ["ru", "rutc"]:
                table += (
                    f"\\textbf{{{results[s][i]:.3%}}}"
                    if i == results[s].argmax()
                    else f"{results[s][i]:.3%}"
                ) + (
                    " & & "
                    if j == 0
                    else ("" if j == len(results.keys()) - 1 else " & ")
                )
            else:
                table += (
                    f"\\textbf{{{results[s][i]:.3f}}}"
                    if i == results[s].argmin()
                    else f"{results[s][i]:.3f}"
                ) + (
                    " & & "
                    if j == 0
                    else ("" if j == len(results.keys()) - 1 else " & ")
                )
        if i == len(models) - 1:
            table += r"\\[0.5em]" + "\n"
        else:
            table += r"\\" + "\n"

    # VIX models
    table += r"""    \textbf{VIX} \\
        \midrule
    """

    for i, (k, v) in enumerate(models_vix.items()):
        i = len(models_novix) + i
        table += v + " & "
        for j, s in enumerate(results.keys()):
            if loss.short_name in ["ru", "rutc"]:
                table += (
                    f"\\textbf{{{results[s][i]:.3%}}}"
                    if i == results[s].argmax()
                    else f"{results[s][i]:.3%}"
                ) + (
                    " & & "
                    if j == 0
                    else ("" if j == len(results.keys()) - 1 else " & ")
                )
            else:
                table += (
                    f"\\textbf{{{results[s][i]:.3f}}}"
                    if i == results[s].argmin()
                    else f"{results[s][i]:.3f}"
                ) + (
                    " & & "
                    if j == 0
                    else ("" if j == len(results.keys()) - 1 else " & ")
                )
        if i == len(models) - 1:
            table += r"\\[0.5em]" + "\n"
        else:
            table += r"\\" + "\n"

    table += (
        r"""    \bottomrule
        \end{tabular}}
        \caption{Descriptive statistics for the """
        + loss.name
        + """ of the different models without and with VIX. The table shows """
        + loss.name
        + """ calculated during the test period from January 2022 to November 2023, averaged on a per-stock basis """
        + (
            "(out of the full sample of \\fullstocks \\ stocks)."
            if universe == "all"
            else (
                "(within the DJIA constituents)."
                if universe == "dow30"
                else "(within the NASDAQ-100 constituents)."
            )
        )
        + r""" The lowest value in each column is highlighted in bold.}
        \label{tbl:"""
        + loss.short_name
        + "_"
        + universe
        + (kernel if kernel != "rv05" else "")
        + r"""}
        \end{center}
    \end{table}"""
    )
    table = table.replace("%", "\\%")

    with open(
        f"tables/{loss.short_name}_{universe}{kernel if kernel != 'rv05' else ''}.tex",
        "w",
    ) as f:
        f.write(table)


# Helper for MCS table
def make_mcs_table(
    df: pd.DataFrame, models: dict, mse: Loss, qlike: Loss, universe: str, kernel: str
) -> pd.DataFrame:
    # Make model index
    models_novix = {rename_model(k, False): v for k, v in models.items()}
    models_vix = {rename_model(k, True): v for k, v in models.items()}
    models_index = list(models_novix.keys()) + list(models_vix.keys())

    # Compute prediction errors
    errors_mse = df.drop(columns=["y_true"]).apply(lambda s: mse.fn(s, df["y_true"]))
    errors_qlike = df.drop(columns=["y_true"]).apply(
        lambda s: qlike.fn(s, df["y_true"])
    )

    # Compute MCS for each loss function
    all_results_mse = []
    all_results_qlike = []

    for symbol, dfg in tqdm(errors_mse.groupby(level="symbol")):
        mcs_mse = MCS(dfg, n_boot=5_000, alpha=0.05, show_progress=False)
        results_mse = mcs_mse.results()[["status"]]
        results_mse["status"] = np.where(results_mse["status"] == "excluded", 0, 1)
        results_mse.columns = [symbol]
        all_results_mse.append(results_mse)

    all_results_mse = pd.concat(all_results_mse, axis=1)
    all_results_mse = all_results_mse.mean(axis=1).reindex(models_index)
    all_results_mse.to_pickle(f"results/mcs/mse_{kernel}_{universe}.pkl")

    for symbol, dfg in tqdm(errors_qlike.groupby(level="symbol")):
        mcs_qlike = MCS(dfg, n_boot=5_000, alpha=0.05, show_progress=False)
        results_qlike = mcs_qlike.results()[["status"]]
        results_qlike["status"] = np.where(results_qlike["status"] == "excluded", 0, 1)
        results_qlike.columns = [symbol]
        all_results_qlike.append(results_qlike)

    all_results_qlike = pd.concat(all_results_qlike, axis=1)
    all_results_qlike = all_results_qlike.mean(axis=1).reindex(models_index)
    all_results_qlike.to_pickle(f"results/mcs/qlike_{kernel}_{universe}.pkl")

    table = (
        r"""\begin{table}[h!]
        \begin{center}
        \begin{tabular}{l cc c cc}
        \toprule
        & \multicolumn{2}{c}{\textbf{MSE}} & & \multicolumn{2}{c}{\textbf{QLIKE}} \\
        \cmidrule{2-3} \cmidrule{5-6}
        & \textbf{No VIX} & \textbf{VIX} & & \textbf{No VIX} & \textbf{VIX} \\
        \midrule """
        + "\n"
    )

    mse_nv = all_results_mse[models_novix.keys()]
    mse_v = all_results_mse[models_vix.keys()]
    qlike_nv = all_results_qlike[models_novix.keys()]
    qlike_v = all_results_qlike[models_vix.keys()]

    nstocks_universe = (
        "\\fullstocks"
        if universe == "all"
        else ("\\dowstocks" if universe == "dow30" else "\\nasdaqstocks")
    )

    def format_best(s: str):
        return "\\textbf{" + s + "}"

    for i, (k, v) in enumerate(models_novix.items()):
        table += v + " & "
        table += (
            format_best(f"{mse_nv.iloc[i]:.1%}")
            if mse_nv.iloc[i] == all_results_mse.max()
            else f"{mse_nv.iloc[i]:.1%}"
        ) + " & "
        table += (
            format_best(f"{mse_v.iloc[i]:.1%}")
            if mse_v.iloc[i] == all_results_mse.max()
            else f"{mse_v.iloc[i]:.1%}"
        ) + " & & "
        table += (
            format_best(f"{qlike_nv.iloc[i]:.1%}")
            if qlike_nv.iloc[i] == all_results_qlike.max()
            else f"{qlike_nv.iloc[i]:.1%}"
        ) + " & "
        table += (
            format_best(f"{qlike_v.iloc[i]:.1%}")
            if qlike_v.iloc[i] == all_results_qlike.max()
            else f"{qlike_v.iloc[i]:.1%}"
        ) + r" \\"
        table += "\n"

    table += (
        r"""    \bottomrule
        \end{tabular}
        \caption{Percentage of assets (out of the full sample of {"""
        + nstocks_universe
        + r"""} stocks) for which the model is part of the best model class according to the MCS procedure with a 95% confidence level.}
        \label{tbl:mcs_results_"""
        + universe
        + (kernel if kernel != "rv05" else "")
        + r"""}
        \end{center}
    \end{table}"""
    )

    table = table.replace("%", "\\%")

    with open(
        f"tables/mcs_results_{universe}{kernel if kernel != 'rv05' else ''}.tex", "w"
    ) as f:
        f.write(table)


# Obtain median spread
def get_spread() -> pd.DataFrame:
    spread = pd.read_pickle("data/bidask_spread.pkl")
    spread = spread.groupby(level=0).rolling(22 * 9).median()
    spread = spread.loc[spread.index.get_level_values("timestamp") >= "2022-01-04"]
    spread.index = spread.index.droplevel(0)
    return spread


qlike = Loss(name="QLIKE", fn=qlike_log, short_name="qlike")
mse = Loss(name="MSEs", fn=squared_error_log_scaled, short_name="mse")
ru = Loss(
    name="realized utilities (without transaction costs)",
    fn=realized_utility,
    short_name="ru",
)
rutc = Loss(
    name="realized utilities (with transaction costs)",
    fn=realized_utility,
    short_name="rutc",
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the configuration file", required=True
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    models = config["models"]

    models_novix = {rename_model(k, False): v for k, v in models.items()}
    models_vix = {rename_model(k, True): v for k, v in models.items()}
    models_index = list(models_novix.keys()) + list(models_vix.keys())

    for universe in config["index"]:
        for kernel in config["kernel"]:
            # Use same seed for everybody
            np.random.seed(72)
            print(f"Evaluating {universe} with {kernel} kernel")
            # Load data (forecasts and true values)
            df = get_forecasts(universe, kernel)

            make_loss_table(df, models, qlike, universe, kernel)
            make_loss_table(df, models, ru, universe, kernel)
            make_loss_table(df, models, mse, universe, kernel)
            make_loss_table(df, models, rutc, universe, kernel)

            make_mcs_table(df, models, mse, qlike, universe, kernel)
