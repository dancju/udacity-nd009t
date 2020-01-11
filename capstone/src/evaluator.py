import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import date2num


def _read_csv(input_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(input_dir, filename),
        index_col="timestamp",
        parse_dates=True,
        squeeze=True,
    )


def _plot_prediction(
    output_dir: str, datasets: list, models: list, y: dict, pred: dict,
) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
    axs[0].set_title("Linear Model")
    axs[1].set_title("LSTM Model")
    for m, ax in zip(models, axs.flat):
        ax.plot(pd.concat([y[d] for d in datasets]))
        ax.plot(pd.concat([pred[d, m] for d in datasets]))
        ax.axvline(y["vali"].index[0], linestyle=":", c="black")
        ax.axvline(y["test"].index[0], linestyle=":", c="black")
        ax.text(
            date2num(y["trai"].index[y["trai"].shape[0] // 2]),
            0.9 * ax.get_ylim()[1],
            "training",
            horizontalalignment="center",
        )
        ax.text(
            date2num(y["vali"].index[y["vali"].shape[0] // 2]),
            0.9 * ax.get_ylim()[1],
            "validation",
            horizontalalignment="center",
        )
        ax.text(
            date2num(y["test"].index[y["test"].shape[0] // 2]),
            0.9 * ax.get_ylim()[1],
            "testing",
            horizontalalignment="center",
        )
        ax.legend(["truth", "prediction"])
        ax.set_xlabel("time")
        ax.set_xlabel("time")
    axs[0].set_ylabel("log(vwap)")
    axs[1].set_ylim(axs[0].get_ylim())
    plt.savefig(os.path.join(output_dir, "evaluate_prediction.png"))


def _plot_loss(
    output_dir: str, datasets: list, models: list, y: dict, loss: dict
) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
    axs[0].set_title("linear model")
    axs[1].set_title("LSTM model")
    for m, ax in zip(models, axs.flat):
        loss_ = pd.concat([loss[d, m] for d in datasets])
        ax.plot(loss_)
        ax.set_xlabel("time")
        ax.axvline(y["vali"].index[0], linestyle=":", c="black")
        ax.axvline(y["test"].index[0], linestyle=":", c="black")
        ax.text(
            date2num(y["trai"].index[y["trai"].shape[0] // 2]),
            0.9 * ax.get_ylim()[1],
            "training",
            horizontalalignment="center",
        )
        ax.text(
            date2num(y["vali"].index[y["vali"].shape[0] // 2]),
            0.9 * ax.get_ylim()[1],
            "validation",
            horizontalalignment="center",
        )
        ax.text(
            date2num(y["test"].index[y["test"].shape[0] // 2]),
            0.9 * ax.get_ylim()[1],
            "testing",
            horizontalalignment="center",
        )
    axs[0].set_ylabel("MSE")
    axs[1].set_ylim(axs[0].get_ylim())
    fig.savefig(os.path.join(output_dir, "evaluate_loss.png"))


def _plot_scatter(
    output_dir: str, datasets: list, models: list, y: dict, pred: dict
) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
    axs[0].set_title("linear model")
    axs[1].set_title("LSTM model")
    for m, ax in zip(models, axs.flat):
        sns.regplot(y["test"], pred["test", m], ax=ax, marker=".", color="tab:blue")
        ax.set_xlabel("truth")
    axs[0].set_ylabel("prediction")
    axs[1].set_ylabel("")
    axs[1].set_xlim(axs[0].get_xlim())
    axs[1].set_ylim(axs[0].get_ylim())
    plt.savefig(os.path.join(output_dir, "evaluate_scatter.png"))


def _inverse_transformer(scaler):
    return lambda y: y * scaler.std_ + scaler.mean_


def evaluate(input_dir: str, output_dir: str) -> None:
    pd.plotting.register_matplotlib_converters()
    datasets = ["trai", "vali", "test"]
    models = ["linear", "lstm"]
    vwap_log = {}
    y = {}
    pred = {}
    loss = {}
    scaler = pickle.load(open(os.path.join(input_dir, "scaler.pkl"), "rb"))
    for d in datasets:
        vwap_log[d] = _read_csv(input_dir, f"{d}_x.csv").vwap_log
        vwap_log[d] = vwap_log[d].apply(_inverse_transformer(scaler.loc["vwap_log"]))
        y[d] = _read_csv(input_dir, f"{d}_y.csv")
        y[d] = y[d].apply(_inverse_transformer(scaler.loc["vwap_log_diff"]))
        # y[d] = (vwap_log[d] + y[d]).apply(exp)
    for m in models:
        for d in datasets:
            pred[d, m] = _read_csv(output_dir, f"pred_{d}_{m}.csv")
            pred[d, m] = pred[d, m].apply(
                _inverse_transformer(scaler.loc["vwap_log_diff"])
            )
            # pred[d, m] = (vwap_log[d] + pred[d, m]).apply(exp)
            loss[d, m] = _read_csv(output_dir, f"loss_{d}_{m}.csv")
    # _plot_prediction(output_dir, datasets, models, y, pred)
    _plot_loss(output_dir, datasets, models, y, loss)
    _plot_scatter(output_dir, datasets, models, y, pred)
