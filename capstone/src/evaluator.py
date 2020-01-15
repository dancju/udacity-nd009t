import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import date2num


def _read_csv(input_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(input_dir, filename),
        dtype=pd.np.float32,
        index_col="timestamp",
        parse_dates=True,
        squeeze=True,
    )


def _plot_dataset(datasets, trut, volu, path):
    fig, axs = plt.subplots(nrows=2, gridspec_kw={"hspace": 0})
    axs[1].set_xlabel("time")
    axs[0].set_ylabel("normalised VWAP")
    axs[1].set_ylabel("volume")
    axs[0].set_xticks([])
    axs[0].plot(pd.concat([trut[d] for d in datasets]))
    axs[1].plot(pd.concat([volu[d] for d in datasets]))
    axs[0].axvline(trut["vali"].index[0], linestyle=":", c="black")
    axs[0].axvline(trut["test"].index[0], linestyle=":", c="black")
    axs[1].axvline(trut["vali"].index[0], linestyle=":", c="black")
    axs[1].axvline(trut["test"].index[0], linestyle=":", c="black")
    axs[0].text(
        date2num(trut["trai"].index[trut["trai"].shape[0] // 2]),
        0.9 * axs[0].get_ylim()[1],
        "training",
        horizontalalignment="center",
    )
    axs[0].text(
        date2num(trut["vali"].index[trut["vali"].shape[0] // 2]),
        0.9 * axs[0].get_ylim()[1],
        "validation",
        horizontalalignment="center",
    )
    axs[0].text(
        date2num(trut["test"].index[trut["test"].shape[0] // 2]),
        0.9 * axs[0].get_ylim()[1],
        "testing",
        horizontalalignment="center",
    )
    plt.savefig(path)


def _plot_prediction(datasets: list, m: str, y: dict, pred: dict, path: str) -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("time")
    ax.set_ylabel("normalised VWAP")
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
    ax.legend(["truth", "prediction"], loc="lower left")
    plt.savefig(path)


def _plot_loss(datasets: list, m: str, y: dict, loss: dict, path: str) -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("time")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, 7e-1)
    _loss = pd.concat([loss[d, m] for d in datasets])
    ax.fill_between(_loss.index, _loss)
    ax.axvline(y["vali"].index[0], linestyle=":", c="black")
    ax.axvline(y["test"].index[0], linestyle=":", c="black")
    ax.text(
        date2num(y["trai"].index[y["trai"].shape[0] // 2]),
        0.9 * ax.get_ylim()[1],
        f"training\n{loss['trai', m].mean():.3e}",
        horizontalalignment="center",
    )
    ax.text(
        date2num(y["vali"].index[y["vali"].shape[0] // 2]),
        0.9 * ax.get_ylim()[1],
        f"validation\n{loss['vali', m].mean():.3e}",
        horizontalalignment="center",
    )
    ax.text(
        date2num(y["test"].index[y["test"].shape[0] // 2]),
        0.9 * ax.get_ylim()[1],
        f"testing\n{loss['test', m].mean():.3e}",
        horizontalalignment="center",
    )
    fig.savefig(path)


def evaluate(input_dir: str, output_dir: str) -> None:
    pd.plotting.register_matplotlib_converters()
    plt.ioff()
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(5.5, 4.4))
    plt.rc("savefig", bbox="tight", dpi=256)
    datasets = ["trai", "vali", "test"]
    models = ["linear", "lstm", "lstmratio"]
    trut = {}
    volu = {}
    pred = {}
    loss = {}
    scaler = pickle.load(open(os.path.join(input_dir, "scaler.pkl"), "rb"))
    for d in datasets:
        trut[d] = _read_csv(input_dir, f"{d}_y.csv").vwap
        volu[d] = _read_csv(input_dir, f"{d}_x.csv").home_notional
        for m in models:
            pred[d, m] = _read_csv(output_dir, f"pred_{d}_{m}.csv")
            loss[d, m] = (scaler.at["vwap", "std_"] * (pred[d, m] - trut[d])) ** 2
            loss[d, m] = pd.Series(loss[d, m], index=trut[d].index)
    for m in models:
        for d in datasets:
            print("%9s %4s %.9f" % (m, d, pd.np.sqrt(loss[d, m].mean())))
    _plot_dataset(datasets, trut, volu, os.path.join(output_dir, f"dataset.png"))
    for m in models:
        _plot_prediction(
            datasets, m, trut, pred, os.path.join(output_dir, f"pred_{m}.png"),
        )
        _plot_loss(
            datasets, m, trut, loss, os.path.join(output_dir, f"loss_{m}.png"),
        )
