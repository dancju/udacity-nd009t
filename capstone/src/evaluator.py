import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn
from matplotlib.dates import date2num

from .models import LinearModel, LSTMModel


def _read_csv(input_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(input_dir, filename), index_col="timestamp", parse_dates=True,
    )


def _load_model(model_dir: str, model_class: str) -> torch.nn.Module:
    archive = torch.load(os.path.join(model_dir, model_class + ".pth"))
    parameters = archive["parameters"]
    if model_class == "lstm":
        model = LSTMModel(**parameters)
    elif model_class == "linear":
        model = LinearModel(**parameters)
    else:
        raise Exception()
    model.load_state_dict(archive["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def _eval(
    seq_len: int, model: torch.nn.Module, x: pd.DataFrame, y: pd.DataFrame
) -> (pd.Series, pd.Series):
    loss_fn = torch.nn.MSELoss(reduction="none")
    with torch.no_grad():
        p0 = []
        for i in range(seq_len):
            batch_size_ = (x.shape[0] - i) // seq_len
            p_ = model(
                torch.tensor(
                    x.to_numpy()[i : i + batch_size_ * seq_len], dtype=torch.float,
                )
            )
            p0.append(p_.squeeze().numpy())
        p1 = pd.DataFrame(p0).transpose().to_numpy()
        p1 = pd.np.concatenate(p1)
        while pd.np.isnan(p1[-1]):
            p1 = p1[:-1]
        y = y[seq_len - 1 :]
        pred = pd.Series(p1, index=y.index)
        loss = pd.Series(
            loss_fn(torch.Tensor(y.squeeze().to_numpy()), torch.Tensor(p1)).numpy(),
            index=y.index,
        )
    return pred, loss


def evaluate(input_dir: str, model_dir: str, seq_len: int) -> None:
    pd.plotting.register_matplotlib_converters()
    datasets = ["trai", "vali", "test"]
    x = {}
    y = {}
    models = {}
    pred = {}
    loss = {}
    for d in datasets:
        x[d] = _read_csv(input_dir, f"{d}_x.csv")
        y[d] = _read_csv(input_dir, f"{d}_y.csv")
    for m in ["linear", "lstm"]:
        models[m] = _load_model(model_dir, m)
    for d in x:
        for m in models:
            pred[d, m], loss[d, m] = _eval(seq_len, models[m], x[d], y[d])
            print("%4s  %6s  %e" % (d, m, loss[d, m].mean()))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    axs[0, 0].plot(pd.concat([y[d] for d in datasets]))
    axs[0, 1].plot(pd.concat([y[d] for d in datasets]))
    axs[1, 0].plot([])
    axs[1, 1].plot([])
    axs[0, 0].plot(
        pd.concat([pred[d, "linear"] for d in datasets]),
        linestyle="",
        marker=".",
        markersize=0.1,
    )
    axs[1, 0].plot(
        pd.concat([loss[d, "linear"] for d in datasets]),
        linestyle="",
        marker=".",
        markersize=0.1,
    )
    axs[0, 1].plot(
        pd.concat([pred[d, "lstm"] for d in datasets]),
        linestyle="",
        marker=".",
        markersize=0.1,
    )
    axs[1, 1].plot(
        pd.concat([loss[d, "lstm"] for d in datasets]),
        linestyle="",
        marker=".",
        markersize=0.1,
    )
    axs[0, 0].axvline(y["vali"].index[0], linestyle=":", c="black")
    axs[0, 0].axvline(y["test"].index[0], linestyle=":", c="black")
    axs[1, 0].axvline(y["vali"].index[0], linestyle=":", c="black")
    axs[1, 0].axvline(y["test"].index[0], linestyle=":", c="black")
    axs[0, 1].axvline(y["vali"].index[0], linestyle=":", c="black")
    axs[0, 1].axvline(y["test"].index[0], linestyle=":", c="black")
    axs[1, 1].axvline(y["vali"].index[0], linestyle=":", c="black")
    axs[1, 1].axvline(y["test"].index[0], linestyle=":", c="black")
    axs[1, 0].set_xlim(axs[0, 0].get_xlim())
    axs[1, 1].set_xlim(axs[0, 1].get_xlim())
    axs[1, 0].set_ylim(0, 0.35)
    axs[1, 1].set_ylim(0, 0.35)
    axs[0, 1].legend(["truth", "prediction"])
    axs[1, 0].text(
        date2num(y["trai"].index[y["trai"].shape[0] // 2]),
        0.3,
        "training",
        horizontalalignment="center",
    )
    axs[1, 0].text(
        date2num(y["vali"].index[y["vali"].shape[0] // 2]),
        0.3,
        "validation",
        horizontalalignment="center",
    )
    axs[1, 0].text(
        date2num(y["test"].index[y["test"].shape[0] // 2]),
        0.3,
        "testing",
        horizontalalignment="center",
    )
    axs[1, 1].text(
        date2num(y["trai"].index[y["trai"].shape[0] // 2]),
        0.3,
        "training",
        horizontalalignment="center",
    )
    axs[1, 1].text(
        date2num(y["vali"].index[y["vali"].shape[0] // 2]),
        0.3,
        "validation",
        horizontalalignment="center",
    )
    axs[1, 1].text(
        date2num(y["test"].index[y["test"].shape[0] // 2]),
        0.3,
        "testing",
        horizontalalignment="center",
    )
    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[1, 0].set_xlabel("time")
    axs[1, 1].set_xlabel("time")
    axs[0, 0].set_ylabel("normalised price")
    axs[1, 0].set_ylabel("loss")
    axs[0, 0].set_title("Linear Model")
    axs[0, 1].set_title("LSTM Model")
    fig.subplots_adjust(hspace=0)
