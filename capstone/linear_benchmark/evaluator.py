import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn

try:
    from .model import model_fn
except ImportError:
    from model import model_fn


def _load_data(input_dir: str, model_dir: str, seq_len: int, dataset: str):
    def read_csv(filename):
        return pd.read_csv(
            os.path.join(input_dir, filename), index_col="timestamp", parse_dates=True,
        )

    x = read_csv(f"x_{dataset}.csv")
    y = read_csv(f"y_{dataset}.csv")
    model = model_fn(model_dir)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        p0 = []
        for i in range(seq_len):
            batch_size_ = (x.shape[0] - i) // seq_len
            p_ = model(
                torch.tensor(
                    x.to_numpy()[i : i + batch_size_ * seq_len], dtype=torch.float,
                ).view(batch_size_, seq_len * x.shape[1])
            )
            p0.append(p_.squeeze().numpy())
        p1 = pd.DataFrame(p0).transpose().to_numpy()
        p1 = pd.np.concatenate(p1)
        while pd.np.isnan(p1[-1]):
            p1 = p1[:-1]
        y = y[seq_len - 1 :]
        pred = pd.Series(p1, index=y.index)
        loss = loss_fn(torch.Tensor(y.squeeze().to_numpy()), torch.Tensor(p1)).item()
    print("loss(%s) = %e" % (dataset, loss))
    return y, pred, loss


def _plot_learning_curve(output_dir: str) -> None:
    losses = pd.read_csv(os.path.join(output_dir, "loss.csv"))
    plt.plot(losses)
    plt.legend(["training", "validation"])
    plt.title("Learning Curve")
    plt.show()


def _plot_scatter(y_trai, p_trai, y_vali, p_vali, y_test, p_test, seq_len: int) -> None:
    plt.scatter(y_trai, p_trai, 0.1)
    plt.scatter(y_vali, p_vali, 0.1)
    plt.scatter(y_test, p_test, 0.1)
    plt.legend(["training", "validation", "testing"])
    plt.xlabel("truth")
    plt.ylabel("prediction")
    plt.show()


def _plot_line(y_trai, p_trai, y_vali, p_vali, y_test, p_test, seq_len: int) -> None:
    plt.plot(pd.concat([y_trai, y_vali, y_test]))
    plt.plot(
        pd.concat([p_trai, p_vali, p_test]), linestyle="", marker=".", markersize=0.5,
    )
    plt.legend(["truth", "prediction"])
    plt.axvline(y_vali.index[0], linestyle=":", c="black")
    plt.axvline(y_test.index[0], linestyle=":", c="black")
    plt.xlabel("time")
    plt.ylabel("scaled price")
    plt.show()


def evaluate(input_dir: str, output_dir: str, model_dir: str, seq_len: int) -> None:
    pd.plotting.register_matplotlib_converters()
    y_trai, p_trai, l_trai = _load_data(input_dir, model_dir, seq_len, "trai")
    y_vali, p_vali, l_vali = _load_data(input_dir, model_dir, seq_len, "vali")
    y_test, p_test, l_test = _load_data(input_dir, model_dir, seq_len, "test")
    _plot_learning_curve(output_dir)
    _plot_scatter(y_trai, p_trai, y_vali, p_vali, y_test, p_test, seq_len)
    _plot_line(y_trai, p_trai, y_vali, p_vali, y_test, p_test, seq_len)
