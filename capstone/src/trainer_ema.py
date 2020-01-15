import os

import pandas as pd


def train(input_dir: str, output_dir: str) -> None:
    def _read_csv(filename: str):
        return pd.read_csv(
            os.path.join(input_dir, filename),
            dtype=pd.np.float32,
            index_col="timestamp",
            parse_dates=True,
        )

    trai_x = _read_csv("trai_x.csv")
    trai_y = _read_csv("trai_y.csv")
    vali_x = _read_csv("vali_x.csv")
    vali_y = _read_csv("vali_y.csv")
    test_x = _read_csv("test_x.csv")
    test_y = _read_csv("test_y.csv")

    def _evaluate(x, y, dataset):
        pred = x.vwap.ewm(span=10).mean()
        pred.to_csv(os.path.join(output_dir, f"pred_{dataset}_ema.csv"), header=True)

    _evaluate(trai_x, trai_y, "trai")
    _evaluate(vali_x, vali_y, "vali")
    _evaluate(test_x, test_y, "test")
