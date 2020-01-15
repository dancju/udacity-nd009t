import argparse
import os
import time

import pandas as pd
import torch
import torch.nn
import torch.optim


class Model(torch.nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, output_dim: int):
        super(Model, self).__init__()
        self.n_features = n_features
        self.hidden = torch.nn.Linear(n_features, hidden_dim)
        self.predict = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.n_features)
        inputs = torch.sigmoid(self.hidden(inputs))
        inputs = self.predict(inputs)
        return inputs


def train(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    n_epochs: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    def _read_csv(filename: str) -> torch.Tensor:
        return pd.read_csv(
            os.path.join(input_dir, filename),
            dtype=pd.np.float32,
            index_col="timestamp",
            parse_dates=True,
        )

    def _load_data(dataset: str):
        x = _read_csv(dataset + "_x.csv")
        y = _read_csv(dataset + "_y.csv")
        vwap_next = y.vwap
        return (
            torch.tensor(x.to_numpy(), device=device),
            torch.tensor(vwap_next.to_numpy(), device=device),
            y.index,
        )

    trai_x, trai_next, trai_idx = _load_data("trai")
    vali_x, vali_next, vali_idx = _load_data("vali")
    test_x, test_next, test_idx = _load_data("test")

    parameters = {
        "n_features": trai_x.shape[1],
        "hidden_dim": hidden_dim,
        "output_dim": 1,
    }
    model = Model(**parameters).to(device)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)

    # training
    print("| epoch | trai loss | vali loss | time/s |")
    print("| ----: | --------- | --------- | -----: |", flush=True)
    for epoch in range(n_epochs):
        start_time = time.time()
        # train
        train_loss = 0
        for i in range(0, trai_x.shape[0], batch_size):
            optimizer.zero_grad()
            pred = model(trai_x[i : i + batch_size])
            loss = loss_fn(pred.squeeze(), trai_next[i : i + batch_size])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # validate
        with torch.no_grad():
            valid_loss = 0
            for i in range(0, vali_x.shape[0], batch_size):
                pred = model(vali_x[i : i + batch_size])
                loss = loss_fn(pred.squeeze(), vali_next[i : i + batch_size])
                valid_loss += loss.item()
        # step
        train_loss = train_loss / trai_x.shape[0]
        valid_loss = valid_loss / vali_x.shape[0]
        print(
            "| %5d | %.3e | %.3e | %6d |"
            % (epoch, train_loss, valid_loss, time.time() - start_time),
            flush=True,
        )
        scheduler.step()

    # evaluating

    def _evaluate(x, vwap_trut, idx, dataset):
        with torch.no_grad():
            pred = model(x)
        pred = pd.Series(pred.squeeze().cpu(), index=idx)
        pred.to_csv(os.path.join(output_dir, f"pred_{dataset}_linear.csv"), header=True)

    _evaluate(trai_x, trai_next, trai_idx, "trai")
    _evaluate(vali_x, vali_next, vali_idx, "vali")
    _evaluate(test_x, test_next, test_idx, "test")
    torch.save(
        {"parameters": parameters, "state_dict": model.cpu().state_dict()},
        os.path.join(model_dir, "linear.pth"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
