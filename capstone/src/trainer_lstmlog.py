import argparse
import os
import pickle
import time

import pandas as pd
import torch
import torch.nn
import torch.optim


class Model(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        output_dim: int,
        lstm_layers: int,
        dropout: float,
    ):
        super(Model, self).__init__()
        self.n_features = n_features
        self.lstm = torch.nn.LSTM(
            n_features, hidden_dim, num_layers=lstm_layers, dropout=dropout
        )
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(
        self, inputs: torch.Tensor, states: (torch.Tensor, torch.Tensor)
    ) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        lstm_out, states = self.lstm(inputs.view(-1, 1, self.n_features), states)
        output = self.linear(lstm_out[:, -1, :])
        states = (states[0].detach(), states[1].detach())
        return output, states


def train(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    hidden_dim: int,
    lstm_layers: int,
    dropout: float,
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
        vl_this = x.vwap_log
        vl_diff = y.vwap_log_diff
        return (
            torch.tensor(x.to_numpy(), device=device),
            torch.tensor(vl_diff.to_numpy(), device=device),
            vl_this,
            y.index,
        )

    trai_x, trai_vl_diff, trai_vl_this, trai_idx = _load_data("trai")
    vali_x, vali_vl_diff, vali_vl_this, vali_idx = _load_data("vali")
    test_x, test_vl_diff, test_vl_this, test_idx = _load_data("test")

    parameters = {
        "n_features": trai_x.shape[1],
        "hidden_dim": hidden_dim,
        "output_dim": 1,
        "lstm_layers": lstm_layers,
        "dropout": dropout,
    }
    model = Model(**parameters).to(device)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)

    # training
    print("| epoch | trai loss | vali loss | time/s |")
    print("| ----: | --------- | --------- | -----: |", flush=True)
    for epoch in range(n_epochs):
        start_time = time.time()
        # train
        train_loss = 0
        state = (
            torch.zeros([lstm_layers, 1, hidden_dim], device=device),
            torch.zeros([lstm_layers, 1, hidden_dim], device=device),
        )
        for i in range(0, trai_x.shape[0], batch_size):
            optimizer.zero_grad()
            pred, state = model(trai_x[i : i + batch_size], state)
            loss = loss_fn(pred.squeeze(), trai_vl_diff[i : i + batch_size])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # validate
        with torch.no_grad():
            valid_loss = 0
            for i in range(0, vali_x.shape[0], batch_size):
                pred, state = model(vali_x[i : i + batch_size], state)
                loss = loss_fn(pred.squeeze(), vali_vl_diff[i : i + batch_size])
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
    scaler = pickle.load(open(os.path.join(input_dir, "scaler.pkl"), "rb"))

    def _transformer(scaler):
        return lambda x: (x - scaler.mean_) / scaler.std_

    def _inverse_transformer(scaler):
        return lambda y: y * scaler.std_ + scaler.mean_

    def _evaluate(x, vl_this, idx, dataset, state):
        with torch.no_grad():
            vl_diff, state = model(x, state)
        vl_diff = pd.Series(vl_diff.squeeze().cpu(), index=idx)
        vl_diff = vl_diff.apply(_inverse_transformer(scaler.loc["vwap_log_diff"]))
        vl_this = vl_this.apply(_inverse_transformer(scaler.loc["vwap_log"]))
        pred = vl_this + vl_diff
        pred = pred.apply(pd.np.exp)
        pred = pred.apply(_transformer(scaler.loc["vwap"]))
        pred.to_csv(
            os.path.join(output_dir, f"pred_{dataset}_lstmlog.csv"), header=True
        )
        return state

    state = (
        torch.zeros([lstm_layers, 1, hidden_dim], device=device),
        torch.zeros([lstm_layers, 1, hidden_dim], device=device),
    )
    state = _evaluate(trai_x, trai_vl_this, trai_idx, "trai", state)
    state = _evaluate(vali_x, vali_vl_this, vali_idx, "vali", state)
    state = _evaluate(test_x, test_vl_this, test_idx, "test", state)
    torch.save(
        {"parameters": parameters, "state_dict": model.cpu().state_dict()},
        os.path.join(model_dir, "lstm.pth"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--input-data-dir", type=str, default=os.environ["SM_CHANNEL_DATA_DIR"]
    )
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--lstm-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    args = parser.parse_args()
    train(
        input_dir=args.input_data_dir,
        output_dir=args.output_data_dir,
        model_dir=args.model_dir,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )
