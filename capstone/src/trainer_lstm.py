import argparse
import os
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
        df = pd.read_csv(
            os.path.join(input_dir, filename), index_col="timestamp", parse_dates=True,
        )
        return torch.tensor(df.to_numpy(), dtype=torch.float, device=device), df.index

    trai_x, trai_x_i = _read_csv("trai_x.csv")
    trai_y, trai_y_i = _read_csv("trai_y.csv")
    vali_x, vali_x_i = _read_csv("vali_x.csv")
    vali_y, vali_y_i = _read_csv("vali_y.csv")
    test_x, test_x_i = _read_csv("test_x.csv")
    test_y, test_y_i = _read_csv("test_y.csv")

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
            loss = loss_fn(pred, trai_y[i : i + batch_size])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # validate
        with torch.no_grad():
            valid_loss = 0
            for i in range(0, vali_x.shape[0], batch_size):
                pred, state = model(vali_x[i : i + batch_size], state)
                loss = loss_fn(pred, vali_y[i : i + batch_size])
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

    def _evaluate(x, y, index, dataset, state):
        loss_fn = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            pred, state = model(x, state)
            loss = loss_fn(pred, y)
            pred = pd.Series(pred.squeeze().cpu().numpy(), index=index)
            loss = pd.Series(loss.squeeze().cpu().numpy(), index=index)
        pred.to_csv(os.path.join(output_dir, f"pred_{dataset}_lstm.csv"), header=True)
        loss.to_csv(os.path.join(output_dir, f"loss_{dataset}_lstm.csv"), header=True)
        print(f"loss of {dataset} is {loss.mean():.3e}")
        return state

    state = (
        torch.zeros([lstm_layers, 1, hidden_dim], device=device),
        torch.zeros([lstm_layers, 1, hidden_dim], device=device),
    )
    state = _evaluate(trai_x, trai_y, trai_y_i, "trai", state)
    state = _evaluate(vali_x, vali_y, vali_y_i, "vali", state)
    state = _evaluate(test_x, test_y, test_y_i, "test", state)
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
