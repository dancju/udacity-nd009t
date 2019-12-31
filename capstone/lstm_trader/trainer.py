import argparse
import os
import time

import pandas as pd
import torch
import torch.nn
import torch.optim

try:
    from .model import Model
except ImportError:
    from model import Model


def train(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    checkpoint_path: str,
    hidden_dim: int,
    n_epochs: int,
    seq_len: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    def read_csv(filename):
        return torch.tensor(
            pd.read_csv(
                os.path.join(input_dir, filename),
                index_col="timestamp",
                parse_dates=True,
            ).values,
            dtype=torch.float,
            device=device,
        )

    x_trai = read_csv("x_trai.csv")
    x_vali = read_csv("x_vali.csv")
    y_trai = read_csv("y_trai.csv")
    y_vali = read_csv("y_vali.csv")

    parameters = {
        "input_dim": x_trai.shape[1],
        "hidden_dim": hidden_dim,
        "output_dim": 1,
    }
    model = Model(**parameters)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Checkpoint loaded.")
    model.to(device)

    train_losses = []
    valid_losses = []
    print("| epoch | trai loss | vali loss | time/s |")
    print("| ----: | --------- | --------- | -----: |", flush=True)
    for epoch in range(n_epochs):
        start_time = time.time()
        # train
        train_loss = 0
        for i in range(seq_len):
            for j in range(i, len(x_trai) - seq_len + 1, seq_len * batch_size):
                optimizer.zero_grad()
                batch_size_ = min(batch_size, (len(x_trai) - j) // seq_len)
                loss = loss_fn(
                    model(
                        x_trai[j : j + seq_len * batch_size_].view(
                            batch_size_, seq_len, parameters["input_dim"]
                        )
                    ),
                    y_trai[j + seq_len - 1 : j + seq_len * batch_size_ : seq_len],
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        train_loss = train_loss / (len(x_trai) - seq_len + 1)
        # validate
        with torch.no_grad():
            valid_loss = 0
            for i in range(seq_len):
                for j in range(i, len(x_vali) - seq_len + 1, seq_len * batch_size):
                    batch_size_ = min(batch_size, (len(x_vali) - j) // seq_len)
                    loss = loss_fn(
                        model(
                            x_vali[j : j + seq_len * batch_size_].view(
                                batch_size_, seq_len, parameters["input_dim"]
                            )
                        ),
                        y_vali[j + seq_len - 1 : j + seq_len * batch_size_ : seq_len],
                    )
                    valid_loss += loss.item()
            valid_loss = valid_loss / (len(x_vali) - seq_len + 1)
        # step
        print(
            "| %5d | %.3e | %.3e | %6d |"
            % (epoch, train_loss, valid_loss, time.time() - start_time),
            flush=True,
        )
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        scheduler.step()

    pd.DataFrame({"train": train_losses, "valid": valid_losses}).to_csv(
        os.path.join(output_dir, "loss.csv"), index=False
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        os.path.join(output_dir, "checkpoint.pth"),
    )
    torch.save(parameters, os.path.join(model_dir, "model_info.pth"))
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--input-data-dir", type=str, default=os.environ.get("SM_CHANNEL_DATA_DIR")
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.environ.get("SM_CHANNEL_CHECKPOINT_PATH"),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    train(
        input_dir=args.input_data_dir,
        output_dir=args.output_data_dir,
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint_path,
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
