import os

import torch
import torch.nn


class Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs: torch.Tensor):
        lstm_out, _ = self.lstm(inputs)
        output = self.linear(lstm_out[:, -1, :])
        return output


def model_fn(model_dir: str) -> Model:
    parameters = torch.load(os.path.join(model_dir, "model_info.pth"))
    model = Model(**parameters)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
