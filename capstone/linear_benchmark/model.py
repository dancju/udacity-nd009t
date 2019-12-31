import os

import torch
import torch.nn


class Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.predict = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.predict(x)
        return x


def model_fn(model_dir: str) -> Model:
    parameters = torch.load(os.path.join(model_dir, "model_info.pth"))
    model = Model(**parameters)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
