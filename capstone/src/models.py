import torch
import torch.nn


class LinearModel(torch.nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, output_dim: int):
        super(LinearModel, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden = torch.nn.Linear(seq_len * n_features, hidden_dim)
        self.predict = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.view(-1, self.seq_len * self.n_features)
        inputs = torch.nn.functional.relu(self.hidden(inputs))
        inputs = self.predict(inputs)
        return inputs


class LSTMModel(torch.nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, output_dim: int):
        super(LSTMModel, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.lstm = torch.nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor):
        lstm_out, _ = self.lstm(inputs.view(-1, self.seq_len, self.n_features))
        output = self.linear(lstm_out[:, -1, :])
        return output
