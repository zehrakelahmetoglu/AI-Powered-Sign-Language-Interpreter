import torch
import torch.nn as nn


class SignLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_classes: int = 226,
        lstm_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.attn_fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        B, T, F = x.shape
        # frame-wise batch norm
        x = self.bn_input(x.reshape(B * T, F)).reshape(B, T, F)
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)
        # attention pooling over time
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)  # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)               # (B, hidden*2)
        out = self.dropout1(context)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        return self.fc2(out)
