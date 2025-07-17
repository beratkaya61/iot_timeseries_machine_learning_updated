# 🧠 Step 4: Define LSTM Model
import torch.nn as nn

# class LSTMForecast(nn.Module):
#     def __init__(self, hidden_size=64):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         output, _ = self.lstm(x)
#         return self.fc(output[:, -1, :])


class LSTMForecast(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])