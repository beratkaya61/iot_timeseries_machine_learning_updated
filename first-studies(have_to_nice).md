Requirements Installation
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
pip install torch torchvision torchaudio
pip install transformers datasets accelerate



📁 Step 1: Dataset Preparation (ETT Dataset)

import pandas as pd
import numpy as np

# Load ETTh1 dataset
url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
df = pd.read_csv(url, parse_dates=['date'])

# Set datetime index
df.set_index('date', inplace=True)

# Preview
print(df.head())



🧼 Step 2: Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Handle missing values
df.interpolate(method='linear', inplace=True)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

# Anomaly detection (simple z-score method)
from scipy.stats import zscore
z_scores = np.abs(zscore(scaled_df))
anomalies = (z_scores > 3)

print(f"Total anomalies detected: {anomalies.values.sum()}")


🔮 Step 3: LSTM Forecasting

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window=48):
        self.window = window
        self.data = torch.tensor(series.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        return self.data[idx:idx+self.window], self.data[idx+self.window]

# Use only one feature for demo (e.g., "OT")
target_series = scaled_df["OT"]
train_dataset = TimeSeriesDataset(target_series, window=48)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.linear(out[:, -1, :])

model = LSTMForecast()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5):  # increase for better results
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")



🔁 Step 4: Transformer (PatchTST, Informer etc.)
For demonstration, we'll use the Informer from a known repo or Hugging Face. Due to complexity, I’ll provide the pipeline later if needed.


🤖 Step 5: LLM Integration via Prompting (Quantization + Reprogramming)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

def to_prompt(series):
    return "Predict next value given the series:\n" + ", ".join(f"{x:.2f}" for x in series) + "\nNext:"

sample_input = target_series[:50].tolist()
prompt = to_prompt(sample_input)
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs, max_length=inputs['input_ids'].shape[1]+1)
prediction = tokenizer.decode(output[0], skip_special_tokens=True)

print("LLM Output:", prediction)




⚙️ Step 6: Optimization Techniques
1.Model checkpoints & early stopping with PyTorch callbacks.

2.Hyperparameter tuning using optuna.

3.Quantization (for LLM):

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16)

4.Data augmentation: Add noise, reverse series, slice-windows for training.

5.Few-shot Prompt Tuning (LLM): Provide 2-3 series + answers, then new query.


📊 Step 7: Evaluation

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(true, pred):
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    smape = 100 * np.mean(2 * np.abs(true - pred) / (np.abs(true) + np.abs(pred)))
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, SMAPE: {smape:.2f}%")


📌 What's Next?
Would you like me to:

Add a real-time streaming demo with MQTT or Kafka?

Integrate a web UI for forecasting visualization?

Package everything into a Jupyter Notebook or GitHub repo format?