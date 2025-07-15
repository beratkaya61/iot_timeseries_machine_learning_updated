import torch
import torch.nn as nn
from einops import rearrange

# ---------------------- Informer Transformer Model Definition  --------------------
# ProbSparse Attention is a simplified version of the original Informer attention mechanism
# which uses a probabilistic approach to select a subset of keys and values for attention computation.
# This is a simplified version and may not include all the optimizations present in the original Informer paper.
# The Informer model is designed for long sequence forecasting tasks and uses a self-attention mechanism
# to capture long-term dependencies in the data. It also includes a probabilistic sparse attention mechanism
# to reduce the computational complexity of the attention mechanism.
# The Informer model is particularly useful for time series forecasting tasks where long-term dependencies
# are important, such as in financial forecasting, weather prediction, and other applications where
# time series data is prevalent.
class ProbSparseSelfAttention(nn.Module):
    """Approximate attention mechanism (ProbSparse Attention)"""
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.attention = ProbSparseSelfAttention(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b t (three h) -> three b t h', three=3)
        out = self.attention(q, k, v)
        out = self.out_proj(out)
        return self.norm(out + x)

class InformerForecast(nn.Module):
    def __init__(self, input_size=1, d_model=64, n_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.encoder = nn.ModuleList([
            InformerBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x[:, -1, :])  # predict based on last token


# -------------------------------- Autoformer Transformer Model Definition --------------------
# The Autoformer model is designed for time series forecasting tasks and uses a self-attention mechanism
# to capture long-term dependencies in the data. It also includes a seasonal decomposition module
# to separate the seasonal and trend components of the time series data. The Autoformer model is particularly
# useful for time series forecasting tasks where long-term dependencies are important, such as in financial
# forecasting, weather prediction, and other applications where time series data is prevalent.
# The Autoformer model is designed to be efficient and effective for time series forecasting tasks,
# and it has been shown to outperform other state-of-the-art models on various benchmark datasets.
# The Autoformer model is particularly useful for time series forecasting tasks where long-term dependencies
# are important, such as in financial forecasting, weather prediction, and other applications where
# time series data is prevalent.
class SeriesDecomposition(nn.Module):
    """Decomposes series into seasonal and trend parts"""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

class AutoformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        attn_out, _ = self.self_attn(seasonal, seasonal, seasonal)
        x = self.norm1(attn_out + seasonal)
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)
        return x + trend

class AutoformerForecast(nn.Module):
    def __init__(self, input_size=1, d_model=64, n_heads=4, num_layers=2, kernel_size=25):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.encoder = nn.ModuleList([
            AutoformerBlock(d_model, n_heads, kernel_size) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x[:, -1, :])


# ----------------------------------- PatchTST Transformer Model Definition --------------------
# The PatchTST model is a simplified version of the TST (Time Series Transformer) model,
# which uses a patch-based approach to process time series data. The PatchTST model is designed
# to be efficient and effective for time series forecasting tasks, and it has been shown to
# outperform other state-of-the-art models on various benchmark datasets. The PatchTST model
# is particularly useful for time series forecasting tasks where long-term dependencies are important,
# such as in financial forecasting, weather prediction, and other applications where time series data is prevalent.
# The PatchTST model uses a patch-based approach to process time series data, which allows it to capture
# long-term dependencies in the data while maintaining efficiency. The PatchTST model is designed to be
# efficient and effective for time series forecasting tasks, and it has been shown to outperform other
# state-of-the-art models on various benchmark datasets. The PatchTST model is particularly useful for
# time series forecasting tasks where long-term dependencies are important, such as in financial forecasting,
# weather prediction, and other applications where time series data is prevalent.
# The PatchTST model is designed to be efficient and effective for time series forecasting tasks,
# and it has been shown to outperform other state-of-the-art models on various benchmark datasets.
# The PatchTST model is particularly useful for time series forecasting tasks where long-term dependencies
# are important, such as in financial forecasting, weather prediction, and other applications where
# time series data is prevalent.
class PatchTST(nn.Module):
    def __init__(self, patch_size=8, d_model=64, n_heads=4, num_layers=2, num_patches=12):
        super().__init__()
        self.patch_embed = nn.Linear(patch_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, patches, patch_size, 1]
        x = x.squeeze(-1)   # [batch, patches, patch_size]
        x = self.patch_embed(x)  # [batch, patches, d_model]
        x = self.transformer(x)
        x = x.mean(dim=1)   # average over patches
        return self.head(x)