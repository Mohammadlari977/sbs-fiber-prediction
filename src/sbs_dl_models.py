"""DL model definitions + portable loader for the SBS nanofiber study.
Models take 8 StandardScaler-transformed features; target is log(diameter_nm)."""
import json, os
import torch
import torch.nn as nn

INPUT_DIM = 8
CUSTOM_NETS = ("MLP", "CNN1D", "FT_Transformer")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class CNN1DRegressor(nn.Module):
    def __init__(self, input_dim, n_filters, kernel_size, fc_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(); self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters, fc_dim); self.fc2 = nn.Linear(fc_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(1); x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1); x = self.dropout(x)
        x = self.relu(self.fc1(x)); return self.fc2(x).squeeze(-1)

class FTTransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.feature_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(input_dim)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
              dim_feedforward=d_model*2, dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        b = x.size(0)
        t = [emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)]
        t = torch.stack(t, dim=1); cls = self.cls_token.expand(b, -1, -1)
        t = torch.cat([cls, t], dim=1); out = self.transformer(t)
        return self.head(out[:, 0, :]).squeeze(-1)

def rebuild(name, c):
    if name == "MLP":
        return MLPRegressor(INPUT_DIM, c["hidden_dims"], c["dropout"])
    if name == "CNN1D":
        return CNN1DRegressor(INPUT_DIM, c["n_filters"], c["kernel_size"], c["fc_dim"], c["dropout"])
    if name == "FT_Transformer":
        return FTTransformerRegressor(INPUT_DIM, c["d_model"], c["n_heads"], c["n_layers"], c["dropout"])
    raise ValueError(name)

def load_dl_models(export_dir):
    """Rebuild all 4 trained DL models from the portable export folder."""
    with open(os.path.join(export_dir, "dl_configs.json")) as f:
        configs = json.load(f)
    models = {}
    for name in CUSTOM_NETS:
        m = rebuild(name, configs[name])
        m.load_state_dict(torch.load(os.path.join(export_dir, f"{name}_state.pt"), map_location="cpu"))
        m.eval(); models[name] = m
    zip_path = os.path.join(export_dir, "TabNet.zip")
    if os.path.exists(zip_path):
        from pytorch_tabnet.tab_model import TabNetRegressor
        tn = TabNetRegressor(); tn.load_model(zip_path); models["TabNet"] = tn
    return models
