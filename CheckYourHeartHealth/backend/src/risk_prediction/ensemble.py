"""Stacking ensemble for heart risk: LightGBM + Random Forest + a small Transformer, combined by logistic regression.

Everything the app needs at prediction time lives here so training and serving use the same code.
"""
import numpy as np
import torch
import torch.nn as nn

# A shared demo host has a small CPU allowance. Avoid large thread pools per request.
torch.set_num_threads(1)

FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active", "bmi"]


class TabTransformer(nn.Module):
    """Each feature becomes one token (value * learned vector + learned bias), then a few self-attention layers."""

    def __init__(self, n_features=len(FEATURES), d_model=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, d_model) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=64, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.w + self.b
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.enc(tokens)[:, 0]).squeeze(-1)


def fit_transformer(X_tr, y_tr, X_va=None, y_va=None, epochs=14, seed=0):
    torch.manual_seed(seed)
    model = TabTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(X_tr) + 511) // 512))
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(np.asarray(y_tr), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad(); loss_fn(model(Xt[idx]), yt[idx]).backward(); opt.step(); sched.step()
    return model


def predict_transformer(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy()


class StackedRiskModel:
    """Loaded form of the trained ensemble. predict_proba(df) -> probability of heart disease."""

    def __init__(self, bundle):
        self.scaler = bundle["scaler"]
        self.lgb = bundle["lgb"]
        self.rf = bundle["rf"]
        self.lgb.set_params(n_jobs=1)
        self.rf.set_params(n_jobs=1)
        self.meta = bundle["meta"]
        self.transformer = TabTransformer()
        self.transformer.load_state_dict(bundle["transformer_state"])
        self.transformer.eval()
        self.features = bundle["features"]

    def base_probs(self, df):
        X = df[self.features]
        Xs = self.scaler.transform(X)
        return np.c_[self.lgb.predict_proba(X)[:, 1], self.rf.predict_proba(X)[:, 1],
                     predict_transformer(self.transformer, Xs)]

    def predict_proba(self, df):
        return self.meta.predict_proba(self.base_probs(df))[:, 1]
