import torch.nn as nn


class AdvancedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, dropout: float = 0.0, aux_regression: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.aux_regression = aux_regression
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev_dim, 1)
        self.reg_head = nn.Linear(prev_dim, 1) if aux_regression else None

    def forward(self, x):
        h = self.backbone(x)
        logit = self.cls_head(h).squeeze(-1)
        if self.aux_regression:
            return logit, self.reg_head(h).squeeze(-1)
        return logit
