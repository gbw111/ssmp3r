import torch
import torch.nn as nn


class StateFusion(nn.Module):
    """Fused read helper for trend/residual states."""

    def __init__(self, dim: int):
        super().__init__()
        self.residual_weight = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, trend_feat, residual_feat):
        w = torch.sigmoid(self.residual_weight)
        return self.norm(trend_feat + w * residual_feat)


class TemporalFactorizer(nn.Module):
    """Token-wise gate to split candidate delta into trend/residual parts."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, delta):
        alpha = torch.sigmoid(self.gate(delta))
        delta_tr = alpha * delta
        delta_res = (1.0 - alpha) * delta
        return delta_tr, delta_res, alpha


class ResidualUpdater(nn.Module):
    """Lightweight residual branch updater for fast corrective dynamics."""

    def __init__(self, dim: int):
        super().__init__()
        self.ctrl = nn.Linear(dim * 2, dim * 2)
        self.delta_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, residual_feat, delta_res, residual_hidden):
        if residual_hidden is None:
            residual_hidden = torch.zeros_like(residual_feat)
        ctrl = self.ctrl(torch.cat([residual_feat, delta_res], dim=-1))
        g, lam = torch.chunk(ctrl, 2, dim=-1)
        g = torch.sigmoid(g)
        lam = torch.sigmoid(lam)
        residual_hidden = lam * residual_hidden + self.delta_proj(delta_res)
        residual_feat = g * residual_feat + self.out_proj(residual_hidden)
        return residual_feat, residual_hidden
