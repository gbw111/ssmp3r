import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendPropagator(nn.Module):
    """Lightweight contraction-style stable propagator for trend-like state."""

    def __init__(self, dim: int):
        super().__init__()
        self.ctrl = nn.Linear(dim * 3, dim * 3)
        self.delta_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, state_feat, candidate_state_feat, trend_hidden):
        delta = candidate_state_feat - state_feat
        ctrl = self.ctrl(torch.cat([state_feat, candidate_state_feat, delta], dim=-1))
        a, b, c = torch.chunk(ctrl, 3, dim=-1)
        a = torch.exp(-F.softplus(a))
        b = torch.sigmoid(b)
        c = torch.sigmoid(c)
        trend_hidden = a * trend_hidden + b * self.delta_proj(delta)
        trend_feat = state_feat + c * self.out_proj(trend_hidden)
        return trend_feat, trend_hidden


def apply_state_update(
    state_feat,
    candidate_state_feat,
    init_state_feat,
    update_mask,
    reset_mask,
    use_stable_propagation=False,
    trend_hidden=None,
    init_trend_hidden=None,
    propagator=None,
):
    """Apply baseline update or stable propagation update with identical masking contract."""

    if not use_stable_propagation:
        state_feat = candidate_state_feat * update_mask + state_feat * (1 - update_mask)
        if reset_mask is not None:
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
        return state_feat, trend_hidden

    if propagator is None:
        raise ValueError("propagator is required when use_stable_propagation=True")

    if trend_hidden is None:
        trend_hidden = torch.zeros_like(state_feat)
    if init_trend_hidden is None:
        init_trend_hidden = torch.zeros_like(state_feat)

    prev_trend_hidden = trend_hidden
    propagated_state, trend_hidden = propagator(
        state_feat=state_feat,
        candidate_state_feat=candidate_state_feat,
        trend_hidden=trend_hidden,
    )

    state_feat = propagated_state * update_mask + state_feat * (1 - update_mask)
    trend_hidden = trend_hidden * update_mask + prev_trend_hidden * (1 - update_mask)

    if reset_mask is not None:
        state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
        trend_hidden = init_trend_hidden * reset_mask + trend_hidden * (1 - reset_mask)

    return state_feat, trend_hidden
