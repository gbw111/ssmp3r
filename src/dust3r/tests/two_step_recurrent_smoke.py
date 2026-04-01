import torch

from dust3r.state_api import pack_state_args, to_persistent_state
from dust3r.stable_propagation import TrendPropagator, apply_state_update


def _make_initial_state(batch=2, tokens=4, dim=8):
    state_feat = torch.randn(batch, tokens, dim)
    state_pos = torch.randn(batch, tokens, 2)
    init_state_feat = state_feat.clone()
    mem = torch.randn(batch, 6, dim)
    init_mem = mem.clone()
    return pack_state_args(
        state_feat=state_feat,
        state_pos=state_pos,
        init_state_feat=init_state_feat,
        mem=mem,
        init_mem=init_mem,
        trend_hidden=None,
        aux={"init_trend_hidden": None},
    )


def _run_two_step(use_stable_propagation: bool):
    state = _make_initial_state()
    state = to_persistent_state(state)
    propagator = TrendPropagator(state.state_feat.shape[-1]) if use_stable_propagation else None

    # Step 1
    candidate_1 = torch.randn_like(state.state_feat)
    update_mask_1 = torch.ones(state.state_feat.shape[0], 1, 1)
    reset_mask_1 = torch.zeros_like(update_mask_1)
    state_feat_1, trend_hidden_1 = apply_state_update(
        state_feat=state.state_feat,
        candidate_state_feat=candidate_1,
        init_state_feat=state.init_state_feat,
        update_mask=update_mask_1,
        reset_mask=reset_mask_1,
        use_stable_propagation=use_stable_propagation,
        trend_hidden=state.trend_hidden,
        init_trend_hidden=(state.aux or {}).get("init_trend_hidden"),
        propagator=propagator,
    )
    state_step_1 = pack_state_args(
        state_feat=state_feat_1,
        state_pos=state.state_pos,
        init_state_feat=state.init_state_feat,
        mem=state.mem,
        init_mem=state.init_mem,
        trend_hidden=trend_hidden_1,
        aux={"init_trend_hidden": torch.zeros_like(state_feat_1) if use_stable_propagation else None},
    )

    # Step 2 (reuse state from Step 1)
    candidate_2 = torch.randn_like(state_step_1.state_feat)
    update_mask_2 = torch.ones_like(update_mask_1)
    reset_mask_2 = torch.zeros_like(update_mask_1)
    state_feat_2, trend_hidden_2 = apply_state_update(
        state_feat=state_step_1.state_feat,
        candidate_state_feat=candidate_2,
        init_state_feat=state_step_1.init_state_feat,
        update_mask=update_mask_2,
        reset_mask=reset_mask_2,
        use_stable_propagation=use_stable_propagation,
        trend_hidden=state_step_1.trend_hidden,
        init_trend_hidden=(state_step_1.aux or {}).get("init_trend_hidden"),
        propagator=propagator,
    )

    return state, state_step_1, state_feat_2, trend_hidden_2


def test_two_step_recurrent_off():
    state_0, state_1, state_feat_2, trend_hidden_2 = _run_two_step(
        use_stable_propagation=False
    )
    assert state_1.trend_hidden is None
    assert trend_hidden_2 is None
    assert state_feat_2.shape == state_0.state_feat.shape


def test_two_step_recurrent_on():
    state_0, state_1, state_feat_2, trend_hidden_2 = _run_two_step(
        use_stable_propagation=True
    )
    assert state_1.trend_hidden is not None
    assert trend_hidden_2 is not None
    assert trend_hidden_2.shape == state_0.state_feat.shape
    # Hidden is reused and evolves across recurrent steps.
    assert not torch.allclose(trend_hidden_2, state_1.trend_hidden)
    assert state_feat_2.shape == state_0.state_feat.shape


if __name__ == "__main__":
    test_two_step_recurrent_off()
    test_two_step_recurrent_on()
    print("two_step_recurrent_smoke: ok")
