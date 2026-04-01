import torch

from dust3r.dual_state_factorization import StateFusion, TemporalFactorizer, ResidualUpdater
from dust3r.stable_propagation import TrendPropagator, apply_state_update
from dust3r.state_api import pack_state_args, to_persistent_state


def _make_state(batch=2, tokens=4, dim=8):
    trend_feat = torch.randn(batch, tokens, dim)
    residual_feat = torch.zeros_like(trend_feat)
    fused = trend_feat.clone()
    return pack_state_args(
        state_feat=fused,
        state_pos=torch.randn(batch, tokens, 2),
        init_state_feat=fused.clone(),
        mem=torch.randn(batch, 6, dim),
        init_mem=torch.randn(batch, 6, dim),
        trend_feat=trend_feat,
        residual_feat=residual_feat,
        trend_hidden=torch.zeros_like(fused),
        residual_hidden=torch.zeros_like(fused),
        aux={
            "init_trend_hidden": torch.zeros_like(fused),
            "init_residual_hidden": torch.zeros_like(fused),
            "init_trend_feat": trend_feat.clone(),
            "init_residual_feat": residual_feat.clone(),
        },
    )


def _dual_step(state_obj, use_stable_propagation):
    state_obj = to_persistent_state(state_obj)
    fusion = StateFusion(state_obj.state_feat.shape[-1])
    factorizer = TemporalFactorizer(state_obj.state_feat.shape[-1])
    residual_updater = ResidualUpdater(state_obj.state_feat.shape[-1])
    trend_propagator = TrendPropagator(state_obj.state_feat.shape[-1])

    fused = fusion(state_obj.trend_feat, state_obj.residual_feat)
    candidate = torch.randn_like(fused)
    delta = candidate - fused
    delta_tr, delta_res, alpha = factorizer(delta)
    assert delta.shape == delta_tr.shape == delta_res.shape
    assert alpha.shape[:2] == delta.shape[:2]

    update_mask = torch.ones(delta.shape[0], 1, 1)
    trend_candidate = state_obj.trend_feat + delta_tr
    trend_feat, trend_hidden = apply_state_update(
        state_feat=state_obj.trend_feat,
        candidate_state_feat=trend_candidate,
        init_state_feat=state_obj.aux["init_trend_feat"],
        update_mask=update_mask,
        reset_mask=torch.zeros_like(update_mask),
        use_stable_propagation=use_stable_propagation,
        trend_hidden=state_obj.trend_hidden,
        init_trend_hidden=state_obj.aux["init_trend_hidden"],
        propagator=trend_propagator if use_stable_propagation else None,
    )
    residual_feat, residual_hidden = residual_updater(
        residual_feat=state_obj.residual_feat,
        delta_res=delta_res,
        residual_hidden=state_obj.residual_hidden,
    )
    fused_next = fusion(trend_feat, residual_feat)
    return pack_state_args(
        state_feat=fused_next,
        state_pos=state_obj.state_pos,
        init_state_feat=state_obj.init_state_feat,
        mem=state_obj.mem,
        init_mem=state_obj.init_mem,
        trend_feat=trend_feat,
        residual_feat=residual_feat,
        trend_hidden=trend_hidden,
        residual_hidden=residual_hidden,
        aux=state_obj.aux,
    )


def test_dual_state_two_step_off_and_on():
    state_0 = _make_state()

    state_1_off = _dual_step(state_0, use_stable_propagation=False)
    state_2_off = _dual_step(state_1_off, use_stable_propagation=False)
    assert state_2_off.trend_feat is not None
    assert state_2_off.residual_feat is not None
    assert state_2_off.state_feat.shape == state_0.state_feat.shape

    state_1_on = _dual_step(state_0, use_stable_propagation=True)
    state_2_on = _dual_step(state_1_on, use_stable_propagation=True)
    assert state_2_on.trend_hidden is not None
    assert state_2_on.residual_hidden is not None
    assert state_2_on.state_feat.shape == state_0.state_feat.shape


def test_no_consolidation_fields_used():
    state = _make_state()
    state_next = _dual_step(state, use_stable_propagation=True)
    assert state_next.eligibility is None


if __name__ == "__main__":
    test_dual_state_two_step_off_and_on()
    test_no_consolidation_fields_used()
    print("dual_state_factorization_smoke: ok")
