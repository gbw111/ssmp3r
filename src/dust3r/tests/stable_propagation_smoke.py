import torch

from dust3r.stable_propagation import TrendPropagator, apply_state_update


def _make_inputs():
    b, n, c = 2, 4, 8
    state_feat = torch.randn(b, n, c)
    candidate_state_feat = torch.randn(b, n, c)
    init_state_feat = state_feat.clone()
    update_mask = torch.ones(b, 1, 1)
    reset_mask = torch.zeros(b, 1, 1)
    return state_feat, candidate_state_feat, init_state_feat, update_mask, reset_mask


def test_off_path_matches_baseline_formula():
    state_feat, candidate_state_feat, init_state_feat, update_mask, reset_mask = (
        _make_inputs()
    )
    out_state, out_hidden = apply_state_update(
        state_feat=state_feat,
        candidate_state_feat=candidate_state_feat,
        init_state_feat=init_state_feat,
        update_mask=update_mask,
        reset_mask=reset_mask,
        use_stable_propagation=False,
        trend_hidden=None,
        init_trend_hidden=None,
        propagator=None,
    )
    expected = candidate_state_feat * update_mask + state_feat * (1 - update_mask)
    assert torch.allclose(out_state, expected)
    assert out_hidden is None


def test_on_path_builds_and_reuses_hidden():
    state_feat, candidate_state_feat, init_state_feat, update_mask, reset_mask = (
        _make_inputs()
    )
    propagator = TrendPropagator(state_feat.shape[-1])
    out_state_1, out_hidden_1 = apply_state_update(
        state_feat=state_feat,
        candidate_state_feat=candidate_state_feat,
        init_state_feat=init_state_feat,
        update_mask=update_mask,
        reset_mask=reset_mask,
        use_stable_propagation=True,
        trend_hidden=None,
        init_trend_hidden=None,
        propagator=propagator,
    )
    assert out_hidden_1 is not None
    assert out_hidden_1.shape == state_feat.shape

    out_state_2, out_hidden_2 = apply_state_update(
        state_feat=out_state_1,
        candidate_state_feat=candidate_state_feat,
        init_state_feat=init_state_feat,
        update_mask=update_mask,
        reset_mask=reset_mask,
        use_stable_propagation=True,
        trend_hidden=out_hidden_1,
        init_trend_hidden=torch.zeros_like(out_hidden_1),
        propagator=propagator,
    )
    assert out_state_2.shape == state_feat.shape
    assert out_hidden_2.shape == state_feat.shape


if __name__ == "__main__":
    test_off_path_matches_baseline_formula()
    test_on_path_builds_and_reuses_hidden()
    print("stable_propagation_smoke: ok")
