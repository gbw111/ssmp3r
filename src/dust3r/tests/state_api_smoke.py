import torch

from dust3r.state_api import PersistentState, pack_state_args, unpack_state_args


def _make_core_state():
    state_feat = torch.randn(2, 4, 8)
    state_pos = torch.randn(2, 4, 2)
    init_state_feat = state_feat.clone()
    mem = torch.randn(2, 6, 8)
    init_mem = mem.clone()
    return state_feat, state_pos, init_state_feat, mem, init_mem


def test_new_persistent_state_path():
    core = _make_core_state()
    state = pack_state_args(*core, trend_feat=core[0], residual_feat=torch.zeros_like(core[0]))
    unpacked = unpack_state_args(state)
    for x, y in zip(core, unpacked):
        assert torch.allclose(x, y), "Pack -> unpack mismatch on PersistentState path"


def test_legacy_tuple_compatibility():
    core = _make_core_state()
    unpacked = unpack_state_args(core)
    for x, y in zip(core, unpacked):
        assert torch.allclose(x, y), "Legacy tuple compatibility failed"


def test_return_and_reuse_step():
    core = _make_core_state()
    history = [pack_state_args(*core)]
    latest = history[-1]
    state_feat, state_pos, init_state_feat, mem, init_mem = unpack_state_args(latest)
    state_feat = state_feat + 1.0
    next_state = pack_state_args(state_feat, state_pos, init_state_feat, mem, init_mem)
    assert isinstance(next_state, PersistentState)
    assert torch.allclose(next_state.state_feat, core[0] + 1.0)


if __name__ == "__main__":
    test_new_persistent_state_path()
    test_legacy_tuple_compatibility()
    test_return_and_reuse_step()
    print("state_api_smoke: ok")
