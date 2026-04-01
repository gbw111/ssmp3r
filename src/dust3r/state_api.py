from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class PersistentState:
    """Structured persistent state container for recurrent CUT3R flow.

    Core fields keep full backward compatibility with baseline state passing.
    Optional fields are Phase-1/2 ready but inactive in Phase 0.
    """

    state_feat: Any
    state_pos: Any
    init_state_feat: Any
    mem: Any
    init_mem: Any
    trend_feat: Any = None
    residual_feat: Any = None
    trend_hidden: Any = None
    residual_hidden: Any = None
    eligibility: Any = None
    aux: Optional[Dict[str, Any]] = field(default_factory=dict)

    def as_legacy_tuple(self) -> Tuple[Any, Any, Any, Any, Any]:
        return (
            self.state_feat,
            self.state_pos,
            self.init_state_feat,
            self.mem,
            self.init_mem,
        )


def pack_state_args(
    state_feat: Any,
    state_pos: Any,
    init_state_feat: Any,
    mem: Any,
    init_mem: Any,
    **kwargs: Any,
) -> PersistentState:
    return PersistentState(
        state_feat=state_feat,
        state_pos=state_pos,
        init_state_feat=init_state_feat,
        mem=mem,
        init_mem=init_mem,
        **kwargs,
    )


def unpack_state_args(state_args: Any) -> Tuple[Any, Any, Any, Any, Any]:
    return to_persistent_state(state_args).as_legacy_tuple()


def to_persistent_state(state_args: Any) -> PersistentState:
    if isinstance(state_args, PersistentState):
        return state_args

    if isinstance(state_args, dict):
        optional_keys = [
            "trend_feat",
            "residual_feat",
            "trend_hidden",
            "residual_hidden",
            "eligibility",
            "aux",
        ]
        optional_values = {k: state_args.get(k) for k in optional_keys if k in state_args}
        return PersistentState(
            state_feat=state_args["state_feat"],
            state_pos=state_args["state_pos"],
            init_state_feat=state_args["init_state_feat"],
            mem=state_args["mem"],
            init_mem=state_args["init_mem"],
            **optional_values,
        )

    if isinstance(state_args, (tuple, list)) and len(state_args) == 5:
        return pack_state_args(*state_args)

    raise TypeError(
        "Unsupported state_args format. Expected PersistentState, dict, or 5-tuple/list."
    )
