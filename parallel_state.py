import numpy as np
import torch.distributed as dist

_DP_SIZE = None # assume dp == decode_ep
_PREFILL_EP_SIZE = None

_WORLD_GROUP = None
_TP_GROUP = None
_PREFILL_EP_WORLD_GROUP = None
# disable dp group because collective perf is low due to topology
# _DP_GROUP = None


def initialize_model_parallel(tp_size: int, prefill_ep_size: int):
    assert dist.is_initialized()
    global _WORLD_GROUP, _TP_GROUP, _DP_SIZE, _PREFILL_EP_SIZE, _PREFILL_EP_WORLD_GROUP  # _DP_GROUP,
    _WORLD_GROUP = [list(range(0, dist.get_world_size()))]
    _TP_GROUP = np.arange(dist.get_world_size()).reshape(-1, tp_size).tolist()
    _DP_SIZE = dist.get_world_size() // tp_size
    # _DP_GROUP = np.arange(dist.get_world_size()).reshape(-1, tp_size)
    # # Fix for special case.
    # if (
    #     _DP_SIZE == 2
    #     and tp_size in {8, 32}
    #     and os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2"
    # ):
    #     # swap TP within DP1
    #     n_half_tp_rank = tp_size // 2
    #     _DP_GROUP[1] = np.concatenate([_DP_GROUP[1, n_half_tp_rank:], _DP_GROUP[1, :n_half_tp_rank]])
    # _DP_GROUP = _DP_GROUP.transpose().tolist()
    _PREFILL_EP_SIZE = prefill_ep_size
    _PREFILL_EP_WORLD_GROUP = (
        np.arange(dist.get_world_size()).reshape(-1, prefill_ep_size * tp_size).tolist()
    )

def get_world_group() -> list[list[int]]:
    assert dist.is_initialized()
    return _WORLD_GROUP

def get_tp_group() -> list[list[int]]:
    assert dist.is_initialized()
    return _TP_GROUP

# def get_dp_group() -> list[list[int]]:
#     assert dist.is_initialized()
#     return _DP_GROUP


def get_prefill_ep_world_group() -> list[list[int]]:
    assert dist.is_initialized()
    return _PREFILL_EP_WORLD_GROUP

def get_tp_size() -> int:
    assert dist.is_initialized()
    return len(_TP_GROUP[0])

def get_prefill_ep_size() -> int:
    assert dist.is_initialized()
    return _PREFILL_EP_SIZE

def get_decode_ep_size() -> int:
    assert dist.is_initialized()
    return _DP_SIZE

def get_dp_size() -> int:
    assert dist.is_initialized()
    return _DP_SIZE

def get_prefill_ep_rank() -> int:
    assert dist.is_initialized()
    return dist.get_rank() // get_tp_size() % get_prefill_ep_size()

def get_decode_ep_rank() -> int:
    assert dist.is_initialized()
    return dist.get_rank() // get_tp_size()

def get_tp_rank() -> int:
    assert dist.is_initialized()
    return dist.get_rank() % get_tp_size()