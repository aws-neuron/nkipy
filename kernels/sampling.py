import numpy as np
from collective import all_gather

# Import config from parent directory
from config import Config
from nkipy.core import tensor_apis
from parallel_state import get_tp_group

# Import kernels from the kernels directory
from .rmsnorm import rmsnorm


def greedy_sampling(
    hidden_states_shard,
    norm_weight,
    lm_head_weight,
    last_token_indices,
    configs: Config,
    is_neuronpy: bool,
):
    """Greedy sampling kernel for token generation."""
    # TODO: sequence parallel
    hidden_states_shard = rmsnorm(
        hidden_states_shard, norm_weight, configs.norm_eps, is_neuronpy=is_neuronpy
    )
    batch_size = configs.max_batch_size_per_dp
    hidden_size = configs.hidden_size
    is_prefill = last_token_indices is not None
    if is_prefill:
        hidden_states = all_gather(
            hidden_states_shard,
            all_gather_dim=0,
            replica_groups=get_tp_group(),
            is_neuronpy=is_neuronpy,
        )

        max_model_len = configs.max_model_len
        assert hidden_states.shape == (batch_size * max_model_len, hidden_size)
        # XXX: if np.take_along_axis gets OOB, check if some batch slot is
        # empty, i.e. last_token_indices >= 0
        hidden_states = hidden_states.reshape((batch_size, max_model_len, hidden_size))
        hidden_states = np.take_along_axis(
            hidden_states,
            last_token_indices.reshape((batch_size, 1, 1)),
            axis=1,
        )
        hidden_states = hidden_states.reshape((batch_size, -1))
    else:
        # tokengen
        # layout [batch, seq, hidden]
        assert len(hidden_states_shard.shape) == 3
        hidden_states = hidden_states_shard[:, 0, :]
        assert hidden_states.shape == (batch_size, hidden_size)
    logits = hidden_states @ lm_head_weight

    logits, next_id = tensor_apis.topk(logits, k=1, axis=1)
    logits_all = all_gather(
        logits,
        all_gather_dim=1,
        replica_groups=get_tp_group(),
        is_neuronpy=is_neuronpy,
    )
    next_id_all = all_gather(
        next_id,
        all_gather_dim=1,
        replica_groups=get_tp_group(),
        is_neuronpy=is_neuronpy,
    )

    _, top_index = tensor_apis.topk(logits_all, k=1, axis=1)
    final_next_id = np.empty_like(next_id, dtype=np.uint32)

    # N.B.: make scalar uint32 explicitly
    vocab_per_device = np.uint32(lm_head_weight.shape[1])
    for b in range(configs.max_batch_size_per_dp):
        device_idx = np.copy(top_index[b])
        local_idx = next_id_all[b, device_idx]
        global_idx = device_idx * vocab_per_device + local_idx
        final_next_id[b] = global_idx

    return final_next_id
