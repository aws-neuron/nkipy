"""Tensor-parallel helpers for the Qwen-Image MMDiT.

The 20B model (~40 GB bf16) does not fit on one trn2 core (24 GB), so the
denoiser is sharded Megatron-style across ``tp_size`` cores:

* **Attention** shards by heads: each core owns ``num_heads // tp_size`` heads
  and holds the corresponding slices of q/k/v (column-parallel, output dim) and
  the output projection (row-parallel, input dim). Heads are independent over the
  full sequence, so joint attention is unaffected; each core computes the partial
  contribution of its heads and the output projections are summed with an
  **all-reduce**.
* **MLP** shards by intermediate: ff0 column-parallel (output dim), ff2
  row-parallel (input dim), summed with an all-reduce after ff2.
* Modulation, LayerNorms, input/output projections, and residuals stay
  **replicated** — every core holds full-hidden activations after each
  all-reduce, so the elementwise/residual math is identical across ranks.

``all_reduce_fn`` abstracts the collective: the kernels always apply it after
each row-parallel projection. The device driver passes a real
``nkipy.distributed.collectives.all_reduce`` wrapper (tensor parallelism is
required — the 20B model does not fit on one core).
"""

import numpy as np


def make_all_reduce(tp_size):
    """Return an all-reduce callable summing across the full replica group.

    Requires ``tp_size > 1`` (tensor parallelism is mandatory). Imported lazily
    so importing this module doesn't require the distributed runtime.
    """
    if tp_size is None or tp_size <= 1:
        raise ValueError("tensor parallelism is required (tp_size > 1)")

    import nkipy.distributed.collectives as cc
    import torch.distributed as dist

    def _all_reduce(x):
        return cc.all_reduce(
            x, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
        )

    return _all_reduce


def shard_heads(weight, bias, rank, tp_size, n_heads, head_dim, axis):
    """Slice a head-partitioned projection weight for ``rank``.

    ``axis`` is the head-carrying axis: the output dim for q/k/v (column-parallel,
    weight (hidden_in, hidden_out)) or the input dim for the output projection
    (row-parallel, weight (hidden_in, hidden_out)). Biases are sharded only for
    column-parallel projections (axis == 1); row-parallel biases are added once
    on the full output and stay replicated.
    """
    local_heads = n_heads // tp_size
    start = rank * local_heads * head_dim
    end = start + local_heads * head_dim

    if axis == 1:  # column-parallel: shard output dim (and bias)
        w = weight[:, start:end]
        b = None if bias is None else bias[start:end]
    else:  # row-parallel: shard input dim; bias stays full (replicated)
        w = weight[start:end, :]
        b = bias
    return np.ascontiguousarray(w), (None if b is None else np.ascontiguousarray(b))


def shard_intermediate(weight, bias, rank, tp_size, axis):
    """Slice an MLP weight along the intermediate dim for ``rank``.

    ``axis == 1`` (ff0, column-parallel): shard output/intermediate dim + bias.
    ``axis == 0`` (ff2, row-parallel): shard input/intermediate dim; bias full.
    """
    inter = weight.shape[axis]
    local = inter // tp_size
    start, end = rank * local, (rank + 1) * local
    if axis == 1:
        w = weight[:, start:end]
        b = None if bias is None else bias[start:end]
    else:
        w = weight[start:end, :]
        b = bias
    return np.ascontiguousarray(w), (None if b is None else np.ascontiguousarray(b))
