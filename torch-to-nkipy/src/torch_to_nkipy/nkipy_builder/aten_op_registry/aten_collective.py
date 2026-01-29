# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx
from torch._C._distributed_c10d import _resolve_process_group
from torch.distributed.distributed_c10d import get_process_group_ranks

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.name import CC_PKG, NUMPY_PKG


@AtenOpRegistry.register("torch.ops._c10d_functional.all_reduce.default")
def all_reduce_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle all_reduce collective communication operation.

    Maps PyTorch's all_reduce operation to the corresponding NKIPy collective
    communication operation.

    Args:
        node: The FX node representing the all_reduce operation
        computation_node: The ComputationNode to add code to

    Raises:
        NotImplementedError: If the reduce operation is not supported
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    tensor = node.args[0].name
    reduce_op = node.args[1]
    group_name = node.args[2]

    # Map reduce operation to NumPy function
    if reduce_op == "sum":
        reduce_func = f"{NUMPY_PKG}.add"
    else:
        raise NotImplementedError(f"Unsupported reduce operation: {reduce_op}")

    # Get process group ranks
    ranks = get_process_group_ranks(_resolve_process_group(group_name))

    # Add the all_reduce operation
    ast_block.add_call_assignment(
        target=node.name,
        pkg_or_obj=CC_PKG,
        func="all_reduce",
        args=[tensor],
        kwargs={"reduce_op": reduce_func, "replica_groups": f"[{ranks}]"},
    )


@AtenOpRegistry.register("torch.ops._c10d_functional.all_gather_into_tensor.default")
def all_gather_into_tensor_default(
    node: fx.Node, computation_node: ComputationNode
) -> None:
    """
    Handle all_gather_into_tensor collective communication operation.

    Maps PyTorch's all_gather_into_tensor operation to the corresponding NKIPy
    collective communication operation.

    Args:
        node: The FX node representing the all_reduce operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments. node.args[1] is world size and will not be used
    tensor = node.args[0].name
    group_name = node.args[2]

    # Get process group ranks
    ranks = get_process_group_ranks(_resolve_process_group(group_name))

    # Add the all_gather operation
    ast_block.add_call_assignment(
        target=node.name,
        pkg_or_obj=CC_PKG,
        func="all_gather",
        args=[tensor],
        kwargs={"all_gather_dim": "0", "replica_groups": f"[{ranks}]"},
    )


@AtenOpRegistry.register("torch.ops._c10d_functional.wait_tensor.default")
def wait_tensor_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle wait_tensor operation.

    Maps PyTorch's wait_tensor operation to a simple assignment that passes the input
    tensor through to the output. This operation typically blocks until the input tensor
    is ready, but in the static graph compilation context, we treat it as an identity.

    Args:
        node: The FX node representing the wait_tensor operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract the input tensor
    input_tensor = node.args[0].name

    # Add simple assignment operation (identity pass-through)
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(input_tensor)
    )
