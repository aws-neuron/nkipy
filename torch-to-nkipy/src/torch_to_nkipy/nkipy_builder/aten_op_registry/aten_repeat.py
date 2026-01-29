# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node

# FIXME np.tile is not supported in NKIPy
# @AtenOpRegistry.register("torch.ops.aten.repeat.default")
# def repeat_default(node: fx.Node, computation_node: ComputationNode) -> None:
#    """
#    Handle tensor repeat operation.
#
#    Maps PyTorch's tensor.repeat() to NumPy's tile function.
#    PyTorch's repeat is equivalent to NumPy's tile, where each dimension
#    is repeated according to the specified number of times.
#
#    PyTorch signature: torch.repeat(input, *sizes) -> Tensor
#    NumPy equivalent: np.tile(input, reps)
#
#    Args:
#        node: The FX node representing the repeat operation
#        computation_node: The ComputationNode to add code to
#
#    Raises:
#        ValueError: If sizes argument is missing
#    """
#    ast_block = computation_node.ast_code_block
#
#    # Extract arguments
#    if len(node.args) < 2:
#        raise ValueError("repeat.default requires input tensor and sizes arguments")
#
#    input_tensor = node.args[0].name
#    # The repeat sizes can be either a tuple/list or multiple arguments
#    if isinstance(node.args[1], (list, tuple)):
#        repeat_sizes = node.args[1]
#    else:
#        repeat_sizes = node.args[1:]
#
#    # Add the tile operation
#    ast_block.add_numpy_call_assignment(
#        target=node.name,
#        func_name="tile",
#        args=[input_tensor, str(tuple(repeat_sizes))],
#    )


@AtenOpRegistry.register("torch.ops.aten.repeat.default")
def repeat_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor repeat operation without using np.tile.
    Maps PyTorch's tensor.repeat() to a sequence of NumPy's reshape and
    broadcast_to operations. This approach is often more compatible with
    backends that may not support np.tile directly.
    The logic is as follows:
    1. Pre-process shapes: If the number of repeat values (`reps`) is greater
       than the number of input dimensions, the input is reshaped to have
       leading dimensions of size 1.
    2. Expand: The tensor is reshaped to interleave a singleton dimension
       before each original dimension (e.g., shape [2, 3] -> [1, 2, 1, 3]).
    3. Broadcast: The expanded tensor is broadcast to a new shape where the
       singleton dimensions are replaced by the repeat values (e.g., shape
       [1, 2, 1, 3] broadcast to [r0, 2, r1, 3]).
    4. Reshape: The broadcasted tensor is reshaped to the final target
       shape (e.g., [r0*2, r1*3]).
    Args:
        node: The FX node representing the repeat operation
        computation_node: The ComputationNode to add code to
    Raises:
        ValueError: If sizes argument is missing or in an unexpected format.
    """
    ast_block = computation_node.ast_code_block
    # Extract arguments
    if len(node.args) < 2:
        raise ValueError("repeat.default requires input tensor and sizes arguments")
    input_node = node.args[0]
    input_tensor = input_node.name
    # The repeat sizes can be either a list/tuple or multiple arguments
    if isinstance(node.args[1], (list, tuple)):
        reps = list(node.args[1])
    else:
        # This case is less common in FX graphs but supported for robustness
        reps = [arg for arg in node.args[1:]]
    input_shape = list(get_shape_from_fx_node(input_node))
    temp_vars = TempVarGenerator(node.name)
    current_tensor = input_tensor
    # Step 1: Pre-process shapes for dimension mismatch
    if len(reps) > len(input_shape):
        num_new_dims = len(reps) - len(input_shape)
        new_shape = [1] * num_new_dims + input_shape
        padded_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=padded_var,
            func_name="reshape",
            args=[current_tensor, str(tuple(new_shape))],
        )
        current_tensor = padded_var
        input_shape = new_shape
    # Step 2: Expand tensor by interleaving singleton dimensions
    # Shape [s0, s1, ...] becomes [1, s0, 1, s1, ...]
    expanded_shape = []
    for dim in input_shape:
        expanded_shape.extend([1, dim])
    expanded_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=expanded_var,
        func_name="reshape",
        args=[current_tensor, str(tuple(expanded_shape))],
    )
    # Step 3: Broadcast to the intermediate tiled shape
    # Shape [1, s0, 1, s1, ...] is broadcast to [r0, s0, r1, s1, ...]
    broadcast_shape = []
    for i, dim in enumerate(input_shape):
        broadcast_shape.extend([reps[i], dim])
    broadcasted_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=broadcasted_var,
        func_name="broadcast_to",
        args=[expanded_var, str(tuple(broadcast_shape))],
    )
    # Step 4: Reshape to the final target shape
    # Shape [r0*s0, r1*s1, ...]
    final_shape = [reps[i] * s for i, s in enumerate(input_shape)]
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="reshape",
        args=[broadcasted_var, str(tuple(final_shape))],
    )
