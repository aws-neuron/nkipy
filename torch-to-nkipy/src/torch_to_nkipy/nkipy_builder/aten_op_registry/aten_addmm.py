# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.addmm.default")
def addmm_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle addmm operation (matrix multiplication with addition).

    Implements: out = beta * input + alpha * (mat1 @ mat2)
    For shapes:
        input: [out_features]
        mat1: [batch, in_features]
        mat2: [in_features, out_features]
        output: [batch, out_features]

    Args:
        node: The FX node representing the addmm operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0].name  # [out_features]
    mat1 = node.args[1].name  # [batch, in_features]
    mat2 = node.args[2].name  # [in_features, out_features]

    # Get beta and alpha from kwargs or use defaults (beta=1, alpha=1)
    beta = node.kwargs.get("beta", 1)
    alpha = node.kwargs.get("alpha", 1)

    # Get dtype for proper precision
    dtype = get_dtype_from_fx_node(node.args[0])
    numpy_dtype = torch_to_numpy_dtype_str(dtype)

    temp_vars = TempVarGenerator(node.name)

    # Step 1: Compute mat1 @ mat2 -> [batch, out_features]
    matmul_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=matmul_var,
        func_name="matmul",
        args=[mat1, mat2],
        kwargs={"dtype": numpy_dtype},
    )

    # Step 2: Scale the matmul result by alpha
    if alpha != 1:
        alpha_scaled = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=alpha_scaled,
            func_name="multiply",
            args=[matmul_var, str(alpha)],
            kwargs={"dtype": numpy_dtype},
        )
        current_matmul = alpha_scaled
    else:
        current_matmul = matmul_var

    # Step 3: Scale the input by beta
    if beta != 1:
        beta_scaled = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=beta_scaled,
            func_name="multiply",
            args=[input_tensor, str(beta)],
            kwargs={"dtype": numpy_dtype},
        )
        current_input = beta_scaled
    else:
        current_input = input_tensor

    # Final step: Add the scaled results with automatic broadcasting
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="add",
        args=[current_matmul, current_input],
        kwargs={"dtype": numpy_dtype},
    )
