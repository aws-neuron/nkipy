# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.name import NUMPY_PKG
from torch_to_nkipy.utils.nki import (
    NKIOpRegistry,
    get_nki_kernel_hash,
    populate_grid,
)


@AtenOpRegistry.register("torch.ops.nkipy.mark_subgraph_identity.default")
def mark_subgraph_identity(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle custom op mark_subgraph_identity.
    """
    ast_block = computation_node.ast_code_block

    input_tensor = node.args[0].name
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(input_tensor)
    )


# NKI kernel handler, borrow the syntax here to isolate this function from the registry
@AtenOpRegistry.register("custom_nki_ops")
def nki_op_handler(node: fx.Node, computation_node: ComputationNode) -> None:
    custom_op_name = f"torch.ops.{node.target}"
    # Do a fake run on the kernel to get the grid
    populate_grid(node.target, node.args)
    wrapped_nki_kernel = NKIOpRegistry.get_nki_kernel(custom_op_name)
    nki_kernel = wrapped_nki_kernel.func
    grid = wrapped_nki_kernel.grid
    # Merge global default compiler args with per-op args
    default_args = NKIOpRegistry.get_default_compiler_args()
    per_op_args = wrapped_nki_kernel.compiler_args
    if default_args and per_op_args:
        compiler_args = f"{default_args}, {per_op_args}"
    else:
        compiler_args = default_args or per_op_args
    alias_map = wrapped_nki_kernel.alias_map
    opt_level = NKIOpRegistry.get_default_opt_level()
    ast_block = computation_node.ast_code_block
    # from nkipy.core.nki_op import NKICustomOp
    # We always import, this overhead should be minimal
    NKI_CUSTOM_OP_CLASS_NAME = "NKICustomOp"
    NKI_CUSTOM_OP_MODULE = "nkipy.core.nki_op"
    ast_block.add_from_import(
        module=NKI_CUSTOM_OP_MODULE, name=NKI_CUSTOM_OP_CLASS_NAME
    )
    # import the NKI kernel itself
    ast_block.add_from_import(
        module=str(nki_kernel.__module__), name=str(nki_kernel.__name__)
    )
    # Import OptLevel if opt_level is set
    if opt_level:
        ast_block.add_from_import(
            module="neuronxcc.nki.compiler.backends.neuron.CompileOpts",
            name="OptLevel",
        )
    # Add a line to create an NKICustomOp object with the NKI kernel
    arg_list = [
        (
            f"{NUMPY_PKG}.empty({tuple(arg.meta['val'].shape)}, "
            + f"{torch_to_numpy_dtype_str(arg.meta['val'].dtype)})"
            if isinstance(arg, fx.Node)
            else f"{arg}"
        )
        for arg in node.args
    ]
    hash_str = get_nki_kernel_hash(
        str(nki_kernel.__name__), ",".join(arg_list), f"{grid}",
        compiler_args + opt_level,
    )
    kernel_name = f"{nki_kernel.__name__}_kernel_{hash_str}"
    # Only add assignment if the kernel is new
    # This line does tracing under the hood when being processed by nkipy,
    # can be pretty slow if there's no caching
    if not NKIOpRegistry.is_processed(hash_str):
        nki_kwargs = {
            "grid": f"{grid}",
            "kernel_return": "True",
            "compiler_args": repr(compiler_args),
        }
        if opt_level:
            nki_kwargs["opt_level"] = opt_level
        ast_block.add_class_call_assignment(
            class_or_obj_name=NKI_CUSTOM_OP_CLASS_NAME,
            args=[
                nki_kernel.__name__,
                arg_list,
            ],
            kwargs=nki_kwargs,
            target_name=kernel_name,
        )
        NKIOpRegistry.add_processed_kernel_hash(hash_str)

    # Actually call the kernel. In case of IO aliasing, change the variable names
    if len(alias_map) == 0:
        nki_output_ast = CodeGenerator.name_store(node.name)
    else:
        ret_val = node.meta["val"]
        tuple_len = 1 if not isinstance(ret_val, tuple) else len(ret_val)
        assert len(alias_map) <= tuple_len, (
            "Cannot have more aliases than total output values!"
        )
        if tuple_len == 1:
            input_arg = node.args[alias_map[0]].name
            nki_output_ast = CodeGenerator.name_store(input_arg)
            computation_node.set_nki_aliased_input(input_arg)
        else:
            temp_vars = TempVarGenerator(node.name)
            all_nki_outputs = []
            # Set outputs to their aliased inputs, and annotate alias information
            for out_idx in range(tuple_len):
                if out_idx in alias_map:
                    input_arg = node.args[alias_map[out_idx]].name
                    all_nki_outputs.append(input_arg)
                    computation_node.set_nki_aliased_input(input_arg)
                else:
                    all_nki_outputs.append(temp_vars.next())
            all_nki_outputs = [
                CodeGenerator.name_store(name) for name in all_nki_outputs
            ]
            nki_output_ast = ast.Tuple(all_nki_outputs)

    kernel_call_args = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
    kernel_call_kwargs = {
        key: value for key, value in node.kwargs.items() if isinstance(value, fx.Node)
    }
    call_ast = CodeGenerator.call(
        kernel_name, "__call__", kernel_call_args, kernel_call_kwargs
    )
    ast_block.add_assignment(nki_output_ast, call_ast)
    # Add an additional assignment so that the rest of the graph can proceed normally
    ast_block.add_assignment(CodeGenerator.name_store(node.name), nki_output_ast)
