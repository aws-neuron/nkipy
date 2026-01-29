# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Callable, Dict, List, Optional

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import stringify_fx_node
from torch_to_nkipy.utils.nki import NKIOpRegistry

logger = logging.getLogger(__name__)


class TempVarGenerator:
    """
    Helper class to generate and track temporary variable names.

    This class creates sequential variable names for multi-step operations
    that require intermediate results. It automatically handles naming and
    tracking of temporary variables.
    """

    def __init__(self, base_name: str):
        """
        Initialize a temporary variable generator.

        Args:
            base_name: Base prefix for all temporary variables
        """
        self.base_name = base_name
        self.counter = 0
        self.vars: List[str] = []

    def next(self) -> str:
        """
        Generate next temporary variable name.

        Creates a new variable name with the pattern {base_name}_np_{counter}
        and increments the counter.

        Returns:
            The generated variable name
        """
        var_name = f"{self.base_name}_np_{self.counter}"
        self.vars.append(var_name)
        self.counter += 1
        return var_name

    def last(self) -> str:
        """
        Get the last generated variable name.

        Useful when chaining operations that need to refer to the previous result.

        Returns:
            The most recently generated variable name

        Raises:
            ValueError: If no variables have been generated yet
        """
        if not self.vars:
            raise ValueError("No variables have been generated yet")
        return self.vars[-1]

    def all(self) -> List[str]:
        """
        Get all generated variable names.

        Returns:
            List of all variable names created by this generator
        """
        return self.vars


# Operation handler registry with decorator pattern
class AtenOpRegistry:
    """
    Registry for ATen/PyTorch operation handlers.

    This class implements a registry pattern to map PyTorch operations to
    their corresponding NKIPy code generators. It provides decorator-based
    registration and a factory for creating simple operation handlers.
    """

    # Maps operation names to their handler functions
    _handlers: Dict[str, Callable[[fx.Node], ComputationNode]] = {}

    @classmethod
    def register(cls, op_name: str) -> Callable:
        """
        Decorator to register an operation handler.

        Usage:
            @AtenOpRegistry.register("torch.ops.aten.add.Tensor")
            def add_tensor_handler(node, computation_node):
                # Implementation...

        Args:
            op_name: The fully qualified name of the ATen operation

        Returns:
            A decorator function that registers the handler
        """

        def decorator(
            handler_func: Callable[[fx.Node, ComputationNode], None],
        ) -> Callable:
            @functools.wraps(handler_func)
            def wrapper(node: fx.Node) -> ComputationNode:
                # Create the computation node
                computation_node = ComputationNode(
                    name=node.name,
                    fx_node_str=stringify_fx_node(node),
                    fx_node_stack_trace=node.stack_trace,
                )

                # Call the actual handler
                handler_func(node, computation_node)

                return computation_node

            cls._handlers[op_name] = wrapper
            logger.debug(f"Registered handler for {op_name}")
            return handler_func

        return decorator

    @classmethod
    def get_handler(
        cls, op_name: str
    ) -> Optional[Callable[[fx.Node], ComputationNode]]:
        """
        Get handler for a given operation.

        Looks up the appropriate handler for an operation by its qualified name.

        Args:
            op_name: The fully qualified operation name

        Returns:
            The handler function if registered, None otherwise
        """
        # Try getting standard op handler first
        handler = cls._handlers.get(op_name)
        # If failing, check if it is a registered NKI custom op
        if handler is None:
            if NKIOpRegistry.is_registered(op_name, canonicalize=False):
                handler = cls._handlers.get("custom_nki_ops")
        if handler is None:
            logger.warning(f"No handler registered for operation: {op_name}")
        return handler

    @classmethod
    def create_simple_op_handler(cls, numpy_func_name: str) -> Callable:
        """
        Factory function that creates handlers for simple operations.

        Works for any operation that can be directly mapped to a NumPy function
        with the same argument structure. This eliminates the need to write
        separate handler functions for operations with identical implementation
        patterns.

        Args:
            numpy_func_name: The NumPy function name to use (e.g., "add", "multiply",
            "transpose")

        Returns:
            A handler function for the operation that generates appropriate AST code
        """

        def handler(node: fx.Node, computation_node: ComputationNode) -> None:
            if node.kwargs:
                raise NotImplementedError(
                    f"create_simple_op_handler cannot handle kwargs from node: {node}."
                )

            # Extract arguments from the node
            inputs = [str(arg) for arg in node.args]

            # Add operation code using AST
            # Maps directly to the corresponding NumPy function
            computation_node.ast_code_block.add_numpy_call_assignment(
                target=node.name, func_name=numpy_func_name, args=inputs
            )

            logger.debug(f"Generated {numpy_func_name} operation for node {node.name}")

        return handler

    @classmethod
    def batch_register(cls, op_mapping: Dict[str, str]) -> None:
        """
        Register multiple simple operations at once.

        Args:
            op_mapping: Dictionary mapping PyTorch op names to NumPy function names
        """
        for op_name, numpy_func in op_mapping.items():
            handler = cls.create_simple_op_handler(numpy_func)
            cls.register(op_name)(handler)
            logger.debug(f"Batch registered {op_name} → numpy.{numpy_func}")
