# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
from ast import _Unparser
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from torch.fx.graph import _parse_stack_trace

from torch_to_nkipy.utils.name import CC_PKG, NUMPY_PKG

logger = logging.getLogger(__name__)


class Comment(ast.expr):
    """Custom AST node representing a source code comment."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

        # Required AST node fields
        self._fields = ("text",)


class Unparser(_Unparser):
    """Enhanced unparser that supports Comment nodes in the AST."""

    def visit_Comment(self, node: Comment) -> None:
        """Handle unparsing of Comment nodes."""
        self.fill("# " + node.text)

    @staticmethod
    def unparse(node: ast.AST) -> str:
        """Unparse an AST with support for Comment nodes.

        Args:
            node: The AST node to unparse

        Returns:
            String representation of the AST
        """
        return Unparser().visit(node)


class NodeType(Enum):
    """Types of nodes in the code structure."""

    INPUT = "input"
    COMPUTATION = "computation"
    OUTPUT = "output"
    ALIASED_OUTPUT = "aliased_output"


class CodeGenerator:
    """Helper class for generating AST nodes for common operations."""

    @staticmethod
    def preprocess_arg(arg):
        if isinstance(arg, str):
            # Handle empty string as a string literal
            if arg == "":
                return ast.Constant(value="")
            # Parse single string argument
            return ast.parse(arg).body[0].value
        elif isinstance(arg, list):
            # For list arguments, create a proper list node with parsed elements
            elements = []
            for item in arg:
                if item == "":
                    elements.append(ast.Constant(value=""))
                else:
                    elements.append(ast.parse(item).body[0].value)
            return ast.List(elts=elements, ctx=ast.Load())
        else:
            raise TypeError(
                f"Unsupported argument type: {type(arg)}.Expected str or List[str]."
            )

    @staticmethod
    def call(
        pkg_or_obj: str,
        func: str,
        args: List[Union[str, List[str]]],
        kwargs: Optional[Dict[str, str]] = None,
    ) -> ast.Call:
        """Create an AST node for a function or method call.

        Args:
            pkg_or_obj: Package name (e.g., "numpy") or object variable name (e.g., "x")
            func: Function or method name
            args: List of argument expressions as strings or lists of strings
            kwargs: Dict of keyword arguments as strings

        Returns:
            AST Call node

        Examples:
            call(NUMPY_PKG, "add", ["x", "y"])  # numpy.add(x, y)
            call("x", "astype", ["float32"])    # x.astype(float32)
        """
        kwargs = kwargs or {}

        try:
            # Process arguments based on their type
            processed_args = [CodeGenerator.preprocess_arg(arg) for arg in args]
            # Create function reference
            func_ref = ast.Attribute(
                value=CodeGenerator.name_load(pkg_or_obj),
                attr=func,
                ctx=ast.Load(),
            )

            return ast.Call(
                func=func_ref,
                args=processed_args,
                keywords=[
                    ast.keyword(arg=k, value=CodeGenerator.preprocess_arg(v))
                    for k, v in kwargs.items()
                ],
            )
        except SyntaxError as e:
            raise SyntaxError(f"Error parsing call to {pkg_or_obj}.{func}: {e}") from e

    @staticmethod
    def subscript(target: str, index_strs: List[str], ctx=ast.Load()) -> ast.Subscript:
        """Create an AST node for an indexed expression.

        Args:
            target: Target variable name
            index_strs: List of index expressions as strings
            ctx: Context for the subscript (ast.Store or ast.Load)

        Returns:
            AST Subscript node
        """

        def create_slice_expr(index_strs: List[str]) -> Union[ast.expr, ast.Tuple]:
            slice_items = []
            for index_str in index_strs:
                if index_str == "":
                    slice_items.append(ast.Slice(lower=None, upper=None, step=None))
                else:
                    slice_items.append(CodeGenerator.name_load(index_str))

            return (
                slice_items[0]
                if len(slice_items) == 1
                else ast.Tuple(elts=slice_items, ctx=ast.Load())
            )

        slice_expr = create_slice_expr(index_strs)
        return ast.Subscript(
            value=CodeGenerator.name_load(target), slice=slice_expr, ctx=ctx
        )

    @staticmethod
    def assignment(target: ast.AST, value: ast.AST) -> ast.Assign:
        """Create an AST node for an assignment statement."""
        return ast.Assign(targets=[target], value=value)

    @staticmethod
    def name_store(id: str) -> ast.Name:
        """Create an AST node for a variable in store context."""
        return ast.Name(id=id, ctx=ast.Store())

    @staticmethod
    def name_load(id: str) -> ast.Name:
        """Create an AST node for a variable in load context."""
        return ast.Name(id=id, ctx=ast.Load())

    @staticmethod
    def comment(text: str) -> Comment:
        """Create a Comment AST node."""
        return Comment(text)

    @staticmethod
    def from_import(module: str, name: str) -> ast.ImportFrom:
        """Create an AST node for "from X import Y"."""
        from_import_stmt = ast.ImportFrom(
            module=module, names=[ast.alias(name=name, asname=None)], level=0
        )
        return from_import_stmt

    @staticmethod
    def class_call(
        class_name: str,
        args: List[Union[str, List[str]]],
        kwargs: Optional[Dict[str, str]] = None,
    ) -> ast.Call:
        kwargs = kwargs or {}
        try:
            # Process arguments based on their type
            processed_args = [CodeGenerator.preprocess_arg(arg) for arg in args]
            processed_kwargs = [
                ast.keyword(arg=k, value=CodeGenerator.preprocess_arg(v))
                for k, v in kwargs.items()
            ]
            call_node = ast.Call(
                func=CodeGenerator.name_load(class_name),
                args=processed_args,  # Positional argument
                keywords=processed_kwargs,  # Keyword argument
            )
            return call_node
        except SyntaxError as e:
            raise SyntaxError(f"Error parsing arguments for {class_name}: {e}") from e


@dataclass
class ASTCodeBlock:
    """A code block represented as AST nodes."""

    statements: List[ast.AST] = field(default_factory=list)

    def add_statement(self, stmt: ast.AST) -> None:
        """Add a statement to the code block."""
        self.statements.append(stmt)

    def add_comment(self, text: str) -> None:
        """Add a comment to the code block."""
        self.statements.append(CodeGenerator.comment(text))

    def add_empty_line(self) -> None:
        """Add an empty line (as a comment) to improve readability."""
        self.add_comment("")

    def add_assignment(self, target: ast.AST, value: ast.AST) -> None:
        """Add an assignment statement to the code block."""
        self.statements.append(CodeGenerator.assignment(target, value))

    def add_call_assignment(
        self,
        target: str,
        pkg_or_obj: str,
        func: str,
        args: List[Union[str, List[str]]],
        kwargs: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add an assignment with a function or method call.

        Args:
            target: Target variable name
            pkg_or_obj: Package name (e.g., NUMPY_PKG) or object variable name
            func: Function or method name
            args: List of argument expressions
            kwargs: Dict of keyword arguments

        Examples:
            add_call_assignment("result", NUMPY_PKG, "add", ["x", "y"])
            add_call_assignment("result", "x", "astype", ["float32"])
        """
        call_ast = CodeGenerator.call(pkg_or_obj, func, args, kwargs)
        target_ast = CodeGenerator.name_store(target)
        self.add_assignment(target_ast, call_ast)

    def add_numpy_call_assignment(
        self,
        target: str,
        func_name: str,
        args: List[Union[str, List[str]]],
        kwargs: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add an assignment with a NumPy function call."""
        self.add_call_assignment(target, NUMPY_PKG, func_name, args, kwargs)

    def add_subscript_assignment(
        self,
        lhs_name: str,
        rhs_name: str,
        indices: List[str],
        lhs_is_indexed: bool = True,
    ) -> None:
        """Add an indexed assignment statement to the code block.

        Supports both patterns:
        - lhs[indices] = rhs  (when lhs_is_indexed=True)
        - lhs = rhs[indices]  (when lhs_is_indexed=False)

        Args:
            lhs_name: Left-hand side variable name
            indices: List of index expressions as strings
            rhs_name: Right-hand side variable name
            lhs_is_indexed: If True, generates lhs[indices] = rhs
                            If False, generates lhs = rhs[indices]
        """
        if lhs_is_indexed:
            # Pattern: lhs[indices] = rhs
            subscript_ast = CodeGenerator.subscript(lhs_name, indices, ctx=ast.Store())
            value_ast = CodeGenerator.name_load(rhs_name)
            self.add_assignment(subscript_ast, value_ast)
        else:
            # Pattern: lhs = rhs[indices]
            subscript_ast = CodeGenerator.subscript(rhs_name, indices, ctx=ast.Load())
            target_ast = CodeGenerator.name_store(lhs_name)
            self.add_assignment(target_ast, subscript_ast)

    def add_from_import(self, module: str, name: str) -> None:
        """Add an "from X import Y" statement to the code block."""
        from_import_stmt = CodeGenerator.from_import(module, name)
        self.add_statement(from_import_stmt)

    def add_class_call_assignment(
        self, class_or_obj_name: str, args, kwargs, target_name: str
    ) -> None:
        """
        This method can be used in two cases:

        1. Create an object of class <class_or_obj_name> with arguments <args> and
        <kwargs>. The created object's name is <target_name>.
        2. Invoke the "__call__" method of an object named <class_or_obj_name> with
        arguments <args> and <kwargs>. The return value's name is <target_name>.

        """
        call_node = CodeGenerator.class_call(class_or_obj_name, args, kwargs)
        target_name_node = CodeGenerator.name_store(target_name)
        self.add_assignment(target_name_node, call_node)

    def merge(self, other: "ASTCodeBlock") -> None:
        """Merge another code block into this one."""
        self.statements.extend(other.statements)


@dataclass
class CodeNode:
    """Base class for all code structure nodes."""

    name: str
    node_type: NodeType = field(init=False)
    ast_code_block: ASTCodeBlock = field(default_factory=ASTCodeBlock)

    def __str__(self) -> str:
        return f"{self.name} ({self.node_type.value})"


@dataclass
class InputNode(CodeNode):
    """Node representing an input parameter."""

    fx_node_str: Optional[str] = None

    def __post_init__(self) -> None:
        self.node_type = NodeType.INPUT


def create_annotation_ast(annotation_str):
    if not annotation_str:
        return None
    if "." in annotation_str:
        parts = annotation_str.split(".")
        value = ast.Name(id=parts[0], ctx=ast.Load())
        for part in parts[1:]:
            value = ast.Attribute(value=value, attr=part, ctx=ast.Load())
        return value
    return ast.Name(id=annotation_str, ctx=ast.Load())


@dataclass
class ComputationNode(CodeNode):
    """Node representing a computation operation."""

    fx_node_str: Optional[str] = None
    fx_node_stack_trace: Optional[str] = None
    alias_info: Optional[Dict[str, str]] = None
    nki_aliased_inputs: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_type = NodeType.COMPUTATION

    def get_stack_trace_summary(self) -> Optional[str]:
        """Extract a readable summary from the stack trace."""
        if self.fx_node_stack_trace and (
            parsed_stack_trace := _parse_stack_trace(self.fx_node_stack_trace)
        ):
            return parsed_stack_trace.get_summary_str()
        return None

    def set_alias_info(self, source: str, target: str) -> None:
        """Set alias information between source and target variables.

        Args:
            source: Name of the output variable
            target: Name of the input variable being aliased
        """
        self.alias_info = {
            "output_name": source,
            "input_name": target,
        }

    def get_alias_info(self) -> Optional[Dict[str, str]]:
        """Get the alias mapping information."""
        return self.alias_info

    def set_nki_aliased_input(self, input_name: str) -> None:
        self.nki_aliased_inputs.append(input_name)

    def get_nki_aliased_inputs(self) -> List[str]:
        return self.nki_aliased_inputs


@dataclass
class OutputNode(CodeNode):
    """Node representing a function output."""

    def __post_init__(self) -> None:
        self.node_type = NodeType.OUTPUT


@dataclass
class AliasedOutputNode(OutputNode):
    """Node representing an output that is an alias of an input."""

    input_idx: Optional[int] = None
    output_idx: Optional[int] = None

    def __post_init__(self) -> None:
        self.node_type = NodeType.ALIASED_OUTPUT


@dataclass
class NKIPyAST:
    """Complete representation of the function code structure."""

    inputs: List[InputNode] = field(default_factory=list)
    computations: List[ComputationNode] = field(default_factory=list)
    outputs: List[OutputNode] = field(default_factory=list)
    alias_map: Dict[int, int] = field(default_factory=dict)
    none_ouput_idx: List[int] = field(default_factory=list)
    nki_aliased_inputs: List[str] = field(default_factory=list)

    def add_input(self, node: InputNode) -> None:
        """Add an input node to the structure."""
        self.inputs.append(node)

    def add_computation(self, node: ComputationNode) -> None:
        """Add a computation node to the structure."""
        self.computations.append(node)
        if node.get_alias_info() is not None:
            self._add_aliased_output(node)
        if len(node.get_nki_aliased_inputs()) > 0:
            self._add_nki_aliased_input(node)

    def add_output(self, node: OutputNode) -> None:
        """Add an output node to the structure."""
        self.outputs.append(node)

    def add_none_output_idx(self, idx: int):
        """Add an none output idx."""
        self.none_ouput_idx.append(idx)

    def _add_aliased_output(self, node: ComputationNode) -> None:
        """Add an aliased output based on a computation node.

        Args:
            node: Computation node containing alias information

        Raises:
            ValueError: If the input variable referenced in alias info doesn't exist
        """
        alias_info = node.get_alias_info()
        if not alias_info:
            return

        output_name = alias_info["output_name"]
        input_name = alias_info["input_name"]

        input_names = [input_node.name for input_node in self.inputs]
        try:
            input_idx = input_names.index(input_name)
        except ValueError:
            raise ValueError(
                f"Aliased input '{input_name}' not found in inputs: {input_names}"
            )

        output_idx = len(self.outputs)
        self.alias_map[output_idx] = input_idx

        aliased_output_node = AliasedOutputNode(
            name=output_name,
            input_idx=input_idx,
            output_idx=output_idx,
        )
        self.outputs.append(aliased_output_node)

    def _add_nki_aliased_input(self, node: ComputationNode) -> None:
        nki_aliased_inputs = node.get_nki_aliased_inputs()
        input_names = [input_node.name for input_node in self.inputs]

        # Create an output node for each NKI aliased input
        # if this input is not already added
        for input_name in nki_aliased_inputs:
            try:
                input_idx = input_names.index(input_name)
            except ValueError:
                raise ValueError(
                    f"NKI aliased input '{input_name}' not found in "
                    f"inputs: {input_names}"
                )
            if input_name in self.nki_aliased_inputs:
                continue
            self.nki_aliased_inputs.append(input_name)

            output_idx = len(self.outputs)
            self.alias_map[output_idx] = input_idx
            aliased_output_node = AliasedOutputNode(
                name=input_name,
                input_idx=input_idx,
                output_idx=output_idx,
            )
            self.outputs.append(aliased_output_node)

    def generate_function_code(self, func_name: str) -> str:
        """Generate complete function code as a string.

        Args:
            func_name: Name of the function to generate

        Returns:
            String containing the generated Python function
        """
        # Create input parameters
        input_params = []
        for input_node in self.inputs:
            annot = None
            if input_node.name in self.nki_aliased_inputs:
                annot = create_annotation_ast("nt.mutable_tensor")
            input_params.append(ast.arg(arg=input_node.name, annotation=annot))

        # Create function arguments
        func_args = ast.arguments(
            posonlyargs=[],
            args=input_params,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

        # Function body
        body_block = ASTCodeBlock()

        # Add input node comments
        for input_node in self.inputs:
            if input_node.fx_node_str:
                body_block.add_comment(input_node.fx_node_str)
            body_block.merge(input_node.ast_code_block)

        # Add computation nodes
        prev_stack_trace = None
        for comp_node in self.computations:
            stack_trace = comp_node.get_stack_trace_summary()
            if stack_trace != prev_stack_trace:
                prev_stack_trace = stack_trace
                body_block.add_empty_line()
                body_block.add_comment(f"STACK_TRACE: {stack_trace}")

            if comp_node.fx_node_str:
                body_block.add_comment(comp_node.fx_node_str)
            body_block.merge(comp_node.ast_code_block)

        # Add return statement
        output_names = [output.name for output in self.outputs]
        return_value = ast.Tuple(
            elts=[ast.Name(id=name, ctx=ast.Load()) for name in output_names],
            ctx=ast.Load(),
        )

        body_block.add_empty_line()
        body_block.add_statement(ast.Return(value=return_value))

        # Create function definition
        func_def = ast.FunctionDef(
            name=func_name,
            args=func_args,
            body=body_block.statements,
            decorator_list=[],
            returns=None,
        )

        # Create module with imports and function definition
        module = ast.Module(
            body=[
                ast.Import(names=[ast.alias(name="numpy", asname=NUMPY_PKG)]),
                ast.ImportFrom(
                    module="nkipy.core",
                    names=[ast.alias(name="tensor_apis", asname=None)],
                    level=0,
                ),
                ast.ImportFrom(
                    module="nkipy.core.language",
                    names=[ast.alias(name="bfloat16", asname=None)],
                    level=0,
                ),
                ast.ImportFrom(
                    module="neuronxcc.nki.language",
                    names=[ast.alias(name="float8_e5m2", asname=None)],
                    level=0,
                ),
                ast.ImportFrom(
                    module="nkipy.distributed",
                    names=[ast.alias(name="collectives", asname=CC_PKG)],
                    level=0,
                ),
                ast.ImportFrom(
                    module="neuronxcc.nki.language",
                    names=[ast.alias(name="nc", asname="VNC")],
                    level=0,
                ),
                ast.Import(
                    names=[ast.alias(name="neuronxcc.nki.typing", asname="nt")],
                    level=0,
                ),
                ast.Import(names=[ast.alias(name="base64")]),
                ast.Expr(value=ast.Constant(value="")),
                func_def,
            ],
            type_ignores=[],
        )

        ast.fix_missing_locations(module)

        code = Unparser.unparse(module)
        return code
