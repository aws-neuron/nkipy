# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for graph manipulation and hashing."""

import copy
import hashlib
import importlib.util
import logging
import operator
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx
from tabulate import tabulate
from torch.fx import GraphModule, Node
from torch.fx.graph import dtype_abbrs
from torch.fx.node import _get_qualified_name
from torch.fx.passes.split_module import split_module

from torch_to_nkipy.utils import ops  # noqa

logger = logging.getLogger(__name__)

_MARK_SG_OP = torch.ops.nkipy.mark_subgraph_identity.default


_compiler_version_str = None


def _get_compiler_version() -> str:
    """Get the neuronxcc compiler version string (cached)."""
    global _compiler_version_str
    if _compiler_version_str is None:
        try:
            import neuronxcc

            _compiler_version_str = str(neuronxcc.__version__)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Neuronxcc is not properly installed!")
    return _compiler_version_str


def hash_gm_with_tensors(
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    hash_length: int = 8,
    additional_compiler_args: str = "",
) -> str:
    """Hash a GraphModule and its input tensors.

    Args:
        graph_module: The FX graph module to hash
        example_inputs: List of input tensors
        hash_length: Length of the output hash string (default: 8)
        additional_compiler_args: Extra compiler flags to include in hash

    Returns:
        str: Hash string of specified length

    Raises:
        ValueError: If hash_length is invalid
    """
    compiler_version_args_str = _get_compiler_version() + additional_compiler_args

    if hash_length > 32:
        raise ValueError("hash_length must be <= 32")
    if hash_length % 2 != 0:
        raise ValueError("hash_length must be even")

    hasher = hashlib.blake2b(
        compiler_version_args_str.encode("utf-8"), digest_size=hash_length // 2
    )
    hasher.update(graph_module.code.encode("utf-8"))

    for inp in example_inputs:
        if not hasattr(inp, "shape"):
            # Non-tensor inputs (e.g. SymInt from dynamic shapes) — hash repr
            hasher.update(repr(inp).encode("utf-8"))
            continue
        shape_str = str(inp.shape)
        dtype_str = str(inp.dtype)
        hasher.update(shape_str.encode("utf-8"))
        hasher.update(dtype_str.encode("utf-8"))

    return hasher.hexdigest()


def load_func_from_file(
    file_path: Union[Path, str],
    func_name: str,
) -> Callable:
    """Load a function from a Python file.

    Args:
        file_path: Path to the Python file
        func_name: Name of the function to load

    Returns:
        Callable: The loaded function

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If module loading fails
        AttributeError: If function not found
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        module_name = str(file_path.stem).replace(os.sep, "_").replace(":", "")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        func = getattr(module, func_name, None)
        if func is None:
            raise AttributeError(f"Function '{func_name}' not found in {file_path}")

        return func
    except Exception as e:
        raise ImportError(f"Error loading function: {e}") from e


def save_string_to_file(
    content: str,
    file_path: Union[Path, str],
    mode: str = "w",
    encoding: str = "utf-8",
) -> None:
    """Save a string to a file.

    Args:
        content: String content to save
        file_path: Path to the output file
        mode: File opening mode
        encoding: File encoding
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, mode, encoding=encoding) as file:
        file.write(str(content))


def save_gm_to_file(
    gm: torch.fx.GraphModule,
    output_dir: Union[Path, str],
    file_prefix: str,
) -> None:
    """Save a GraphModule to files in tabulate and readable formats.

    Args:
        gm: The GraphModule to save
        output_dir: Directory to save the files
        file_prefix: Prefix for the output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tabulate format
    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes]
    tabulate_str = tabulate(
        node_specs,
        headers=["opcode", "name", "target", "args", "kwargs"],
        tablefmt="grid",
    )
    save_string_to_file(tabulate_str, output_dir / f"{file_prefix}_tabulate.txt")

    # Save readable format
    readable_str = gm.print_readable(print_output=False)
    save_string_to_file(readable_str, output_dir / f"{file_prefix}_readable.txt")


def stringify_shape(shape: Iterable[int]) -> str:
    """
    Convert a tensor shape to a string representation.

    Args:
        shape: Tensor dimensions to convert to string

    Returns:
        A string format like "[1, 2, 3]"
    """
    return f"[{', '.join(str(x) for x in shape)}]"


def format_args(args: Iterable[Any], kwargs: Dict[str, Any]) -> str:
    """
    Format function arguments and keyword arguments as a string.

    Creates a comma-separated representation of both positional and
    keyword arguments for function call string representation.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        A formatted string like "arg1, arg2, key1=val1"
    """
    args_s = ", ".join(str(a) for a in args)
    kwargs_s = ", ".join(f"{k}={str(v)}" for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f"{args_s}, {kwargs_s}"
    return args_s or kwargs_s


def get_shape_from_fx_node(node: torch.fx.Node) -> Optional[torch.Size]:
    """
    Extract tensor shape from an FX node's metadata.

    Safely handles cases where metadata might be missing.

    Args:
        node: The FX node to extract shape from

    Returns:
        The tensor shape if available, None otherwise
    """
    try:
        meta_val = node.meta.get("val")
        if meta_val is not None:
            return meta_val.shape
    except (AttributeError, KeyError):
        pass
    return None


def get_dtype_from_fx_node(node: torch.fx.Node) -> Optional[torch.dtype]:
    """
    Extract tensor dtype from an FX node's metadata.

    Safely handles cases where metadata might be missing.

    Args:
        node: The FX node to extract dtype from

    Returns:
        The tensor dtype if available, None otherwise
    """
    try:
        meta_val = node.meta.get("val")
        if meta_val is not None:
            return meta_val.dtype
    except (AttributeError, KeyError):
        pass
    return None


def stringify_fx_node(node: torch.fx.Node) -> str:
    """
    Create a string representation of placeholder or call_function FX nodes.

    Args:
        node: The FX node to stringify (must be placeholder or call_function)

    Returns:
        A string representation of the node

    Raises:
        ValueError: If node type is not placeholder or call_function
    """
    # Safely get shape and dtype
    shape = get_shape_from_fx_node(node)
    dtype = get_dtype_from_fx_node(node)

    # Format shape and dtype info
    dtype_shape_str = ""
    if shape is not None and dtype is not None:
        shape_str = stringify_shape(shape)
        dtype_str = dtype_abbrs.get(dtype, str(dtype))
        dtype_shape_str = f"{dtype_str}{shape_str}"

    # Handle only placeholder and call_function
    if node.op == "placeholder":
        fx_node_str = f"{node}: {dtype_shape_str}"
    elif node.op == "call_function":
        fx_node_str = (
            f"{node}: {dtype_shape_str} = {_get_qualified_name(node.target)}"
            f"({format_args(node.args, node.kwargs)})"
        )
    elif node.op == "get_attr":
        fx_node_str = f"{node} = self.{node.target}"
    else:
        raise ValueError(
            f"stringify_fx_node only supports 'placeholder' and"
            f"'call_function' nodes, got: {node.op}"
        )

    return fx_node_str


def resolve_shape_placeholder_expand(
    node: torch.fx.Node, shape: Sequence[int]
) -> List[int]:
    """
    Resolve shape specification for expand operation containing -1 placeholders.

    In PyTorch's expand operation, -1 in a shape specification means "keep the original
    dimension size unchanged", unlike reshape/view where it means "infer this
    dimension".

    Args:
        node: FX node containing the input tensor metadata
        shape: Target shape specification which may contain -1 placeholders

    Returns:
        List[int]: The resolved shape with -1 replaced by the original dimension size

    Raises:
        ValueError: If node doesn't have shape metadata
    """
    # Convert shape to list for modification
    shape_list = list(shape)

    # If no -1 in shape, return as is
    if -1 not in shape_list:
        return shape_list

    # Check for shape metadata
    input_shape = get_shape_from_fx_node(node)
    if input_shape is None:
        raise ValueError(f"Node {node} does not have shape metadata")

    # Validate shape length
    if len(shape_list) < len(input_shape):
        raise ValueError(
            f"Expand shape {shape_list} cannot have fewer dimensions than input shape "
            f"{input_shape}"
        )

    # Count occurrences of -1
    neg_ones = shape_list.count(-1)
    if neg_ones > 1:
        # Replace each -1 with the corresponding dimension from the input shape
        for i, dim in enumerate(shape_list):
            if dim == -1:
                # Ensure the index is valid for the input shape
                if i >= len(input_shape):
                    raise ValueError(
                        f"Cannot use -1 at position {i} "
                        f"as input has only {len(input_shape)} dimensions"
                    )
                shape_list[i] = input_shape[i]
        return shape_list

    # Find the position of -1
    neg_one_index = shape_list.index(-1)

    # Ensure the index is valid for the input shape
    if neg_one_index >= len(input_shape):
        raise ValueError(
            f"Cannot use -1 at position {neg_one_index} "
            f"as input has only {len(input_shape)} dimensions"
        )

    # Replace -1 with the original dimension size
    shape_list[neg_one_index] = input_shape[neg_one_index]

    return shape_list


def _count_subgraph_markers(gm: GraphModule) -> int:
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and n.target is _MARK_SG_OP
    )


def remove_trailing_markers(gm: GraphModule) -> GraphModule:
    """
    Removes any continuous sequence of subgraph marker nodes
    that immediately precede the 'output' node.

    It searches backward from the 'output' node and deletes
    all contiguous marker nodes until it hits the first non-marker node.
    """

    # *** Ensure _MARK_SG_OP is your correct marker op target function ***

    nodes = list(gm.graph.nodes)

    # If the graph is empty or the last node isn't an output, just return
    if not nodes or nodes[-1].op != "output":
        return gm

    nodes_to_delete: List[Node] = []
    changed = False

    # Iterate backward, starting from the node right before the "output" node
    for n in reversed(nodes[:-1]):
        is_marker = n.op == "call_function" and n.target is _MARK_SG_OP

        if is_marker:
            nodes_to_delete.append(n)
            changed = True
        else:
            # Found the first non-marker node, stop searching
            break

    if changed:
        # Must re-wire all nodes first before deleting
        # nodes_to_delete is in reverse graph order (e.g., [marker2, marker1])
        for n in nodes_to_delete:
            # Assume the marker's first argument (args[0]) is the data it passes through
            if not n.args:
                raise RuntimeError(
                    f"Cannot remove trailing marker {n.name} "
                    f"because it has no input arguments."
                )

            # Re-point all users of this marker node to use the marker's *input*
            # e.g., output(marker2) -> output(marker1)
            input_node = n.args[0]
            n.replace_all_uses_with(input_node)

        # Now that all nodes are safely re-wired, we can safely erase them
        # (in the order we found them, i.e., reverse graph order)
        for n in nodes_to_delete:
            gm.graph.erase_node(n)

        gm.graph.lint()
        gm.recompile()

    return gm


def split_on_subgraph_markers(gm: GraphModule) -> Tuple[GraphModule, bool]:
    """
    Splits the graph sequentially, incrementing the partition ID for each _MARK_SG_OP.

    New Logic:
    - Start in partition 0 (pid = 0).
    - Iterate through nodes, assigning them to the current pid.
    - When a _MARK_SG_OP is encountered:
        1.  Check if the current 'pid' partition contains any "real content".
        2.  "Real content" is defined as any node that is NOT a
            placeholder, output, or maker.
        3.  If it DOES have content, increment the partition ID (pid += 1).
        4.  If it does NOT have content (i.e., the partition is empty),
            the partition ID does *not* increment.
        5.  The _MARK_SG_OP node itself is assigned to the new (or unchanged) pid.
    - If no makers exist, or if all nodes end up in a single partition,
      return the original graph unchanged.
    """
    if _count_subgraph_markers(gm) == 0:
        return gm, False  # no-op

    pid = 0
    pmap: Dict[Node, int] = {}

    # Tracks whether the current partition has "real content"
    current_partition_has_content = False

    for n in gm.graph.nodes:
        is_marker = n.op == "call_function" and n.target is _MARK_SG_OP

        if is_marker:
            # Encountered a maker.
            # Check if we should move to the next partition.
            # We only switch if the current partition actually did something.
            if current_partition_has_content:
                pid += 1  # Move to the next partition
                # Reset the content flag for the new partition
                current_partition_has_content = False

            # If current_partition_has_content was False,
            # pid remains unchanged, and this maker will be "absorbed"
            # into the current empty partition.

            # Assign the maker node itself to the (potentially new) partition
            pmap[n] = pid
            # Note: The marker node itself does *not* count as "content"

        else:
            # Not a maker node
            pmap[n] = pid  # Assign to the current partition

            # Check if this node counts as "content"
            # Placeholders and outputs don't count as real computation
            if n.op not in ("placeholder", "output"):
                current_partition_has_content = True

    # If only one partition was generated in the end
    # (e.g., markers existed but all partitions were empty),
    # then skip the split.
    if len({pmap[n] for n in pmap}) <= 1:
        return gm, False

    out = split_module(gm, gm, lambda n: pmap[n])
    return out, True


def _remove_all_markers(gm: GraphModule) -> GraphModule:
    """
    Finds and removes *all* instances of the marker op from the graph.

    This function "bypasses" the marker nodes by connecting their
    inputs directly to their outputs, and then erases the
    marker nodes themselves.
    """

    nodes_to_delete: List[Node] = []

    # Iterate through the graph to find all marker nodes to be deleted
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is _MARK_SG_OP:
            nodes_to_delete.append(n)

    # If none were found, we can clock out early
    if not nodes_to_delete:
        return gm

    # Iterate the delete list, "grafting" their "users" onto their "inputs"
    for n in nodes_to_delete:
        # Again, assume the marker's first arg (args[0]) is the data it passes through
        if not n.args:
            raise RuntimeError(
                f"Cannot remove marker {n.name} because it has no input arguments."
            )

        # Re-point all users of this marker node to use the marker's *input*
        input_node = n.args[0]
        n.replace_all_uses_with(input_node)

    # Since they have no users now, they can be safely erased from the graph
    # (The order doesn't matter since we're iterating from a separate list)
    for n in nodes_to_delete:
        gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()

    return gm


def _format_target_name(target: Any) -> str:
    """Return a readable identifier for a callable target (e.g., aten.mm.default)."""
    return getattr(target, "__name__", str(target))


def _build_node_index(gm: GraphModule) -> Dict[Node, int]:
    """Stable index for nodes in this graph (used to reference producers)."""
    return {n: i for i, n in enumerate(gm.graph.nodes)}


def graph_signature_hash(
    gm: GraphModule,
    *,
    include_tensor_meta: bool = True,
    include_literal_kwargs: bool = True,
    include_io_structure: bool = True,
) -> str:
    """
    Compute a structural, weight-agnostic hash for an FX graph.

    - Ignores `placeholder` and `output` unless `include_io_structure=True`
      (then we encode input order/meta and output structure).
    - Treats `get_attr` generically so different param names hash the same.
    - Optionally ignores a known marker op.
    - Optionally includes dtype/shape from node.meta['tensor_meta'].
    - Optionally includes literal kwargs (ints/floats/bools/None/str).

    Returns a SHA-256 hex digest string.
    """
    node_index = _build_node_index(gm)

    def encode_operand(obj: Any):
        """Encode args/kwargs in a stable, hashable way."""
        if isinstance(obj, Node):
            return ("node", node_index[obj])
        if isinstance(obj, (int, float, bool, str)) or obj is None:
            return ("lit", obj)
        if isinstance(obj, (tuple, list)):
            return ("seq", tuple(encode_operand(x) for x in obj))
        if isinstance(obj, dict):
            return (
                "dict",
                tuple(sorted((k, encode_operand(v)) for k, v in obj.items())),
            )
        return ("type", type(obj).__name__)

    signature_rows: List[tuple] = []

    for n in gm.graph.nodes:
        # Inputs
        if n.op == "placeholder":
            if include_io_structure:
                row = [("IN",)]  # position implied by traversal order
                if include_tensor_meta:
                    tm = n.meta.get("tensor_meta")
                    if tm is not None:
                        row.append(("nkipy", str(tm.dtype), tuple(tm.shape)))
                signature_rows.append(tuple(row))
            continue

        # Body nodes
        if n.op == "call_function":
            head = ("F", _format_target_name(n.target))
        elif n.op == "call_method":
            head = ("M", str(n.target))
        elif n.op == "get_attr":
            head = ("A",)  # attribute name intentionally ignored (weight-agnostic)
            logger.warning("'get_attr' appears in graph, check it!")
        elif n.op == "call_module":
            try:
                head = ("C", type(gm.get_submodule(n.target)).__name__)
            except Exception:
                head = ("C", "unknown")
        elif n.op == "output":
            if include_io_structure:
                signature_rows.append(
                    (
                        ("OUT",),
                        ("result", encode_operand(n.args[0])),
                    )
                )
            continue
        else:
            head = (n.op,)

        row = [head, ("args", encode_operand(n.args))]

        if include_literal_kwargs and n.kwargs:
            lits = tuple(
                sorted(
                    (k, v)
                    for k, v in n.kwargs.items()
                    if isinstance(v, (int, float, bool, str, type(None)))
                )
            )
            if lits:
                row.append(("kw", lits))

        if include_tensor_meta:
            tm = n.meta.get("tensor_meta")
            if tm is not None:
                row.append(("nkipy", str(tm.dtype), tuple(tm.shape)))

        signature_rows.append(tuple(row))

    blob = repr(tuple(signature_rows)).encode()
    return hashlib.sha256(blob).hexdigest()


def deduplicate_middle_partitions_fx(
    split_gm: GraphModule,
    *,
    include_tensor_meta: bool = True,
    include_literal_kwargs: bool = True,
    include_io_structure: bool = True,
) -> GraphModule:
    """
    FX-only transform: deduplicate identical middle partitions (children 1..n-2).

    - Computes a structural hash for each child GraphModule.
    - For children with identical hashes, rewrites top calls to the
      first-seen representative and removes unused duplicate submodules.

    Returns a NEW GraphModule.
    """
    if not isinstance(split_gm, GraphModule):
        raise TypeError("split_gm must be a torch.fx.GraphModule")

    gm = copy.deepcopy(split_gm)
    children: List[tuple[str, GraphModule]] = list(gm.named_children())
    if len(children) < 3:
        return gm  # nothing to dedupe

    # Build hash → representative map for middle partitions
    name_to_hash: Dict[str, str] = {}
    hash_to_rep: Dict[str, str] = {}
    name_to_rep: Dict[str, str] = {}

    for name, sub in children:
        sub_copy = copy.deepcopy(sub)
        sub_copy = _remove_all_markers(sub_copy)
        h = graph_signature_hash(
            sub_copy,
            include_tensor_meta=include_tensor_meta,
            include_literal_kwargs=include_literal_kwargs,
            include_io_structure=include_io_structure,
        )
        name_to_hash[name] = h
        rep = hash_to_rep.setdefault(h, name)  # pick the first as canonical
        name_to_rep[name] = rep

    # Rewrite top-graph call_module targets for duplicates
    used_targets = set()
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        tgt = node.target
        if tgt in name_to_rep and tgt != name_to_rep[tgt]:
            node.target = name_to_rep[tgt]
        used_targets.add(node.target)

    gm.graph.lint()
    gm.recompile()

    # Drop unused duplicate children
    for name, _ in list(gm.named_children()):
        if name not in used_targets:
            delattr(gm, name)

    return gm


def move_inplace_copies(gm: GraphModule) -> GraphModule:
    changed = False
    g = gm.graph
    for n in list(g.nodes):  # snapshot since we'll mutate order
        if n.op == "call_function" and n.target is torch.ops.aten.copy_.default:
            dst, src = n.args[:2]
            assert isinstance(src, Node) and isinstance(dst, Node), (
                "copy_ args must be Nodes"
            )
            src.append(n)
            changed = True

    if changed:
        g.lint()  # fail fast if the schedule is now invalid
        gm.recompile()
    return gm


def canonicalize_split_returns(parent_gm):
    def is_split_submod(name, mod):
        return name.startswith("submod_") and isinstance(mod, GraphModule)

    def _tupleize_sub_return(sub_gm: GraphModule) -> bool:
        out = next(n for n in sub_gm.graph.nodes if n.op == "output")
        val = out.args[0]
        if isinstance(val, tuple):
            return False
        out.args = ((val,),)
        sub_gm.graph.lint()
        sub_gm.recompile()
        return True

    def _replace_in_tree(obj, needle, repl):
        if obj is needle:
            return repl
        if isinstance(obj, (tuple, list)):
            t = tuple if isinstance(obj, tuple) else list
            return t(_replace_in_tree(x, needle, repl) for x in obj)
        if isinstance(obj, dict):
            return {k: _replace_in_tree(v, needle, repl) for k, v in obj.items()}
        return obj

    def _rewrite_parent_uses_to_getitem0(gm: GraphModule, call_node: Node) -> bool:
        users = list(call_node.users)
        non_getitem_users = [
            u
            for u in users
            if not (u.op == "call_function" and u.target is operator.getitem)
        ]
        if not non_getitem_users:
            return False
        with gm.graph.inserting_after(call_node):
            get0 = gm.graph.call_function(operator.getitem, args=(call_node, 0))
        changed_local = False
        for u in non_getitem_users:
            new_args = _replace_in_tree(u.args, call_node, get0)
            new_kwargs = _replace_in_tree(u.kwargs, call_node, get0)
            if new_args is not u.args:
                u.args = new_args
                changed_local = True
            if new_kwargs is not u.kwargs:
                u.kwargs = new_kwargs
                changed_local = True
        return changed_local

    changed = False
    name2mod = dict(parent_gm.named_modules())
    touched = set()

    for name, mod in name2mod.items():
        if is_split_submod(name, mod):
            if _tupleize_sub_return(mod):
                changed = True
                touched.add(name)

    for node in parent_gm.graph.nodes:
        if node.op == "call_module" and str(node.target) in touched:
            if _rewrite_parent_uses_to_getitem0(parent_gm, node):
                changed = True

    if changed:
        parent_gm.graph.lint()
        parent_gm.recompile()
    return parent_gm


def gm_split_and_wrap(gm, wrapper_class, options):
    gm = remove_trailing_markers(gm)
    gm = move_inplace_copies(gm)
    gm, _ = split_on_subgraph_markers(gm)
    gm = deduplicate_middle_partitions_fx(gm)
    gm = canonicalize_split_returns(gm)

    name2mod: Dict[str, torch.nn.Module] = dict(gm.named_modules())
    wrapped: Dict[str, torch.nn.Module] = {}

    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        tgt = str(n.target)
        sub = name2mod.get(tgt, None)
        if sub is None:
            raise RuntimeError(f"Missing submodule: {tgt}")
        if not isinstance(sub, torch.fx.GraphModule):
            continue
        if tgt not in wrapped:
            wrapper = wrapper_class(sub, options)
            wrapped[tgt] = wrapper
            setattr(gm, tgt, wrapper)

    return gm
