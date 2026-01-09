#!/usr/bin/env python3
"""
Generate API documentation for NKIPy operations.

This script introspects the ops module and numpy dispatch to automatically
generate documentation pages for:
1. Operations Reference - all supported ops with their backends
2. NumPy API Compatibility - mapping of numpy functions to nkipy ops

Usage:
    python docs/generate_api_docs.py

This script is designed to be run from the repository root directory.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add the nkipy source to path for introspection
REPO_ROOT = Path(__file__).parent.parent
NKIPY_SRC = REPO_ROOT / "nkipy" / "src"
sys.path.insert(0, str(NKIPY_SRC))

# Paths
OPS_DIR = NKIPY_SRC / "nkipy" / "core" / "ops"
NUMPY_DISPATCH_FILE = NKIPY_SRC / "nkipy" / "core" / "_numpy_dispatch.py"
DOCS_API_DIR = REPO_ROOT / "docs" / "api"


def parse_ops_module(filepath: Path) -> Dict[str, dict]:
    """Parse an ops module file to extract operation definitions.

    Returns a dict mapping op names to their metadata.
    """
    ops = {}

    with open(filepath, "r") as f:
        content = f.read()

    # Parse the AST to find Op() instantiations
    tree = ast.parse(content)

    # Find all assignments like: add = Op("add") or add = _make_binary_op("add", np.add)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    op_name = target.id
                    # Check if it's an Op() call
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        # Direct Op() call
                        if isinstance(func, ast.Name) and func.id == "Op":
                            if node.value.args and isinstance(
                                node.value.args[0], ast.Constant
                            ):
                                ops[op_name] = {
                                    "name": node.value.args[0].value,
                                    "type": "direct",
                                }
                        # Factory function call like _make_binary_op
                        elif isinstance(func, ast.Name) and func.id.startswith(
                            "_make_"
                        ):
                            if node.value.args and isinstance(
                                node.value.args[0], ast.Constant
                            ):
                                ops[op_name] = {
                                    "name": node.value.args[0].value,
                                    "type": "factory",
                                    "factory": func.id,
                                }

    return ops


def parse_numpy_dispatch() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Parse the numpy dispatch file to extract registered APIs and unsupported ops.

    Returns:
        - List of (numpy_func, ops_func) tuples for supported operations
        - List of (numpy_func, reason) tuples for unsupported operations
    """
    supported = []
    unsupported = []

    with open(NUMPY_DISPATCH_FILE, "r") as f:
        content = f.read()

    # Find all _register_numpy_api calls
    # Pattern: _register_numpy_api(np.func, ops.func)
    register_pattern = r"_register_numpy_api\(np\.(\w+),\s*ops\.(\w+)\)"
    for match in re.finditer(register_pattern, content):
        numpy_func = match.group(1)
        ops_func = match.group(2)
        supported.append((numpy_func, ops_func))

    # Find NOT SUPPORTED comments
    # Pattern: # np.func - reason
    unsupported_pattern = r"#\s*np\.(\w+)\s*-\s*(.+?)(?:\n|$)"
    for match in re.finditer(unsupported_pattern, content):
        numpy_func = match.group(1)
        reason = match.group(2).strip()
        unsupported.append((numpy_func, reason))

    return supported, unsupported


def get_ops_by_category() -> Dict[str, Dict[str, dict]]:
    """Get all operations organized by category (file name).

    Returns a dict mapping category names to dicts of operations.
    """
    categories = {}

    # Map file names to display names
    category_names = {
        "binary.py": "Binary Operations",
        "unary.py": "Unary Operations",
        "reduce.py": "Reduction Operations",
        "transform.py": "Transform Operations",
        "creation.py": "Creation Operations",
        "indexing.py": "Indexing Operations",
        "linalg.py": "Linear Algebra Operations",
        "nn.py": "Neural Network Operations",
        "conv.py": "Convolution Operations",
        "collectives.py": "Collective Operations",
    }

    for filename in OPS_DIR.glob("*.py"):
        if filename.name.startswith("_"):
            continue

        category = category_names.get(
            filename.name, filename.stem.title() + " Operations"
        )
        ops = parse_ops_module(filename)

        if ops:
            categories[category] = ops

    return categories


def get_backend_support() -> Dict[str, List[str]]:
    """Get backend support for each operation by importing the ops module.

    Returns a dict mapping op names to list of supported backends.
    """
    backend_support = {}

    try:
        from nkipy.core import ops
        from nkipy.core.ops._registry import Op

        # Get all Op instances from the ops module
        for name in dir(ops):
            obj = getattr(ops, name)
            if isinstance(obj, Op):
                backend_support[name] = list(obj._impls.keys())
    except ImportError as e:
        print(f"Warning: Could not import ops module for backend introspection: {e}")
        print("Backend support information will not be available.")

    return backend_support


def generate_ops_reference() -> str:
    """Generate the operations reference documentation."""
    categories = get_ops_by_category()
    backend_support = get_backend_support()
    numpy_supported, _ = parse_numpy_dispatch()

    # Create a mapping from ops name to numpy name
    ops_to_numpy = {ops_func: numpy_func for numpy_func, ops_func in numpy_supported}

    lines = [
        "# Operations Reference",
        "",
        "This page documents all operations supported by NKIPy, organized by category.",
        "",
        "## Overview",
        "",
        "NKIPy operations are traced and lowered to HLO for compilation by the Neuron Compiler.",
        "",
    ]

    # Generate table of contents
    lines.append("## Categories")
    lines.append("")
    for category in categories:
        anchor = category.lower().replace(" ", "-")
        lines.append(f"- [{category}](#{anchor})")
    lines.append("")

    # Generate each category section
    for category, ops in categories.items():
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Operation | Backend(s) | NumPy Equivalent |")
        lines.append("|-----------|-----------|------------------|")

        for op_name, op_info in sorted(ops.items()):
            backends = backend_support.get(op_name, [])
            backends_str = ", ".join(backends) if backends else "—"
            numpy_equiv = ops_to_numpy.get(op_name, "—")
            if numpy_equiv != "—":
                numpy_equiv = f"`np.{numpy_equiv}`"

            lines.append(f"| `{op_name}` | {backends_str} | {numpy_equiv} |")

        lines.append("")

    # Add API reference section
    lines.extend(
        [
            "## API Reference",
            "",
            "```{eval-rst}",
            ".. automodule:: nkipy.core.ops",
            "   :members:",
            "   :undoc-members:",
            "   :show-inheritance:",
            "```",
        ]
    )

    return "\n".join(lines)


def generate_numpy_compat() -> str:
    """Generate the NumPy compatibility documentation."""
    supported, unsupported = parse_numpy_dispatch()

    # Organize supported ops by category
    categories = {
        "Binary Operations": [],
        "Comparison Operations": [],
        "Unary Operations": [],
        "Reduction Operations": [],
        "Linear Algebra": [],
        "Transform Operations": [],
        "Creation Operations": [],
        "Indexing Operations": [],
        "Broadcast and Copy Operations": [],
    }

    # Categorize based on the comments in _numpy_dispatch.py
    binary_ops = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "maximum",
        "minimum",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    }
    comparison_ops = {
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
    }
    unary_ops = {
        "abs",
        "exp",
        "log",
        "sqrt",
        "square",
        "negative",
        "sin",
        "cos",
        "tan",
        "arctan",
        "tanh",
        "ceil",
        "floor",
        "rint",
        "trunc",
        "sign",
        "invert",
        "bitwise_not",
        "logical_not",
    }
    reduction_ops = {"sum", "max", "min", "mean", "any"}
    linalg_ops = {"matmul"}
    transform_ops = {
        "reshape",
        "transpose",
        "expand_dims",
        "concatenate",
        "split",
        "copy",
        "repeat",
    }
    creation_ops = {"zeros_like", "empty_like", "full_like"}
    indexing_ops = {"where", "take", "take_along_axis", "put_along_axis"}
    broadcast_ops = {"broadcast_to", "copyto"}

    for numpy_func, ops_func in supported:
        entry = (numpy_func, ops_func)
        if numpy_func in binary_ops:
            categories["Binary Operations"].append(entry)
        elif numpy_func in comparison_ops:
            categories["Comparison Operations"].append(entry)
        elif numpy_func in unary_ops:
            categories["Unary Operations"].append(entry)
        elif numpy_func in reduction_ops:
            categories["Reduction Operations"].append(entry)
        elif numpy_func in linalg_ops:
            categories["Linear Algebra"].append(entry)
        elif numpy_func in transform_ops:
            categories["Transform Operations"].append(entry)
        elif numpy_func in creation_ops:
            categories["Creation Operations"].append(entry)
        elif numpy_func in indexing_ops:
            categories["Indexing Operations"].append(entry)
        elif numpy_func in broadcast_ops:
            categories["Broadcast and Copy Operations"].append(entry)

    lines = [
        "# NumPy API Compatibility",
        "",
        "NKIPy provides NumPy-compatible APIs that allow you to use familiar NumPy functions",
        "with NKIPy tensors. When you call a NumPy function on an NKIPy tensor, it automatically",
        "dispatches to the corresponding NKIPy operation.",
        "",
        "## Usage Example",
        "",
        "```python",
        "import numpy as np",
        "from nkipy.core import ops",
        "",
        "# Inside a traced kernel, you can use NumPy functions directly:",
        "# result = np.add(tensor_a, tensor_b)  # Dispatches to ops.add",
        "# result = np.matmul(tensor_a, tensor_b)  # Dispatches to ops.matmul",
        "```",
        "",
        "## Supported NumPy Functions",
        "",
    ]

    # Generate tables for each category
    for category, ops_list in categories.items():
        if not ops_list:
            continue

        lines.append(f"### {category}")
        lines.append("")
        lines.append("| NumPy Function | NKIPy Operation |")
        lines.append("|----------------|-----------------|")

        for numpy_func, ops_func in sorted(ops_list):
            lines.append(f"| `np.{numpy_func}` | `ops.{ops_func}` |")

        lines.append("")

    # Add unsupported operations section
    lines.extend(
        [
            "## Unsupported NumPy Operations",
            "",
            "The following NumPy operations are **not supported** due to hardware limitations:",
            "",
            "| NumPy Function | Reason |",
            "|----------------|--------|",
        ]
    )

    for numpy_func, reason in unsupported:
        lines.append(f"| `np.{numpy_func}` | {reason} |")

    lines.extend(
        [
            "",
            "### Workarounds",
            "",
            "For some unsupported operations, workarounds exist:",
            "",
            "- **`np.mod` / `np.remainder`**: Use `a - b * np.floor(a/b)`",
            "- **`np.positive`**: Use `np.copy(x)` for the `y = +x` operation",
            "",
        ]
    )

    return "\n".join(lines)


def update_index() -> str:
    """Generate updated index.md content."""
    return """# API Reference

```{toctree}
:maxdepth: 2

core
ops
numpy_compat
distributed
runtime
```
"""


def main():
    """Generate all API documentation files."""
    print("Generating API documentation...")

    # Ensure output directory exists
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)

    # Generate operations reference
    ops_content = generate_ops_reference()
    ops_file = DOCS_API_DIR / "ops.md"
    with open(ops_file, "w") as f:
        f.write(ops_content)
    print(f"  Generated: {ops_file}")

    # Generate NumPy compatibility
    numpy_content = generate_numpy_compat()
    numpy_file = DOCS_API_DIR / "numpy_compat.md"
    with open(numpy_file, "w") as f:
        f.write(numpy_content)
    print(f"  Generated: {numpy_file}")

    # Update index
    index_content = update_index()
    index_file = DOCS_API_DIR / "index.md"
    with open(index_file, "w") as f:
        f.write(index_content)
    print(f"  Updated: {index_file}")

    print("\nDone! Generated documentation files:")
    print(f"  - {ops_file}")
    print(f"  - {numpy_file}")
    print(f"  - {index_file}")


if __name__ == "__main__":
    main()
