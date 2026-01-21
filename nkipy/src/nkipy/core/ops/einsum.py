# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Einstein summation (einsum) operation.

Provides a flexible interface for expressing tensor contractions using
Einstein notation, such as matrix multiplication, batch operations,
traces, and more.
"""

import numpy as np
from typing import Dict, List, Set, Tuple

from nkipy.core.ops._registry import Op

# =============================================================================
# Einsum Subscript Parsing
# =============================================================================


def parse_einsum_subscripts(
    subscripts: str, num_operands: int
) -> Tuple[List[str], str]:
    """Parse einsum subscript string into input and output specifications.

    Args:
        subscripts: Einstein notation string (e.g., 'ij,jk->ik' or 'ij,jk')
        num_operands: Number of input operands

    Returns:
        Tuple of (input_specs, output_spec) where:
        - input_specs: List of strings, one per operand (e.g., ['ij', 'jk'])
        - output_spec: Output string (e.g., 'ik'), inferred if not provided

    Examples:
        >>> parse_einsum_subscripts('ij,jk->ik', 2)
        (['ij', 'jk'], 'ik')

        >>> parse_einsum_subscripts('ii', 1)  # trace - inferred output is ''
        (['ii'], '')
    """
    # Remove whitespace
    subscripts = subscripts.replace(" ", "")

    # Check for explicit output specification
    if "->" in subscripts:
        input_str, output_spec = subscripts.split("->")
        input_specs = input_str.split(",")
    else:
        input_specs = subscripts.split(",")
        # Infer output: all indices that appear exactly once across all inputs
        all_indices: Dict[str, int] = {}
        for spec in input_specs:
            for idx in spec:
                all_indices[idx] = all_indices.get(idx, 0) + 1

        # Output contains indices that appear exactly once, in order of first appearance
        # For implicit output with ..., we need to keep ... if present
        output_spec = ""
        seen: Set[str] = set()
        
        # Collect all indices that appear exactly once
        unique_indices = sorted([idx for idx, count in all_indices.items() if count == 1 and idx != "."])
        
        # In implicit mode, we preserve order of appearance
        for spec in input_specs:
            for idx in spec:
                if idx in unique_indices and idx not in seen:
                    output_spec += idx
                    seen.add(idx)

    if len(input_specs) != num_operands:
        raise ValueError(
            f"Number of subscripts ({len(input_specs)}) does not match "
            f"number of operands ({num_operands})"
        )

    return input_specs, output_spec


def analyze_einsum_pattern(
    input_specs: List[str], output_spec: str, shapes: List[Tuple[int, ...]]
) -> Dict:
    """Analyze einsum pattern to determine dimension mapping and operation type.

    Args:
        input_specs: List of input subscript strings
        output_spec: Output subscript string
        shapes: List of input tensor shapes

    Returns:
        Dictionary containing:
        - 'input_dims': Dict mapping each input index to dimension size
        - 'contracting_dims': Set of indices being contracted (summed over)
        - 'batch_dims': Set of indices appearing in all inputs and output
        - 'output_order': List of indices in output order
    """
    # Build index -> dimension size mapping
    input_dims: Dict[str, int] = {}
    for spec, shape in zip(input_specs, shapes):
        if len(spec) != len(shape):
            raise ValueError(
                f"Subscript '{spec}' has {len(spec)} indices but shape has "
                f"{len(shape)} dimensions: {shape}"
            )
        for idx, size in zip(spec, shape):
            if idx in input_dims and input_dims[idx] != size:
                raise ValueError(
                    f"Index '{idx}' has inconsistent dimensions: "
                    f"{input_dims[idx]} vs {size}"
                )
            input_dims[idx] = size

    # Collect all unique indices
    all_indices = set()
    for spec in input_specs:
        all_indices.update(spec)

    # Determine contracting dimensions (in inputs but not output)
    contracting_dims = all_indices - set(output_spec)

    # Determine batch dimensions (in all inputs and output)
    batch_dims = set(output_spec)
    for spec in input_specs:
        batch_dims &= set(spec)

    return {
        "input_dims": input_dims,
        "contracting_dims": contracting_dims,
        "batch_dims": batch_dims,
        "output_order": list(output_spec),
    }


# =============================================================================
# Einsum Operation
# =============================================================================
einsum = Op("einsum")


@einsum.impl("hlo")
def _einsum_hlo(subscripts, *operands, dtype=None):
    """Einstein summation convention on tensors (HLO implementation).

    Implements einsum using HLO operations: transpose, dot_general, reduce.
    Supports common patterns like matrix multiplication, batch operations,
    traces, outer products, and more.

    Args:
        subscripts: Einstein notation string (e.g., 'ij,jk->ik')
        *operands: Input tensors
        dtype: Optional output dtype (if None, inferred from inputs)

    Returns:
        Result tensor according to einsum specification

    Examples:
        >>> # Matrix multiply
        >>> einsum('ij,jk->ik', A, B)

        >>> # Batch matrix multiply
        >>> einsum('bij,bjk->bik', A, B)

        >>> # Trace
        >>> einsum('ii->', A)

        >>> # Outer product
        >>> einsum('i,j->ij', a, b)
    """
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    if not operands:
        raise ValueError("einsum requires at least one operand")

    # Get shapes
    shapes = []
    real_operands = []
    ctx = get_hlo_context()
    
    for op in operands:
        if isinstance(op, NKIPyTensorRef):
            real_operands.append(op)
            shapes.append(op.backend_tensor.shape)
        else:
            # Assume it's an HLO tensor or similar
            # Wrappping it might be needed if we call tensor ops? 
            # The original code handled wrapping later. 
            # We need shapes now for ellipsis expansion.
            real_operands.append(op)
            shapes.append(op.shape)

    # Parse subscripts
    input_specs, output_spec = parse_einsum_subscripts(subscripts, len(operands))

    # Handle repeated indices (Diagonal/Trace)
    # This might modify operands (insert diagonal ops) and specs
    cleaned_input_specs = []
    processed_operands = []
    
    for i, (spec, op) in enumerate(zip(input_specs, real_operands)):
        # Check for repeated indices
        if len(set(spec)) != len(spec):
             new_op, new_spec = _handle_repeated_indices(ctx, op, spec)
             processed_operands.append(new_op)
             cleaned_input_specs.append(new_spec)
        else:
             processed_operands.append(op)
             cleaned_input_specs.append(spec)
    
    input_specs = cleaned_input_specs
    
    # Refresh shapes after potential diagonal reductions
    hlo_operands = []
    final_shapes = []
    for op in processed_operands:
        if isinstance(op, NKIPyTensorRef):
             hlo_operands.append(op.backend_tensor)
             final_shapes.append(op.backend_tensor.shape)
        else:
             hlo_operands.append(op)
             final_shapes.append(op.shape)

    # Analyze pattern
    analysis = analyze_einsum_pattern(input_specs, output_spec, final_shapes)

    # Handle special cases for optimization
    if len(hlo_operands) == 1:
        return _einsum_unary(
            ctx, hlo_operands[0], input_specs[0], output_spec, analysis
        )
    elif len(hlo_operands) == 2:
        return _einsum_binary(
            ctx,
            hlo_operands[0],
            hlo_operands[1],
            input_specs[0],
            input_specs[1],
            output_spec,
            analysis,
        )
    else:
        # General case: reduce to binary operations
        return _einsum_nary(ctx, hlo_operands, input_specs, output_spec, analysis)





def _handle_repeated_indices(ctx, operand, spec: str):
    """Handle repeated indices in a single spec (e.g., 'ii') by taking diagonal."""
    from nkipy.core.tensor import NKIPyTensorRef
    from nkipy.core.backend.hlo import as_hlo_tensor
    import collections
    
    current_operand = operand
    if isinstance(current_operand, NKIPyTensorRef):
        current_operand = current_operand.backend_tensor
    current_spec = list(spec)
    
    while True:
        counts = collections.Counter(current_spec)
        repeated = [char for char, count in counts.items() if count > 1]
        
        if not repeated:
            break
            
        # Handle first repeated index
        idx = repeated[0]
        # Find first two positions
        positions = [i for i, char in enumerate(current_spec) if char == idx]
        pos1, pos2 = positions[0], positions[1]
        
        # Verify dimensions
        shape = current_operand.shape
        if shape[pos1] != shape[pos2]:
             raise ValueError(f"Repeated index {idx} has incompatible dimensions {shape[pos1]} and {shape[pos2]}")
             
        dim_size = shape[pos1]
        
        # Move pos1 and pos2 to the end
        # Permutation: All other indices + pos1 + pos2
        other_indices = [i for i in range(len(shape)) if i != pos1 and i != pos2]
        perm = other_indices + [pos1, pos2]
        
        current_operand = ctx.build_op(
            "transpose", [current_operand], 
            tuple(shape[i] for i in perm), 
            current_operand.dtype, 
            {"permutation": perm}
        )
        
        # Now shape is (..., N, N)
        # Create Identity Mask (N, N)
        # iota dimension 0
        iota0 = ctx.build_op("iota", [], (dim_size, dim_size), "int32", {"iota_dimension": 0})
        # iota dimension 1
        iota1 = ctx.build_op("iota", [], (dim_size, dim_size), "int32", {"iota_dimension": 1})
        
        # Mask = (iota0 == iota1)
        pred = ctx.build_op("compare", [iota0, iota1], (dim_size, dim_size), np.bool_, {"comparison_direction": "EQ"})
        
        # Convert to dtype
        mask = ctx.build_op("convert", [pred], (dim_size, dim_size), current_operand.dtype, {})
        
        # Broadcast mask to matches current_operand magnitude
        # Mask has shape (N, N). Operand has (..., N, N).
        # We broadcast mask to operands shape.
        # Dimensions to broadcast are the '...' ones (0 to len-3).
        # We map the mask dimensions [0, 1] to Result dimensions [rank-2, rank-1].
        
        rank = len(current_operand.shape)
        mask_broadcast = ctx.build_op(
            "broadcast", [mask], current_operand.shape, current_operand.dtype,
            {"broadcast_dimensions": [rank-2, rank-1]}
        )
        
        # Multiply
        masked_op = ctx.build_op("multiply", [current_operand, mask_broadcast], current_operand.shape, current_operand.dtype)
        
        # Reduce sum over the last dimension (pos2) - which is now at rank-1
        # Reduce dims: [rank-1]
        # Init value for add: 0.0
        init_val = as_hlo_tensor(ctx, 0.0, current_operand.dtype)
        
        reduced_shape = current_operand.shape[:-1]
        current_operand = ctx.build_op(
            "reduce", [masked_op, init_val], reduced_shape, current_operand.dtype,
            {"dimensions": [rank-1], "computation": "add"}
        )
        
        # Update spec
        # We removed the char at pos2 (which was moved to end).
        # The char at pos1 (which was moved to rank-2) is now at rank-1 (end).
        # The other chars are at 0 ... rank-2.
        # So new spec order is: [others] + [idx].
        
        new_spec_list = [current_spec[i] for i in other_indices] + [idx]
        current_spec = new_spec_list
        
    return current_operand, "".join(current_spec)



def _einsum_unary(ctx, operand, input_spec, output_spec, analysis):
    """Handle single-operand einsum (transpose, trace, reduction)."""
    from nkipy.core.backend.hlo import as_hlo_tensor
    from nkipy.core.tensor import NKIPyTensorRef

    # If output is empty, it's a full reduction
    if not output_spec:
        # Reduce all dimensions
        init_tensor = as_hlo_tensor(ctx, 0.0, operand.dtype)
        result = ctx.build_op(
            "reduce",
            [operand, init_tensor],
            (),  # scalar output
            operand.dtype,
            {
                "dimensions": list(range(len(operand.shape))),
                "computation": "add",
            },
        )
        return NKIPyTensorRef(result)

    # Use analysis to determine which dimensions to reduce
    # Contracting dims are those in input but not in output
    dims_to_reduce = []
    output_dims = []

    for i, idx in enumerate(input_spec):
        if idx in analysis["contracting_dims"]:
            dims_to_reduce.append(i)
        else:
            output_dims.append((idx, i, analysis["input_dims"][idx]))

    # Sort output dimensions by their order in output_spec
    # output_dims contains (idx, original_pos, size)
    # The 'operand' tensor currently has these dimensions in the order they appeared in input_spec (minus reduced ones).
    # XLA Reduce preserves relative order of remaining dimensions.
    
    current_indices = [idx for idx, _, _ in output_dims]
    
    # If there are dimensions to reduce
    if dims_to_reduce:
        # The shape expected by reduce op is the shape of the RESULT of reduction? 
        # Or the shape of the operands? 
        # Usually XLA build_op('reduce') might take output shape?
        # If so, it should match the input-ordered result (since no transpose happens during reduce).
        reduced_shape = tuple(size for _, _, size in output_dims)
        
        init_tensor = as_hlo_tensor(ctx, 0.0, operand.dtype)
        operand = ctx.build_op(
            "reduce",
            [operand, init_tensor],
            reduced_shape,
            operand.dtype,
            {
                "dimensions": dims_to_reduce,
                "computation": "add",
            },
        )

    # Check if we need to transpose to match output spec
    if current_indices != list(output_spec):
        # Build permutation
        # We want output to be output_spec.
        # Current tensor has dims in `current_indices` order.
        # We need to mapp `current_indices` -> `output_spec`.
        # Transpose perm[i] is the index in input that maps to output[i].
        
        try:
            perm = [current_indices.index(idx) for idx in output_spec]
        except ValueError as e:
            # Should not happen if analysis is correct
            raise RuntimeError(f"Internal einsum error: indices mismatch {current_indices} vs {output_spec}") from e
            
        transposed_shape = tuple(operand.shape[i] for i in perm)
        operand = ctx.build_op(
            "transpose",
            [operand],
            transposed_shape,
            operand.dtype,
            {"permutation": perm},
        )

    return NKIPyTensorRef(operand)


def _einsum_binary(ctx, lhs, rhs, lhs_spec, rhs_spec, output_spec, analysis):
    """Handle two-operand einsum (matmul, outer product, etc.)."""
    from nkipy.core.tensor import NKIPyTensorRef

    # Find contracting, batch, and free dimensions
    lhs_indices = list(lhs_spec)
    rhs_indices = list(rhs_spec)

    contracting_dims = analysis["contracting_dims"]

    # Identify dimension roles for each operand
    lhs_contracting = [
        i for i, idx in enumerate(lhs_indices) if idx in contracting_dims
    ]
    rhs_contracting = [
        i for i, idx in enumerate(rhs_indices) if idx in contracting_dims
    ]

    lhs_batch = [
        i for i, idx in enumerate(lhs_indices) if idx in analysis["batch_dims"]
    ]
    rhs_batch = [
        i for i, idx in enumerate(rhs_indices) if idx in analysis["batch_dims"]
    ]

    # Compute output shape
    output_shape = tuple(analysis["input_dims"][idx] for idx in output_spec)

    # Use dot_general for contraction
    if contracting_dims:
        result = ctx.build_op(
            "dot",
            [lhs, rhs],
            output_shape,
            lhs.dtype,
            {
                "lhs_contracting_dimensions": lhs_contracting,
                "rhs_contracting_dimensions": rhs_contracting,
                "lhs_batch_dimensions": lhs_batch,
                "rhs_batch_dimensions": rhs_batch,
            },
        )
    else:
        # No contraction - it's an outer product or broadcast multiply
        # Reshape both operands to have compatible shapes, then multiply
        # For now, use broadcasting via reshape + multiply

        # Determine the position of each operand's dimensions in output
        lhs_out_positions = [output_spec.index(idx) for idx in lhs_indices]
        rhs_out_positions = [output_spec.index(idx) for idx in rhs_indices]

        # Reshape lhs: add dimensions at positions not in lhs
        new_lhs_shape = [1] * len(output_shape)
        for i, pos in enumerate(lhs_out_positions):
            new_lhs_shape[pos] = lhs.shape[i]
        lhs_reshaped = ctx.build_op("reshape", [lhs], tuple(new_lhs_shape), lhs.dtype)

        # Broadcast lhs to output shape
        lhs_broadcasted = ctx.build_op(
            "broadcast",
            [lhs_reshaped],
            output_shape,
            lhs.dtype,
            {"broadcast_dimensions": lhs_out_positions},
        )

        # Reshape rhs similarly
        new_rhs_shape = [1] * len(output_shape)
        for i, pos in enumerate(rhs_out_positions):
            new_rhs_shape[pos] = rhs.shape[i]
        rhs_reshaped = ctx.build_op("reshape", [rhs], tuple(new_rhs_shape), rhs.dtype)

        # Broadcast rhs to output shape
        rhs_broadcasted = ctx.build_op(
            "broadcast",
            [rhs_reshaped],
            output_shape,
            rhs.dtype,
            {"broadcast_dimensions": rhs_out_positions},
        )

        # Multiply
        result = ctx.build_op(
            "multiply", [lhs_broadcasted, rhs_broadcasted], output_shape, lhs.dtype
        )

    return NKIPyTensorRef(result)


def _einsum_nary(ctx, operands, input_specs, output_spec, analysis):
    """Handle n-ary einsum by reducing to binary operations."""
    # Chain binary operations left-to-right
    result = operands[0]
    current_spec = input_specs[0]

    for i in range(1, len(operands)):
        # Determine intermediate output spec (union of remaining indices)
        remaining_specs = input_specs[i:]
        remaining_indices = set(output_spec)
        for spec in remaining_specs:
            remaining_indices.update(spec)

        # Build intermediate spec in canonical order
        intermediate_spec = "".join(
            idx
            for idx in current_spec + input_specs[i]
            if idx in remaining_indices
            and idx
            not in "".join(
                idx for idx in current_spec + input_specs[i] if idx in remaining_indices
            )[: current_spec.index(idx) if idx in current_spec else len(current_spec)]
        )

        # Perform binary einsum
        shapes = [result.shape, operands[i].shape]
        sub_analysis = analyze_einsum_pattern(
            [current_spec, input_specs[i]],
            intermediate_spec if i < len(operands) - 1 else output_spec,
            shapes,
        )

        result_ref = _einsum_binary(
            ctx,
            result,
            operands[i],
            current_spec,
            input_specs[i],
            intermediate_spec if i < len(operands) - 1 else output_spec,
            sub_analysis,
        )
        result = result_ref.backend_tensor
        current_spec = intermediate_spec if i < len(operands) - 1 else output_spec

    from nkipy.core.tensor import NKIPyTensorRef

    return NKIPyTensorRef(result)
