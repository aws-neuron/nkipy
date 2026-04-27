"""
Knob API for annotating tensors with transformation hints.

This module provides a knob() function similar to OpenMP pragmas in C++,
allowing users to annotate tensors with transformation directives that
get recorded as MLIR attributes on operations or injected as custom ops.
"""

from typing import Union, Any, Optional, List
from mlir import ir
from .traced_array import TracedArray
from nkipy_kernelgen._mlir.dialects import nkipy as nkipy_d


def knob(
    tensor: Union[TracedArray, Any],
    partition_dim: Optional[int] = None,
    mem_space: Optional[str] = None,
    tile_size: Optional[List[int]] = None,
    reduction_tile: Optional[List[int]] = None,
) -> Union[TracedArray, Any]:
    """
    Annotate a tensor with transformation hints.

    This function acts like OpenMP pragmas in C++, marking tensors with
    transformation directives that get recorded as MLIR nkipy.annotate operations.

    Args:
        tensor: The tensor to annotate (TracedArray or regular array)
        partition_dim: Dimension to partition (int, must be 0 for NISA compatibility)
        mem_space: Memory space placement (must be "Hbm", "Psum", "Sbuf", or "SharedHbm", optional)
        tile_size: Tile sizes for each dimension (list of ints, optional).
            Must have exactly the same number of elements as the tensor rank.
            E.g., for a 3D tensor [16, 128, 512], use tile_size=[1, 128, 128].
        reduction_tile: Tile sizes for reduction dimensions (list of ints, optional).
            Used for contraction ops like matmul where the iteration space has more
            dimensions than the output tensor. For matmul C=A@B, this is the K tile.
            E.g., tile_size=[128, 128] for output dims + reduction_tile=[128] for K.

    Returns:
        The same tensor (pass-through), but with annotate op injected if parameters specified

    Raises:
        ValueError: If partition_dim is >= tensor rank, mem_space is invalid,
            tile_size doesn't match tensor rank, or reduction_tile has negative values.

    Examples:
        # Specify only memory space
        tensor = knob(tensor, mem_space="Hbm")

        # Specify only partition dimension
        tensor = knob(tensor, partition_dim=1)

        # Specify memory space and partition dimension
        tensor = knob(tensor, partition_dim=1, mem_space="Sbuf")

        # Specify tile size for a 2D tensor [256, 256]
        tensor = knob(tensor, tile_size=[128, 128])

        # Specify tile size for a 3D tensor [16, 128, 512]
        tensor = knob(tensor, tile_size=[1, 128, 128])

        # Matmul with separate reduction tile: C[M,N] = A[M,K] @ B[K,N]
        output = knob(output, mem_space="Psum", tile_size=[128, 128], reduction_tile=[128])
    """
    # If not a TracedArray, just return as-is (for regular NumPy execution)
    if not isinstance(tensor, TracedArray):
        return tensor

    # If no parameters are specified, just return (no-op)
    if (
        mem_space is None
        and partition_dim is None
        and tile_size is None
        and reduction_tile is None
    ):
        return tensor

    # Get the MLIR value from the traced array
    value = tensor.value

    # Get the operation that defines this value
    defining_op = value.owner

    if defining_op is None:
        # Value is a block argument, cannot annotate
        return tensor

    # Normalize scalar tile_size/reduction_tile to lists
    if isinstance(tile_size, int):
        tile_size = [tile_size]
    if isinstance(reduction_tile, int):
        reduction_tile = [reduction_tile]

    # Validate mem_space
    if mem_space is not None:
        valid_mem_spaces = {"Hbm", "Psum", "Sbuf", "SharedHbm"}
        if mem_space not in valid_mem_spaces:
            raise ValueError(
                f"Invalid mem_space '{mem_space}'. Must be one of: {valid_mem_spaces}"
            )

    # Validate partition_dim against tensor rank
    if partition_dim is not None:
        # Get the tensor type from the MLIR value
        tensor_type = value.type
        if hasattr(tensor_type, "shape"):
            rank = len(tensor_type.shape)
            if partition_dim >= rank:
                raise ValueError(
                    f"partition_dim {partition_dim} must be less than tensor rank {rank}"
                )
        # Also validate it's non-negative
        if partition_dim < 0:
            raise ValueError(f"partition_dim must be non-negative, got {partition_dim}")

    # Validate tile_size against tensor rank
    if tile_size is not None:
        tensor_type = value.type
        if hasattr(tensor_type, "shape"):
            rank = len(tensor_type.shape)
            # tile_size must either match the full tensor rank, or when
            # reduction_tile is also provided, tile_size + reduction_tile
            # together must cover all dimensions (e.g. for reductions with
            # keepdims=True where tile_size covers non-reduction dims and
            # reduction_tile covers reduction dims).
            n_reduction = len(reduction_tile) if reduction_tile is not None else 0
            if len(tile_size) != rank and len(tile_size) + n_reduction != rank:
                raise ValueError(
                    f"tile_size has {len(tile_size)} elements but tensor has rank {rank}; "
                    f"tile_size must have exactly one element per dimension"
                )
        # Validate non-negative
        if any(t <= 0 for t in tile_size):
            raise ValueError(f"tile_size values must be positive, got {tile_size}")

    # Validate reduction_tile
    if reduction_tile is not None:
        if any(t <= 0 for t in reduction_tile):
            raise ValueError(
                f"reduction_tile values must be positive, got {reduction_tile}"
            )

    # Inject nkipy.annotate op when at least one parameter is specified
    _inject_annotate_op(value, mem_space, partition_dim, tile_size, reduction_tile)

    # Return the tensor unchanged (knob is just an annotation)
    return tensor


def _inject_annotate_op(
    value: ir.Value,
    mem_space: Optional[str],
    partition_dim: Optional[int],
    tile_size: Optional[List[int]],
    reduction_tile: Optional[List[int]] = None,
) -> None:
    """
    Inject a nkipy.annotate op into the IR after the tensor's value.

    Args:
        value: The MLIR value to annotate
        mem_space: Memory space (Hbm, Psum, Sbuf, or SharedHbm), optional
        partition_dim: Dimension to partition, optional
        tile_size: Tile sizes for each dimension, optional
        reduction_tile: Tile sizes for reduction dimensions, optional
    """
    # Get the defining operation for location info
    defining_op = value.owner
    if defining_op is None:
        return

    loc = defining_op.location

    # Build attributes dict with only specified parameters
    if mem_space is not None:
        # Map mem_space string to enum value.
        # Values MUST match mlir/include/nkipy/Dialect/NkipyAttrs.td.
        # Zero is intentionally reserved: MemRefType::get drops an
        # IntegerAttr(0) memorySpace, so zero-valued enum cases cannot be
        # attached to a memref. See the comment in NkipyAttrs.td.
        mem_space_map = {
            "Hbm": 1,
            "Psum": 2,
            "Sbuf": 3,
            "SharedHbm": 4,
        }
        mem_space_value = mem_space_map[mem_space]
        mem_space_attr = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32), mem_space_value
        )
        mem_space = mem_space_attr

    if partition_dim is not None:
        partition_dim_attr = ir.IntegerAttr.get(
            ir.IntegerType.get_unsigned(32), partition_dim
        )
        partition_dim = partition_dim_attr

    if tile_size is not None:
        # Convert to DenseI64ArrayAttr as defined in NkipyOps.td
        tile_size_attr = ir.DenseI64ArrayAttr.get(tile_size)
        tile_size = tile_size_attr

    if reduction_tile is not None:
        # Convert to DenseI64ArrayAttr as defined in NkipyOps.td
        reduction_tile_attr = ir.DenseI64ArrayAttr.get(reduction_tile)
        reduction_tile = reduction_tile_attr

    # Simply create the operation at the current insertion point
    # Since we're being called during tracing, the insertion point is already
    # set correctly to insert after the current operation
    # Note: The context has allow_unregistered_dialects=True, so we can create
    # nkipy.annotate ops directly without needing to register the dialect
    nkipy_d.AnnotateOp(
        target=value,
        mem_space=mem_space,
        partition_dim=partition_dim,
        tile_size=tile_size,
        reduction_tile=reduction_tile,
        loc=loc,
    )
