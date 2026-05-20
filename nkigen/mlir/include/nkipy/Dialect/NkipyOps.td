#ifndef NKIPY_OPS
#define NKIPY_OPS

include "nkipy/Dialect/NkipyDialect.td"
include "nkipy/Dialect/NkipyAttrs.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"


def Nkipy_LayoutOp : Nkipy_Op<"layout", [
    DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferizesToMemoryRead", "bufferizesToMemoryWrite",
       "getAliasingValues", "bufferize"]>]>
{
    let summary = "layout";
    let description = [{
        nkipy_d.layout(target, mem_space=None, partition_dim=None, tile_size=None)

        Describe the memory layout of a Tensor: its memory space, partition
        dimension, and physical tile shape.

        Parameters
        * target (Tensor) - The tensor to annotate
        * mem_space ({Hbm, Psum, Sbuf, SharedHbm}, optional) - The memory space for the tensor
        * partition_dim (int, optional) - The dimension to be partitioned (must be 0 for NISA)
        * tile_size (array<int>, optional) - The physical tile sizes for each dimension.
          Must have exactly the same number of elements as the tensor rank.
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$target,
                     OptionalAttr<MemSpaceEnum>:$mem_space,
                     OptionalAttr<UI32Attr>:$partition_dim,
                     OptionalAttr<DenseI64ArrayAttr>:$tile_size);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `)` attr-dict
    }];
}

def Nkipy_TileOp : Nkipy_Op<"tile_op", [
    DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferizesToMemoryRead", "bufferizesToMemoryWrite",
       "getAliasingValues", "bufferize"]>]>
{
    let summary = "tile_op";
    let description = [{
        nkipy_d.tile_op(target, loop_tile_size=None)

        Describe the loop-tiling hints for an op that produces this tensor.

        Parameters
        * target (Tensor) - The tensor produced by the op to be tiled
        * loop_tile_size (array<int>, optional) - One tile size per iterator
          of the producing op, in linalg iterator order (no reordering — entry
          i applies to iterator i). For elementwise ops this matches the
          output rank. For a reduction on input [M, N] reducing axis -1, it is
          [M_t, N_t] (iterators [parallel, reduction]). For matmul A[M,K] x
          B[K,N] -> C[M,N], it is [M_t, N_t, K_t] (iterators [parallel,
          parallel, reduction]).
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$target,
                     OptionalAttr<DenseI64ArrayAttr>:$loop_tile_size);
    let results = (outs );
    let assemblyFormat = [{
        `(` $target `:` type($target) `)` attr-dict
    }];
}

def Nkipy_FuseOp : Nkipy_Op<"fuse_op">
{
    let summary = "fuse_op";
    let description = [{
        nkipy_d.fuse_op(targets...)

        Hint that the scf.for loops producing the listed target tensors
        should be fused into a single loop.  Runs after tiling, so the
        loops must have matching lower/upper/step bounds — i.e. the
        producer ops must already be tiled with identical loop_tile_size.
        At least two targets are required.

        Parameters
        * targets (Tensors) - The tensors whose producing loops to fuse.
    }];

    let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$targets);
    let results = (outs );
    let assemblyFormat = [{
        `(` $targets `:` type($targets) `)` attr-dict
    }];
    let hasVerifier = 1;
}

def Nkipy_YieldOp : Nkipy_Op<"yield", [Pure, ReturnLike, Terminator]>
{
    let summary = "Yield values from an nkipy region";
    let description = [{
        Terminates the reference implementation region of an nkipy op.
        Yields the computed result values back to the enclosing op.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def Nkipy_GatherOp : Nkipy_Op<"gather", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferizesToMemoryRead", "bufferizesToMemoryWrite",
       "getAliasingValues", "bufferize"]>,
    DeclareOpInterfaceMethods<TilingInterface,
      ["getLoopIteratorTypes", "getIterationDomain",
       "getTiledImplementation", "getResultTilePosition"]>]>
{
    let summary = "Gather rows from a tensor using an index tensor";
    let description = [{
        Gathers rows from `source` at positions given by `indices` along axis 0.

        Semantics: result[i, ...] = source[indices[i], ...]

        This op is emitted during tracing for:
          - table[indices]        (fancy indexing)
          - np.take(table, indices, axis=0)

        It is lowered to nisa.dma_copy_indirect during the linalg-to-nisa pass.

        The source must be a 2D tensor/memref. The indices must be a 1D integer
        tensor/memref. The result shape is [indices.shape[0], source.shape[1]].

        The `output` operand is the DPS init (destination-passing style): an
        uninitialized buffer of the same shape as the result.  Bufferization
        allocates the backing storage via tensor.empty -> memref.alloc.

        The op carries an optional reference implementation region containing
        standard linalg/tensor ops. This region is used for LLVM CPU simulation
        (inlined before JIT) and ignored during NISA lowering.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyTensor, AnyMemRef]>:$source,
        AnyTypeOf<[AnyTensor, AnyMemRef]>:$indices,
        AnyTypeOf<[AnyTensor, AnyMemRef]>:$output
    );
    let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$result);
    let regions = (region AnyRegion:$reference_impl);

    let assemblyFormat = [{
        `(` $source `,` $indices `)` `outs` `(` $output `)` ($reference_impl^)? attr-dict `:` `(` type($source) `,` type($indices) `)` `outs` `(` type($output) `)` `->` type($result)
    }];
}

#endif // NKIPY_OPS
