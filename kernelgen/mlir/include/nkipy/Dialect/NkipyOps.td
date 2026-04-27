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


def Nkipy_AnnotateOp : Nkipy_Op<"annotate", [
    DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferizesToMemoryRead", "bufferizesToMemoryWrite",
       "getAliasingValues", "bufferize"]>]>
{
    let summary = "annotate";
    let description = [{
        nkipy_d.annotate(target, mem_space=None, partition_dim=None, tile_size=None, reduction_tile=None)

        Annotate a Tensor with memory space and/or partition dimension information.

        Users can optionally specify the memory space where the tensor should reside
        (Hbm, Psum, Sbuf, SharedHbm), the partition dimension, and tile sizes.

        Parameters
        * target (Tensor) - The tensor to be annotated
        * mem_space ({Hbm, Psum, Sbuf, SharedHbm}, optional) - The memory space for the tensor
        * partition_dim (int, optional) - The dimension to be partitioned (must be 0 for NISA)
        * tile_size (array<int>, optional) - The tile sizes for each dimension.
          Must have exactly the same number of elements as the tensor rank.
        * reduction_tile (array<int>, optional) - Tile sizes for reduction dimensions.
          Used for contraction ops (e.g., matmul) where the iteration space has more
          dimensions than the output tensor. For matmul, this is the K tile size.
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$target,
                     OptionalAttr<MemSpaceEnum>:$mem_space,
                     OptionalAttr<UI32Attr>:$partition_dim,
                     OptionalAttr<DenseI64ArrayAttr>:$tile_size,
                     OptionalAttr<DenseI64ArrayAttr>:$reduction_tile);
    let results = (outs );
    // All optional attributes go in attr-dict so they can appear in any order
    let assemblyFormat = [{
        `(` $target `:` type($target) `)` attr-dict
    }];
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
