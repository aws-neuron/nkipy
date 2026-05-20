#ifndef NKIPY_TRANSFORM_OPS
#define NKIPY_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PromoteTensorOp
//===----------------------------------------------------------------------===//

def Nkipy_PromoteTensorOp : Op<Transform_Dialect, "nkipy.promote_tensor",
                         [DeclareOpInterfaceMethods<TransformOpInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Request a tensor value to live in a specific memory space "
                "after bufferization";
  let description = [{
    Requests that a tensor value lives in a specific memory space for its
    lifetime. This is achieved by allocating a new tensor in the desired
    memory space with `bufferization.alloc_tensor` and optionally materializing
    the source value into that allocation with
    `bufferization.materialize_in_destination`. All uses of the original value
    are then redirected to the promoted value.

    The generated code for promoting tensor value %0 resembles the following:

      %1 = bufferization.alloc_tensor(<dynamic dims of %0>)
           { memory_space = memory_space }
      // Note: the materialization is omitted if %0 is never read and is only
      // written into (i.e., it behaves as a result tensor).
      %2 = bufferization.materialize_in_destination %0 in %1
      // ...
      <all users of %0 now use %2 instead>

    Deallocation is not handled by this transform.

    Return modes:
    - Produces a silenceable failure if the given handle does not point to
      tensor-typed values.
    - Succeeds otherwise and returns a handle to the promoted value(s), i.e.,
      the result of materialization if present and the allocation otherwise.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$tensor,
      OptionalAttr<AnyAttr>:$memory_space);
  let results = (outs TransformValueHandleTypeInterface:$promoted);

  let assemblyFormat =
      "(`to` $memory_space^)? $tensor attr-dict `:` functional-type($tensor, $promoted)";
}

#endif // NKIPY_TRANSFORM_OPS
