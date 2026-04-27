#ifndef NKIPY_ATTRS
#define NKIPY_ATTRS

include "nkipy/Dialect/NkipyDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

// https://mlir.llvm.org/docs/OpDefinitions/#enum-attributes
// Memory space enumeration for annotate operation.
//
// IMPORTANT: All values must be NON-ZERO. MemSpaceEnumAttr is built on
// IntegerAttr, and mlir::MemRefType::get() treats a memorySpace attribute
// whose integer value is 0 as "no memory space" — it is silently dropped
// during uniquing. That would make `root.setType(memref<..., Hbm>)` a no-op
// for HBM=0, which in turn breaks the fixpoint loop in
// AnnotateMemorySpace::inferHbmForCopySources (it keeps re-applying HBM
// every iteration and never converges).
// Values 1..5 keep all memspaces representable on memref types.
def Hbm: I32EnumAttrCase<"Hbm", 1>;
def Psum: I32EnumAttrCase<"Psum", 2>;
def Sbuf: I32EnumAttrCase<"Sbuf", 3>;
def SharedHbm: I32EnumAttrCase<"SharedHbm", 4>;
def Constant: I32EnumAttrCase<"Constant", 5>;  // Scalar broadcast constants (marker, not real memory)

def MemSpaceEnum: I32EnumAttr<"MemSpaceEnum",
        "Memory space enumeration",
        [Hbm, Psum, Sbuf, SharedHbm, Constant]> {
    let cppNamespace = "mlir::nkipy";
    let stringToSymbolFnName = "ConvertToMemSpaceEnum";
    let symbolToStringFnName = "ConvertToMemSpaceString";
}

#endif // NKIPY_ATTRS
