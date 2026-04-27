#ifndef NKIPY_DIALECT
#define NKIPY_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"


def Nkipy_Dialect : Dialect {
    let name = "nkipy";
    let summary = "A nkipy out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let useDefaultTypePrinterParser = 1;
    let cppNamespace = "::mlir::nkipy";
}

class Nkipy_Op<string mnemonic, list<Trait> traits = []> :
        Op<Nkipy_Dialect, mnemonic, traits>;

// class Nkipy_Type<string mnemonic, list<Trait> traits = []> : 
//         TypeDef<Nkipy_Dialect, mnemonic, traits>;

class Nkipy_Attr<string mnemonic, list<Trait> traits = []> : 
        AttrDef<Nkipy_Dialect, mnemonic, traits>;

#endif // NKIPY_DIALECT
