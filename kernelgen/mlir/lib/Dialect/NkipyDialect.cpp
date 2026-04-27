#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ADT/TypeSwitch.h"

#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyOps.h"

using namespace mlir;
using namespace mlir::nkipy;

#include "nkipy/Dialect/NkipyDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "nkipy/Dialect/NkipyAttrs.cpp.inc"

#include "nkipy/Dialect/NkipyEnums.cpp.inc"


void NkipyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nkipy/Dialect/NkipyOps.cpp.inc"
      >();
  addAttributes< 
#define GET_ATTRDEF_LIST
#include "nkipy/Dialect/NkipyAttrs.cpp.inc"
      >();
}

mlir::Type NkipyDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(),
                   "nkipy dialect has no custom types");
  return mlir::Type();
}

void NkipyDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm_unreachable("nkipy dialect has no custom types");
}