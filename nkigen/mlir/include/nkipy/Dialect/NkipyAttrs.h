#ifndef NKIPY_ATTRS_H
#define NKIPY_ATTRS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "nkipy/Dialect/NkipyEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "nkipy/Dialect/NkipyAttrs.h.inc"

#endif // NKIPY_ATTRS_H