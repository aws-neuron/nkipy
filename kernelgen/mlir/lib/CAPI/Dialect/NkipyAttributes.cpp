#include "nkipy-c/Dialect/NkipyAttributes.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;
using namespace nkipy;

bool mlirAttributeIsAMemSpace(MlirAttribute attr) {
  return mlir::isa<MemSpaceEnumAttr>(unwrap(attr));
}

MlirAttribute mlirMemSpaceGet(MlirContext ctx, MlirAttribute space) {
  auto attr = llvm::cast<mlir::IntegerAttr>(unwrap(space));
  MemSpaceEnum spaceEnum =
      static_cast<MemSpaceEnum>(attr.getInt());
  return wrap(MemSpaceEnumAttr::get(unwrap(ctx), spaceEnum));
}
