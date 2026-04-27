#include "nkipy-c/Dialect/Dialects.h"

#include "nkipy/Dialect/NkipyDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Nkipy, nkipy, mlir::nkipy::NkipyDialect)