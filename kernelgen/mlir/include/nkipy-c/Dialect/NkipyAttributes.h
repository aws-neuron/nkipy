#ifndef NKIPY_MLIR_C_ATTRIBUTES__H
#define NKIPY_MLIR_C_ATTRIBUTES__H

#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IntegerSet.h"

#ifdef __cplusplus
extern "C" {
#endif

// MLIR_CAPI_EXPORTED bool mlirAttributeIsAIntegerSet(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirIntegerSetAttrGet(MlirIntegerSet set);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAMemSpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirMemSpaceGet(MlirContext ctx,
                                                  MlirAttribute space);

#ifdef __cplusplus
}
#endif

#endif // NKIPY_MLIR_C_ATTRIBUTES__H
