
#ifndef NKIPY_C_DIALECT__H
#define NKIPY_C_DIALECT__H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(NKIPY, nkipy);

#ifdef __cplusplus
}
#endif

#endif // NKIPY_C_DIALECT__H