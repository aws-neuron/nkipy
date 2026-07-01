#ifndef NKIPY_OPS_H
#define NKIPY_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyAttrs.h"

#define GET_OP_CLASSES
#include "nkipy/Dialect/NkipyOps.h.inc"

#endif // NKIPY_OPS_H