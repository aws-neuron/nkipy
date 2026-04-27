#ifndef NKIPY_BINDINGS_PYTHON_IRMODULES_H
#define NKIPY_BINDINGS_PYTHON_IRMODULES_H

// #include "NanobindUtils.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace mlir {
namespace python {

// void populateNkipyIRTypes(nanobind::module_ &m);
void populateNkipyAttributes(nanobind::module_ &m);

} // namespace python
} // namespace mlir

#endif // NKIPY_BINDINGS_PYTHON_IRMODULES_H
