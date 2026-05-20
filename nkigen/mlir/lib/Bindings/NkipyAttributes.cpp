#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "nkipy-c/Dialect/NkipyAttributes.h"
#include "nkipy/Bindings/NkipyModule.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

using namespace mlir;
using namespace mlir::python;

namespace nanobind {
namespace detail {

/// Casts object <-> MlirIntegerSet.
template <> struct type_caster<MlirIntegerSet> {
  NB_TYPE_CASTER(MlirIntegerSet, const_name("MlirIntegerSet"));
  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    nb::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToIntegerSet(capsule.ptr());
    if (mlirIntegerSetIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle from_cpp(MlirIntegerSet v, rv_policy, cleanup_list *) noexcept {
    nb::object capsule =
        nb::steal<nb::object>(mlirPythonIntegerSetToCapsule(v));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("IntegerSet")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

} // namespace detail
} // namespace nanobind

void mlir::python::populateNkipyAttributes(nb::module_ &m) {
  mlir_attribute_subclass(m, "IntegerSetAttr", mlirAttributeIsAIntegerSet)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirIntegerSet IntegerSet, MlirContext ctx) {
            return cls(mlirIntegerSetAttrGet(IntegerSet));
          },
          nb::arg("cls"), nb::arg("integer_set"),
          nb::arg("context").none() = nb::none(),
          "Gets an attribute wrapping an IntegerSet.");

  mlir_attribute_subclass(m, "MemSpaceEnum", mlirAttributeIsAMemSpace)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirAttribute space, MlirContext ctx) {
            return cls(mlirMemSpaceGet(ctx, space));
          },
          nb::arg("cls"), nb::arg("space"), nb::arg("context").none() = nb::none(),
          "Gets an attribute wrapping a memory space.");

}
