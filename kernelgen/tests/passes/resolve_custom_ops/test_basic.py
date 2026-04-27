"""
Tests for the custom-op resolution step.

The standalone C++ `--resolve-custom-ops` pass was removed during
open-sourcing and re-implemented as `_resolve_custom_ops()` inside the
Python `linalg_to_nisa_py` phase, so these tests drive that function
directly against small hand-written fixtures.  Each test:

1.  Parses an MLIR module (with a `nkipy.custom_op_bodies` dict, a
    body-less `func.func private` decl, and a call site) into the
    NKI-wheel MLIR context.
2.  Runs `_resolve_custom_ops`.
3.  Asserts the call is gone, the decl is gone, the module attribute is
    gone, and the inlined ops are present.

Run with: python -m pytest tests/passes/resolve_custom_ops/test_basic.py -v
"""

import pytest

from nki.compiler._internal import ir as nk_ir
from nki.compiler._internal._mlir_libs import _nki

from nkipy_kernelgen.transforms.linalg_to_nisa_py import _resolve_custom_ops
from passes.pass_utils import run_filecheck


def _escape_mlir_string(s: str) -> str:
    """Escape a string for embedding in an MLIR string attribute."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _run_resolve(mlir_input: str) -> str:
    """Parse in a fresh NKI context, run _resolve_custom_ops, return IR text."""
    ctx = nk_ir.Context()
    _nki.register_all_dialects(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = nk_ir.Module.parse(mlir_input, ctx)
        _resolve_custom_ops(module, ctx)
        return str(module)


# ============================================================================
# Basic resolution: single custom op with output-as-argument convention
# ============================================================================


def test_resolve_single_custom_op():
    """
    Resolve a single custom op declaration + call site.

    The NISA body uses output-as-argument convention: trailing args are outputs.
    After resolution, the body is inlined at the call site with an alloc for output.
    """
    nisa_body = _escape_mlir_string(
        "module attributes {nisa.target = #nisa.target<trn2>} {"
        "  func.func @my_op("
        "    %arg0: memref<128x128xf32, #nisa.mem<shared_hbm>>,"
        "    %arg1: memref<128x128xf32, #nisa.mem<shared_hbm>>"
        '  ) attributes {nki.output_names = ["output"]} {'
        "    return"
        "  }"
        "}"
    )

    mlir_input = f'''
module attributes {{
  nkipy.custom_op_bodies = {{
    "__custom_op__my_op" = "{nisa_body}"
  }}
}} {{
  func.func @main_kernel(
    %arg0: memref<128x128xf32, #nisa.mem<shared_hbm>>
  ) -> memref<128x128xf32, #nisa.mem<shared_hbm>> {{
    %result = func.call @__custom_op__my_op(%arg0)
      : (memref<128x128xf32, #nisa.mem<shared_hbm>>) -> memref<128x128xf32, #nisa.mem<shared_hbm>>
    return %result : memref<128x128xf32, #nisa.mem<shared_hbm>>
  }}

  func.func private @__custom_op__my_op(
    memref<128x128xf32, #nisa.mem<shared_hbm>>
  ) -> memref<128x128xf32, #nisa.mem<shared_hbm>>
    attributes {{nkipy.custom_op}}
}}
'''

    result = _run_resolve(mlir_input)

    # Body is inlined: alloc for output, no call instruction, no declaration
    check_patterns = """
    CHECK: func.func @main_kernel
    CHECK: memref.alloc
    CHECK: return
    CHECK-NOT: call @__custom_op__my_op
    CHECK-NOT: func.func private @__custom_op__my_op
    CHECK-NOT: nkipy.custom_op_bodies
    CHECK-NOT: nkipy.custom_op
    """
    run_filecheck(result, check_patterns)


# ============================================================================
# No custom ops: resolve should be a no-op
# ============================================================================


def test_no_custom_ops_is_noop():
    """When there's no nkipy.custom_op_bodies attribute, resolve is a no-op."""
    mlir_input = """
module {
  func.func @main_kernel(
    %arg0: memref<128x128xf32, #nisa.mem<shared_hbm>>
  ) -> memref<128x128xf32, #nisa.mem<shared_hbm>> {
    return %arg0 : memref<128x128xf32, #nisa.mem<shared_hbm>>
  }
}
"""
    result = _run_resolve(mlir_input)

    check_patterns = """
    CHECK: func.func @main_kernel
    CHECK: return
    CHECK-NOT: nkipy.custom_op_bodies
    """
    run_filecheck(result, check_patterns)


# ============================================================================
# Multiple call sites for the same custom op
# ============================================================================


def test_multiple_call_sites():
    """
    When the same custom op is called multiple times,
    each call site gets its own inlined body with separate allocs.
    """
    nisa_body = _escape_mlir_string(
        "module attributes {nisa.target = #nisa.target<trn2>} {"
        "  func.func @my_op("
        "    %arg0: memref<64x64xf32, #nisa.mem<shared_hbm>>,"
        "    %arg1: memref<64x64xf32, #nisa.mem<shared_hbm>>"
        '  ) attributes {nki.output_names = ["output"]} {'
        "    return"
        "  }"
        "}"
    )

    mlir_input = f'''
module attributes {{
  nkipy.custom_op_bodies = {{
    "__custom_op__my_op" = "{nisa_body}"
  }}
}} {{
  func.func @main_kernel(
    %arg0: memref<64x64xf32, #nisa.mem<shared_hbm>>,
    %arg1: memref<64x64xf32, #nisa.mem<shared_hbm>>
  ) -> memref<64x64xf32, #nisa.mem<shared_hbm>> {{
    %r0 = func.call @__custom_op__my_op(%arg0)
      : (memref<64x64xf32, #nisa.mem<shared_hbm>>) -> memref<64x64xf32, #nisa.mem<shared_hbm>>
    %r1 = func.call @__custom_op__my_op(%arg1)
      : (memref<64x64xf32, #nisa.mem<shared_hbm>>) -> memref<64x64xf32, #nisa.mem<shared_hbm>>
    return %r1 : memref<64x64xf32, #nisa.mem<shared_hbm>>
  }}

  func.func private @__custom_op__my_op(
    memref<64x64xf32, #nisa.mem<shared_hbm>>
  ) -> memref<64x64xf32, #nisa.mem<shared_hbm>>
    attributes {{nkipy.custom_op}}
}}
'''

    result = _run_resolve(mlir_input)

    # Two inlined bodies = two allocs, no call instructions
    check_patterns = """
    CHECK: func.func @main_kernel
    CHECK: memref.alloc
    CHECK: memref.alloc
    CHECK: return
    CHECK-NOT: call @__custom_op__my_op
    CHECK-NOT: func.func private @__custom_op__my_op
    """
    run_filecheck(result, check_patterns)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
