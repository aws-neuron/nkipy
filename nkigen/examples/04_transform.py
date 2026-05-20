"""
Example demonstrating how the IR should be knob() API for transformation annotations.
"""

import numpy as np
from nkigen import trace
from nkigen.apis import knob

from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import scf, pdl, func, arith, linalg
from mlir.dialects.transform import (
    get_parent_op,
    apply_patterns_canonicalization,
    apply_cse,
    any_op_t,
)
from mlir.dialects.transform.structured import structured_match
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.extras import types as T
from mlir.dialects.builtin import module, ModuleOp


def construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


@construct_and_print_in_module
def test_named_sequence(module_):

    # func entry (traced function in nkipy tensorizer)
    @func.func()
    def loop_unroll_op():
        for i in scf.for_(0, 42, 5):
            v = arith.addi(i, i)
            scf.yield_([])

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            m = structured_match(any_op_t(), target, ops=["arith.addi"])
            loop = get_parent_op(pdl.op_t(), m, op_name="scf.for")
            loop_unroll(loop, 4)

    # The identifier (name) of the function becomes the Operation
    assert isinstance(mod.opview, ModuleOp)

    print(module_)

    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print(module_)
