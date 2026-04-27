# RUN: %PYTHON %s | FileCheck %s

import numpy as np
from mlir.ir import Context, Location

from nkipy_kernelgen import trace
from nkipy_kernelgen.apis import knob


def run(f):
    """Simple test runner: prints a label, runs the test."""
    print(f"\nTEST: {f.__name__}")
    f()
    print(f"TEST_END: {f.__name__}")
    return f


# CHECK-LABEL: TEST: test_knob_partition_dim_only
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x4xf32>, 0)
# CHECK: TEST_END: test_knob_partition_dim_only
@run
def test_knob_partition_dim_only():
    """Test that knob with only partition_dim injects annotate op correctly."""
    
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_with_partition(A, B):
        temp = np.add(A, B)
        temp = knob(temp, partition_dim=0)
        result = np.multiply(temp, 2.0)
        return result
    
    module = func_with_partition.to_mlir()
    print(module)


# CHECK-LABEL: TEST: test_knob_mem_space_only
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x4xf32>, Hbm)
# CHECK: TEST_END: test_knob_mem_space_only
@run
def test_knob_mem_space_only():
    """Test that knob with only mem_space injects annotate op correctly."""
    
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_with_mem_space(A, B):
        temp = np.add(A, B)
        temp = knob(temp, mem_space="Hbm")
        result = np.multiply(temp, 2.0)
        return result
    
    module = func_with_mem_space.to_mlir()
    print(module)


# CHECK-LABEL: TEST: test_knob_both_params
# CHECK: nkipy.annotate(%{{.*}} : tensor<8x8xf32>, Sbuf, 1)
# CHECK: TEST_END: test_knob_both_params
@run
def test_knob_both_params():
    """Test that knob with both parameters injects annotate op correctly."""
    
    @trace(input_specs=[((8, 8), "f32"), ((8, 8), "f32")])
    def func_with_both(A, B):
        temp = np.add(A, B)
        temp = knob(temp, partition_dim=1, mem_space="Sbuf")
        result = np.multiply(temp, 3.0)
        return result
    
    module = func_with_both.to_mlir()
    print(module)


# CHECK-LABEL: TEST: test_knob_no_params
# CHECK-NOT: nkipy.annotate
# CHECK: TEST_END: test_knob_no_params
@run
def test_knob_no_params():
    """Test that knob without parameters does not inject annotate op."""
    
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_no_knob_params(A, B):
        temp = np.add(A, B)
        temp = knob(temp)  # No parameters, should be no-op
        result = np.multiply(temp, 2.0)
        return result
    
    module = func_no_knob_params.to_mlir()
    print(module)


# CHECK-LABEL: TEST: test_knob_multiple_annotations
# CHECK: nkipy.annotate(%{{.*}} : tensor<8x8xf32>, Hbm)
# CHECK: nkipy.annotate(%{{.*}} : tensor<8x8xf32>, 0)
# CHECK: nkipy.annotate(%{{.*}} : tensor<8x8xf32>, Sbuf, 1)
# CHECK: TEST_END: test_knob_multiple_annotations
@run
def test_knob_multiple_annotations():
    """Test multiple knob annotations in a single function."""
    
    @trace(input_specs=[((8, 8), "f32"), ((8, 8), "f32")])
    def func_multiple_knobs(A, B):
        temp0 = np.add(A, B)
        temp0 = knob(temp0, mem_space="Hbm")
        
        temp1 = np.multiply(temp0, 2.0)
        temp1 = knob(temp1, partition_dim=0)
        
        temp2 = np.square(temp1)
        temp2 = knob(temp2, partition_dim=1, mem_space="Sbuf")
        
        result = np.multiply(temp2, 3.0)
        return result
    
    module = func_multiple_knobs.to_mlir()
    print(module)


# CHECK-LABEL: TEST: test_knob_mem_space_values
# CHECK: TEST: Hbm
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x4xf32>, Hbm)
# CHECK: TEST: Psum
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x4xf32>, Psum)
# CHECK: TEST: Sbuf
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x4xf32>, Sbuf)
# CHECK: TEST: SharedHbm
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x4xf32>, SharedHbm)
# CHECK: TEST_END: test_knob_mem_space_values
@run
def test_knob_mem_space_values():
    """Test that different mem_space values map to correct enum values."""
    
    # Test Hbm (0)
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_hbm(A, B):
        result = np.add(A, B)
        result = knob(result, mem_space="Hbm")
        return result
    
    print("TEST: Hbm")
    print(func_hbm.to_mlir())
    
    # Test Psum (1)
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_psum(A, B):
        result = np.add(A, B)
        result = knob(result, mem_space="Psum")
        return result
    
    print("TEST: Psum")
    print(func_psum.to_mlir())
    
    # Test Sbuf (2)
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_sbuf(A, B):
        result = np.add(A, B)
        result = knob(result, mem_space="Sbuf")
        return result
    
    print("TEST: Sbuf")
    print(func_sbuf.to_mlir())
    
    # Test SharedHbm (3)
    @trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
    def func_sharedhbm(A, B):
        result = np.add(A, B)
        result = knob(result, mem_space="SharedHbm")
        return result
    
    print("TEST: SharedHbm")
    print(func_sharedhbm.to_mlir())


# CHECK-LABEL: TEST: test_knob_with_matmul
# CHECK: linalg.matmul
# CHECK: nkipy.annotate(%{{.*}} : tensor<4x5xf32>, Psum)
# CHECK: TEST_END: test_knob_with_matmul
@run
def test_knob_with_matmul():
    """Test knob annotation on matmul result."""
    
    @trace(input_specs=[((4, 3), "f32"), ((3, 5), "f32")])
    def matmul_with_knob(A, B):
        C = np.matmul(A, B)
        C = knob(C, mem_space="Psum")
        return C
    
    module = matmul_with_knob.to_mlir()
    print(module)
