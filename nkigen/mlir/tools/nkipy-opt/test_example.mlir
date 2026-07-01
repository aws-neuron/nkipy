// Test example for nkipy-opt
// This file demonstrates a simple MLIR module with memref operations
// that can be optimized using the memref-dce pass

module {
  func.func @test_dce(%arg0: memref<10xf32>) -> f32 {
    // This allocation is never loaded from - should be removed by memref-dce
    %dead_alloc = memref.alloc() : memref<10xf32>
    
    // This allocation is used - should be kept
    %used_alloc = memref.alloc() : memref<5xf32>
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // Load from input argument
    %val = memref.load %arg0[%c0] : memref<10xf32>
    
    // Store to used allocation
    memref.store %val, %used_alloc[%c0] : memref<5xf32>
    
    // Load from used allocation
    %result = memref.load %used_alloc[%c1] : memref<5xf32>
    
    return %result : f32
  }
}
