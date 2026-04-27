module attributes {nisa.target = #nisa.target<trn2>} {
  func.func @qwen3_layer(%arg0: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg1: memref<256x1xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg2: memref<256x1xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg3: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg4: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg5: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg6: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg7: memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg8: memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg9: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg10: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg11: memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>>) -> memref<256x256xf32, #nisa.mem<shared_hbm>> attributes {nki.output_names = ["output"]} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.0883883461 : f32
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %cst_3 = arith.constant 3.906250e-03 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %mem = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg0[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.activation(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], bias=f32 %cst_4, scale=f32 %cst, op=square) engine=scalar
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1]) engine=vector
        nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_5 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      nisa.memset(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_5[d0, %arg12 + 0, 0, d1], value=f32 %cst_4) engine=vector
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc : memref<128x1xf32, #nisa.mem<sbuf>>
        nisa.tensor_reduce_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem[d0, %arg12 + 0, %arg13 + 0, d1], op=add, negated=false, num_r_dim=1) engine=vector
        nisa.tensor_tensor_arith(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_5[d0, %arg12 + 0, 0, d1], lhs<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_5[d0, %arg12 + 0, 0, d1], rhs<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], op=add) engine=vector
        nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_6 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_5[d0, %arg12 + 0, 0, d1], operand0=f32 %cst_3, op0=multiply, reverse_operands=none_) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_6[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_5 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_7 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_6[d0, %arg12 + 0, 0, d1], operand0=f32 %cst_2, op0=add, reverse_operands=none_) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_7[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_6 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_8 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.activation(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_7[d0, %arg12 + 0, 0, d1], bias=f32 %cst_4, scale=f32 %cst, op=sqrt) engine=scalar
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_8[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_7 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_9 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_10 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.reciprocal(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_8[d0, %arg12 + 0, 0, d1]) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_10[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_8 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg0[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_scalar_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], operand0<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_10[d0, %arg12 + 0, 0, d1], op0=multiply, reverse_operands=none_) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_9[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1]) engine=vector
        nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_10 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_11 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<256x1xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg1[%0 + d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_scalar_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_9[d0, %arg12 + 0, %arg13 + 0, d1], operand0<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], op0=multiply, reverse_operands=none_) engine=vector
        nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_11[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1]) engine=vector
        nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_9 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_12 = nisa.alloc alignment=64 : memref<256x256xf32, #nisa.mem<shared_hbm>>
    %mem_13 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_13[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_11[d0, %arg13 + 0, %arg12 + 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    %mem_14 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_14[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg3[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_13[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_14[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.dma_copy(dst<128| 128>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_12[%0 + d0, %1 + d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_14 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_13 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_15 = nisa.alloc alignment=64 : memref<256x256xf32, #nisa.mem<shared_hbm>>
    %mem_16 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_16[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_11[d0, %arg13 + 0, %arg12 + 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    %mem_17 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_17[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg4[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_16[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_17[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.dma_copy(dst<128| 128>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_15[%0 + d0, %1 + d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_17 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_16 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_18 = nisa.alloc alignment=64 : memref<256x256xf32, #nisa.mem<shared_hbm>>
    %mem_19 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_19[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_11[d0, %arg13 + 0, %arg12 + 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    nisa.release %mem_11 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_20 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_20[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg5[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_19[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_20[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.dma_copy(dst<128| 128>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_18[%0 + d0, %1 + d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_20 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_19 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_21 = nisa.alloc alignment=64 : memref<2x2x128x128xf32, #nisa.mem<hbm>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c128 step %c1 {
        %mem_78 = nisa.alloc : memref<128x2xf32, #nisa.mem<sbuf>>
        %mem_79 = nisa.alloc : memref<2x128xf32, #nisa.mem<sbuf>>
        %0 = arith.muli %arg12, %c128 : index
        nisa.dma_copy(dst<128| 2>=memref<128x2xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 2>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_12[%0 + d0, %arg13 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.dma_transpose(dst<2| 128>=memref<2x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 2>=memref<128x2xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x2xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<2| 128>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_21[%arg12 + 0, d0, d1, %arg13 + 0], src<2| 128>=memref<2x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_79 : memref<2x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_22 = nisa.alloc alignment=64 : memref<2x2x128x128xf32, #nisa.mem<hbm>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c128 step %c1 {
        %mem_78 = nisa.alloc : memref<128x2xf32, #nisa.mem<sbuf>>
        %mem_79 = nisa.alloc : memref<2x128xf32, #nisa.mem<sbuf>>
        %0 = arith.muli %arg12, %c128 : index
        nisa.dma_copy(dst<128| 2>=memref<128x2xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 2>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_15[%0 + d0, %arg13 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.dma_transpose(dst<2| 128>=memref<2x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 2>=memref<128x2xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x2xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<2| 128>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_22[%arg12 + 0, d0, d1, %arg13 + 0], src<2| 128>=memref<2x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_79 : memref<2x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_23 = nisa.alloc alignment=64 : memref<2x2x128x128xf32, #nisa.mem<shared_hbm>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c128 step %c1 {
        %mem_78 = nisa.alloc : memref<128x2xf32, #nisa.mem<sbuf>>
        %mem_79 = nisa.alloc : memref<2x128xf32, #nisa.mem<sbuf>>
        %0 = arith.muli %arg12, %c128 : index
        nisa.dma_copy(dst<128| 2>=memref<128x2xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 2>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_18[%0 + d0, %arg13 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.dma_transpose(dst<2| 128>=memref<2x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 2>=memref<128x2xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x2xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<2| 128>=memref<2x2x128x128xf32, #nisa.mem<shared_hbm>> %mem_23[%arg12 + 0, d0, d1, %arg13 + 0], src<2| 128>=memref<2x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_79 : memref<2x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_24 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_21[%0 + 0, %1 + d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg7[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_24[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_25 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_21[%0 + 0, %1 + d0, d1, d2 + 64], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg8[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_25[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_26 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], lhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_24[%arg12 + d0, d1, d2], rhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_25[%arg12 + d0, d1, d2], op=subtract) engine=vector
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_26[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_25 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    nisa.release %mem_24 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    %mem_27 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_21[%0 + 0, %1 + d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg8[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_27[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_28 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_21[%0 + 0, %1 + d0, d1, d2 + 64], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg7[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_28[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_29 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], lhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_27[%arg12 + d0, d1, d2], rhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_28[%arg12 + d0, d1, d2], op=add) engine=vector
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_29[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_28 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    nisa.release %mem_27 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    %mem_30 = nisa.alloc alignment=64 : memref<4x128x128xf32, #nisa.mem<shared_hbm>>
    nisa.dma_copy(dst<4| 128, 64>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_30[d0, d1, d2], src<4| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_26[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
    nisa.release %mem_26 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    nisa.dma_copy(dst<4| 128, 64>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_30[d0, d1, d2 + 64], src<4| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_29[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
    nisa.release %mem_29 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    %mem_31 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_22[%0 + 0, %1 + d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg7[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_31[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_32 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_22[%0 + 0, %1 + d0, d1, d2 + 64], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg8[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_32[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_33 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], lhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_31[%arg12 + d0, d1, d2], rhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_32[%arg12 + d0, d1, d2], op=subtract) engine=vector
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_33[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_32 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    nisa.release %mem_31 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    %mem_34 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_22[%0 + 0, %1 + d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg8[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_34[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_35 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 128, 64>=memref<2x2x128x128xf32, #nisa.mem<hbm>> %mem_22[%0 + 0, %1 + d0, d1, d2 + 64], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg7[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], lhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], rhs<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], op=multiply) engine=vector
      nisa.release %mem_79 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_35[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_80[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    %mem_36 = nisa.alloc alignment=64 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<1x128x64xf32, #nisa.mem<sbuf>>
      nisa.tensor_tensor_arith(dst<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], lhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_34[%arg12 + d0, d1, d2], rhs<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_35[%arg12 + d0, d1, d2], op=add) engine=vector
      nisa.dma_copy(dst<1| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_36[%arg12 + d0, d1, d2], src<1| 128, 64>=memref<1x128x64xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_78 : memref<1x128x64xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_35 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    nisa.release %mem_34 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    %mem_37 = nisa.alloc alignment=64 : memref<4x128x128xf32, #nisa.mem<shared_hbm>>
    nisa.dma_copy(dst<4| 128, 64>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_37[d0, d1, d2], src<4| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_33[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
    nisa.release %mem_33 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    nisa.dma_copy(dst<4| 128, 64>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_37[d0, d1, d2 + 64], src<4| 128, 64>=memref<4x128x64xf32, #nisa.mem<sbuf>> %mem_36[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
    nisa.release %mem_36 : memref<4x128x64xf32, #nisa.mem<sbuf>>
    %mem_38 = nisa.alloc alignment=64 : memref<4x128x128xf32, #nisa.mem<shared_hbm>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_37[%arg12 + 0, d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.dma_transpose(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<128| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_38[%arg12 + 0, d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
    }
    %mem_39 = nisa.alloc alignment=64 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<shared_hbm>>
      %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.dma_transpose(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_30[%arg12 + 0, d0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_80[d0, d1], src<128| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_38[%arg12 + 0, d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_81 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
      nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_81[d0, d1], stationary<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], moving<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_80[d0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
      nisa.release %mem_80 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      %mem_82 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.tensor_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_82[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_81[d0, d1]) engine=vector
      nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<shared_hbm>> %mem_78[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_82[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_81 : memref<128x128xf32, #nisa.mem<psum>>
      nisa.dma_copy(dst<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_39[d0, 0, %arg12 + 0, 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<shared_hbm>> %mem_78[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
    }
    %mem_40 = nisa.alloc alignment=64 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1x128xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1], src<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_39[d0, 0, %arg12 + 0, 0, d1], operand0=f32 %cst_1, op0=multiply, reverse_operands=none_) engine=vector
      nisa.tensor_copy(dst<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_40[d0, 0, %arg12 + 0, 0, d1], src<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1x128xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_39 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    %mem_41 = nisa.alloc alignment=64 : memref<128x1x4x1x1xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      nisa.memset(dst<128| 1, 1, 1, 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_41[d0, d1, %arg12 + d2, d3, d4], value=f32 %cst_0) engine=vector
    }
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.tensor_reduce_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_40[d0, 0, %arg12 + 0, 0, d1], op=max, negated=false, num_r_dim=1) engine=vector
      nisa.tensor_tensor_arith(dst<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_41[d0, 0, %arg12 + 0, 0, d1], lhs<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_41[d0, 0, %arg12 + 0, 0, d1], rhs<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], op=max) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    %mem_42 = nisa.alloc alignment=64 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1x128xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1], src<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_40[d0, 0, %arg12 + 0, 0, d1], operand0<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_41[d0, 0, %arg12 + 0, 0, d1], op0=subtract, reverse_operands=none_) engine=vector
      nisa.tensor_copy(dst<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_42[d0, 0, %arg12 + 0, 0, d1], src<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1x128xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_41 : memref<128x1x4x1x1xf32, #nisa.mem<sbuf>>
    nisa.release %mem_40 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    %mem_43 = nisa.alloc alignment=64 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1x128xf32, #nisa.mem<sbuf>>
      nisa.activation(dst<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1], src<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_42[d0, 0, %arg12 + 0, 0, d1], bias=f32 %cst_4, scale=f32 %cst, op=exp) engine=scalar
      nisa.tensor_copy(dst<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_43[d0, 0, %arg12 + 0, 0, d1], src<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1x128xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_42 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    %mem_44 = nisa.alloc alignment=64 : memref<128x1x4x1x1xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      nisa.memset(dst<128| 1, 1, 1, 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_44[d0, d1, %arg12 + d2, d3, d4], value=f32 %cst_4) engine=vector
    }
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.tensor_reduce_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_43[d0, 0, %arg12 + 0, 0, d1], op=add, negated=false, num_r_dim=1) engine=vector
      nisa.tensor_tensor_arith(dst<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_44[d0, 0, %arg12 + 0, 0, d1], lhs<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_44[d0, 0, %arg12 + 0, 0, d1], rhs<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], op=add) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    %mem_45 = nisa.alloc alignment=64 : memref<128x4x128xf32, #nisa.mem<shared_hbm>>
    %mem_46 = nisa.alloc alignment=64 : memref<128x1x4x1x1xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1x1xf32, #nisa.mem<sbuf>>
      nisa.reciprocal(dst<128| 1>=memref<128x1x1xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1], src<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_44[d0, 0, %arg12 + 0, 0, d1]) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_46[d0, 0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1x1xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_44 : memref<128x1x4x1x1xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1x128xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1], src<128| 128>=memref<128x1x4x1x128xf32, #nisa.mem<sbuf>> %mem_43[d0, 0, %arg12 + 0, 0, d1], operand0<128| 1>=memref<128x1x4x1x1xf32, #nisa.mem<sbuf>> %mem_46[d0, 0, %arg12 + 0, 0, d1], op0=multiply, reverse_operands=none_) engine=vector
      nisa.dma_copy(dst<128| 1, 128>=memref<128x4x128xf32, #nisa.mem<shared_hbm>> %mem_45[d0, %arg12 + d1, d2], src<128| 1, 128>=memref<128x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_78 : memref<128x1x128xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_46 : memref<128x1x4x1x1xf32, #nisa.mem<sbuf>>
    nisa.release %mem_43 : memref<128x1x4x1x128xf32, #nisa.mem<sbuf>>
    %mem_47 = nisa.alloc alignment=64 : memref<4x128x128xf32, #nisa.mem<shared_hbm>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      scf.for %arg13 = %c0 to %c128 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<1x1x128xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<1| 1, 128>=memref<1x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1, d2], src<1| 1, 128>=memref<128x4x128xf32, #nisa.mem<shared_hbm>> %mem_45[%arg13 + d0, %arg12 + d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
        %mem_79 = nisa.alloc alignment=64 : memref<1x1x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<1| 128>=memref<1x1x128xf32, #nisa.mem<sbuf>> %mem_79[d0, 0, d1], src<1| 128>=memref<1x1x128xf32, #nisa.mem<sbuf>> %mem_78[d0, 0, d1]) engine=vector
        nisa.release %mem_78 : memref<1x1x128xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<1| 1, 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_47[%arg12 + d0, %arg13 + d1, d2], src<1| 1, 128>=memref<1x1x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1, d2], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_79 : memref<1x1x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_48 = nisa.alloc alignment=64 : memref<4x128x128xf32, #nisa.mem<shared_hbm>>
    scf.for %arg12 = %c0 to %c4 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.dma_transpose(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_47[%arg12 + 0, d0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      %0 = arith.divui %arg12, %c2 : index
      %1 = arith.remui %arg12, %c2 : index
      nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<2x2x128x128xf32, #nisa.mem<shared_hbm>> %mem_23[%0 + 0, %1 + 0, d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      %mem_80 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
      nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_80[d0, d1], stationary<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], moving<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
      nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      %mem_81 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
      nisa.tensor_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_81[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_80[d0, d1]) engine=vector
      nisa.dma_copy(dst<128| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_48[%arg12 + 0, d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_81[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      nisa.release %mem_80 : memref<128x128xf32, #nisa.mem<psum>>
    }
    %mem_49 = nisa.alloc alignment=64 : memref<2x128x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c128 step %c1 {
        %0 = arith.muli %arg12, %c2 : index
        nisa.dma_transpose(dst<128| 2>=memref<2x128x2x128xf32, #nisa.mem<sbuf>> %mem_49[%arg12 + d0, 0, 0, %arg13 + d1], src<2| 128>=memref<4x128x128xf32, #nisa.mem<shared_hbm>> %mem_48[%0 + d0, 0, %arg13 + d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    %mem_50 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_51 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg13, %c128 : index
        %1 = arith.muli %arg12, %c128 : index
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_51[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<2x128x2x128xf32, #nisa.mem<sbuf>> %mem_49[%0 + d0, %1 + 0, 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    nisa.release %mem_49 : memref<2x128x2x128xf32, #nisa.mem<sbuf>>
    %mem_52 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_52[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg6[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_51[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_52[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_50[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_52 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_51 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_53 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg0[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_tensor_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], lhs<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], rhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_50[d0, %arg12 + 0, %arg13 + 0, d1], op=add) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_53[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1]) engine=vector
        nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_50 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_54 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.activation(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_53[d0, %arg12 + 0, %arg13 + 0, d1], bias=f32 %cst_4, scale=f32 %cst, op=square) engine=scalar
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_54[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_55 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      nisa.memset(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_55[d0, %arg12 + 0, 0, d1], value=f32 %cst_4) engine=vector
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc : memref<128x1xf32, #nisa.mem<sbuf>>
        nisa.tensor_reduce_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_54[d0, %arg12 + 0, %arg13 + 0, d1], op=add, negated=false, num_r_dim=1) engine=vector
        nisa.tensor_tensor_arith(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_55[d0, %arg12 + 0, 0, d1], lhs<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_55[d0, %arg12 + 0, 0, d1], rhs<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], op=add) engine=vector
        nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_54 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_56 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_55[d0, %arg12 + 0, 0, d1], operand0=f32 %cst_3, op0=multiply, reverse_operands=none_) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_56[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_55 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_57 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.tensor_scalar_arith(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_56[d0, %arg12 + 0, 0, d1], operand0=f32 %cst_2, op0=add, reverse_operands=none_) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_57[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_56 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_58 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.activation(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_57[d0, %arg12 + 0, 0, d1], bias=f32 %cst_4, scale=f32 %cst, op=sqrt) engine=scalar
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_58[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_57 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_59 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_60 = nisa.alloc alignment=64 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
      nisa.reciprocal(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_58[d0, %arg12 + 0, 0, d1]) engine=vector
      nisa.tensor_copy(dst<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_60[d0, %arg12 + 0, 0, d1], src<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
      nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
    }
    nisa.release %mem_58 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_scalar_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_53[d0, %arg12 + 0, %arg13 + 0, d1], operand0<128| 1>=memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>> %mem_60[d0, %arg12 + 0, 0, d1], op0=multiply, reverse_operands=none_) engine=vector
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_59[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_60 : memref<128x2x1x1xf32, strided<[1, 128, 128, 128]>, #nisa.mem<sbuf>>
    %mem_61 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x1xf32, #nisa.mem<sbuf>>
        nisa.dma_copy(dst<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 1>=memref<256x1xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg2[%0 + d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        %mem_79 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_scalar_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_59[d0, %arg12 + 0, %arg13 + 0, d1], operand0<128| 1>=memref<128x1xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], op0=multiply, reverse_operands=none_) engine=vector
        nisa.release %mem_78 : memref<128x1xf32, #nisa.mem<sbuf>>
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_61[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_79[d0, d1]) engine=vector
        nisa.release %mem_79 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_59 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_62 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_63 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_63[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_61[d0, %arg13 + 0, %arg12 + 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    %mem_64 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_64[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg9[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_63[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_64[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_62[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_63 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_65 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_66 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_66[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_61[d0, %arg13 + 0, %arg12 + 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    nisa.release %mem_61 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_67 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_67[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg10[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_66[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_67[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_65[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_67 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_66 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_68 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_scalar_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_62[d0, %arg12 + 0, %arg13 + 0, d1], operand0=f32 %cst_4, op0=subtract, reverse_operands=first) engine=vector
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_68[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    %mem_69 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.activation(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_68[d0, %arg12 + 0, %arg13 + 0, d1], bias=f32 %cst_4, scale=f32 %cst, op=exp) engine=scalar
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_69[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_68 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_70 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_scalar_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_69[d0, %arg12 + 0, %arg13 + 0, d1], operand0=f32 %cst, op0=add, reverse_operands=none_) engine=vector
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_70[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_69 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_71 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.reciprocal(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_70[d0, %arg12 + 0, %arg13 + 0, d1]) engine=vector
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_71[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_70 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_72 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_tensor_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], lhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_62[d0, %arg12 + 0, %arg13 + 0, d1], rhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_71[d0, %arg12 + 0, %arg13 + 0, d1], op=multiply) engine=vector
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_72[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_71 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_62 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_73 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_tensor_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], lhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_72[d0, %arg12 + 0, %arg13 + 0, d1], rhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_65[d0, %arg12 + 0, %arg13 + 0, d1], op=multiply) engine=vector
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_73[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_72 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_65 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_74 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_75 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_75[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_73[d0, %arg13 + 0, %arg12 + 0, d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    nisa.release %mem_73 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_76 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg12, %c128 : index
        %1 = arith.muli %arg13, %c128 : index
        nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_76[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<256x256xf32, strided<[?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg11[%0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
      }
    }
    scf.for %arg12 = %c0 to %c2 step %c1 {
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
        scf.for %arg14 = %c0 to %c2 step %c1 {
          nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_75[d0, %arg14 + 0, %arg12 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_76[d0, %arg14 + 0, %arg13 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
        }
        nisa.tensor_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_74[d0, %arg12 + 0, %arg13 + 0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_78[d0, d1]) engine=vector
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<psum>>
      }
    }
    nisa.release %mem_76 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_75 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    %mem_77 = nisa.alloc alignment=64 : memref<256x256xf32, #nisa.mem<shared_hbm>>
    scf.for %arg12 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg12, %c128 : index
      scf.for %arg13 = %c0 to %c2 step %c1 {
        %1 = arith.muli %arg13, %c128 : index
        %mem_78 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
        nisa.tensor_tensor_arith(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], lhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_53[d0, %arg12 + 0, %arg13 + 0, d1], rhs<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_74[d0, %arg12 + 0, %arg13 + 0, d1], op=add) engine=vector
        nisa.dma_copy(dst<128| 128>=memref<256x256xf32, #nisa.mem<shared_hbm>> %mem_77[%0 + d0, %1 + d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_78[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        nisa.release %mem_78 : memref<128x128xf32, #nisa.mem<sbuf>>
      }
    }
    nisa.release %mem_74 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    nisa.release %mem_53 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    return %mem_77 : memref<256x256xf32, #nisa.mem<shared_hbm>>
  }
}