// ===== BMM 3D DMA index fix =====
//
// Root cause: LinalgToNisa's getBaseAndOffsets + createCopyMap produce wrong
// affine maps for 3D HBM memrefs accessed through rank-reducing subviews.
//
// After KnobDrivenTiling, the batch dim is extracted via:
//   tensor.extract_slice %t[%b, 0, 0] [1, 256, 256] ... : tensor<2x256x256> to tensor<256x256>
//
// After bufferization this becomes a rank-reducing memref.subview:
//   memref.subview %arg0[%b, 0, 0] [1, 256, 256] ... : memref<2x256x256> to memref<256x256>
//
// Then tiling creates further 2D subviews:
//   memref.subview %sv[%m_off, %n_off] [128, 128] ... : memref<256x256> to memref<128x128>
//
// When getBaseAndOffsets traces the chain: %tile -> %sv_2d -> %arg0_3d
//   - From %tile subview: indices = [%m_off, %n_off] (2 entries)
//   - From %sv_2d subview: source is 3D %arg0 with offsets [%b, 0, 0]
//   - BUG: assert(indices.size() == offsets.size()) fails (2 != 3)
//   - In release mode (no assert): indices wrongly accumulate as 2D, then
//     createCopyMap maps d0->dim0 producing: %arg0[%b + d0, %m_off, d1]
//
// BUGGY pattern (from buggy.mlir):
//   %2 = arith.addi %0, %arg2           // %0 = tile_offset, %arg2 = batch
//   %arg0[%2 + d0, %1 + 0, d1]          // d0 (128-range partition) added to batch dim (size 2)!
//
// CORRECT pattern (this file):
//   %arg0[%arg2 + 0, %0 + d0, %1 + d1]  // batch is pure offset, d0 in M dim, d1 in N dim
//
// The 4D SBUF memrefs (128x2x2x128) are fine - they don't use rank-reducing subviews.
//
module attributes {nisa.target = #nisa.target<trn2>} {
  func.func @bmm_kernel(%arg0: memref<2x256x256xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>>, %arg1: memref<2x256x256xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>>) -> memref<2x256x256xf32, #nisa.mem<shared_hbm>> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %mem = nisa.alloc alignment=64 : memref<2x256x256xf32, #nisa.mem<shared_hbm>>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %mem_0 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %0 = arith.muli %arg4, %c128 : index
          %1 = arith.muli %arg3, %c128 : index
          nisa.dma_transpose(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_0[d0, %arg3 + 0, %arg4 + 0, d1], src<128| 128>=memref<2x256x256xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg0[%arg2 + 0, %0 + d0, %1 + d1], permutation=[1, 0], dge_mode=no_dge, oob_is_err=true) engine=dma
        }
      }
      %mem_1 = nisa.alloc alignment=64 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %0 = arith.muli %arg3, %c128 : index
          %1 = arith.muli %arg4, %c128 : index
          nisa.dma_copy(dst<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_1[d0, %arg3 + 0, %arg4 + 0, d1], src<128| 128>=memref<2x256x256xf32, strided<[?, ?, ?], offset: ?>, #nisa.mem<shared_hbm>> %arg1[%arg2 + 0, %0 + d0, %1 + d1], dge_mode=no_dge, oob_is_err=true) engine=dma
        }
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %0 = arith.muli %arg3, %c128 : index
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %1 = arith.muli %arg4, %c128 : index
          %mem_2 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<psum>>
          scf.for %arg5 = %c0 to %c2 step %c1 {
            nisa.matmul(dst<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_2[d0, d1], stationary<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_0[d0, %arg5 + 0, %arg3 + 0, d1], moving<128| 128>=memref<128x2x2x128xf32, #nisa.mem<sbuf>> %mem_1[d0, %arg5 + 0, %arg4 + 0, d1], row_pos=index %c0, col_pos=index %c0, is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor
          }
          %mem_3 = nisa.alloc alignment=64 : memref<128x128xf32, #nisa.mem<sbuf>>
          nisa.tensor_copy(dst<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_3[d0, d1], src<128| 128>=memref<128x128xf32, #nisa.mem<psum>> %mem_2[d0, d1]) engine=vector
          nisa.dma_copy(dst<128| 128>=memref<2x256x256xf32, #nisa.mem<shared_hbm>> %mem[%arg2 + 0, %0 + d0, %1 + d1], src<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %mem_3[d0, d1], dge_mode=no_dge, oob_is_err=true) engine=dma
          nisa.release %mem_2 : memref<128x128xf32, #nisa.mem<psum>>
        }
      }
      nisa.release %mem_0 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
      nisa.release %mem_1 : memref<128x2x2x128xf32, #nisa.mem<sbuf>>
    }
    return %mem : memref<2x256x256xf32, #nisa.mem<shared_hbm>>
  }
}
