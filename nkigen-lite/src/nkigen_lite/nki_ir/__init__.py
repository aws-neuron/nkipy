"""NKI-level IR for NeuronCore targets."""

from nkigen_lite.nki_ir.ir import (
    DimSlice,
    MemorySpace,
    TileType,
    NisaActivationOp,
    NisaArithOp,
    NisaRangeSelectCmp,
    NisaReduceOp,
    Graph,
    Builder,
    unroll_tile_loops,
    PARTITION_MAX,
    PSUM_FREE_MAX,
    MATMUL_STATIONARY_FREE_MAX,
    MATMUL_MOVING_FREE_MAX,
    SBUF_PER_PARTITION_BYTES,
    PSUM_PER_PARTITION_BYTES,
    PSUM_BANKS,
    PSUM_BANK_ELEMENTS,
)

from nkigen_lite.nki_ir.interpret import (
    interpret,
    run,
    eval_nisa_op,
)

from nkigen_lite.nki_ir.insert_deallocs import insert_deallocs
