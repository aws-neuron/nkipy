"""
Python wrapper for nkipy-opt passes that can't run from Python.

These passes use NISA dialect which has global constructors that conflict with
Python bindings. They must be run via the nkipy-opt C++ tool instead.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def _pass_to_arg(pass_name: str) -> str:
    """Convert a pass spec to a CLI argument.

    Examples:
        'prepare-arithmetic' -> '--prepare-arithmetic'
        'one-shot-bufferize="opt1 opt2"' -> '--one-shot-bufferize=opt1 opt2'
        'insert-spill-reload="target=trn2"' -> '--insert-spill-reload=target=trn2'
    """
    if '=' in pass_name:
        name, opts = pass_name.split('=', 1)
        opts = opts.strip('"').strip("'")
        return f'--{name}={opts}'
    return f'--{pass_name}'


def get_nkipy_opt_path():
    """Get the path to the nkipy-opt executable."""
    # Assumes we're in the NKIPyKernelGen package
    package_dir = Path(__file__).parent.parent.parent
    nkipy_opt = package_dir / "build" / "bin" / "nkipy-opt"

    if not nkipy_opt.exists():
        raise FileNotFoundError(
            f"nkipy-opt not found at {nkipy_opt}. "
            "Please build the project first."
        )

    return str(nkipy_opt)


def run_nkipy_opt_passes(
    mlir_module,
    passes: list[str],
    print_ir_after_all: bool = False,
    print_stderr: bool = False,
    print_debuginfo: bool = False,
    print_generic: bool = False,
) -> str:
    """
    Run nkipy-opt passes on an MLIR module.

    Args:
        mlir_module: MLIR module (string or Module object)
        passes: List of pass names (e.g., ['cleanup-bufferization-artifacts'])
        print_ir_after_all: If True, print IR after each pass (adds --mlir-print-ir-after-all)
        print_stderr: If True, print stderr output (useful for debugging pass diagnostics)
        print_debuginfo: If True, include source locations in output (adds --mlir-print-debuginfo)

    Returns:
        Transformed MLIR module text (when print_ir_after_all=False)
        Or IR dumps from all passes followed by final module (when print_ir_after_all=True)

    Raises:
        RuntimeError: If nkipy-opt fails
    """
    nkipy_opt = get_nkipy_opt_path()

    # Convert Module object to string if needed
    mlir_text = str(mlir_module) if not isinstance(mlir_module, str) else mlir_module

    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f_in:
        f_in.write(mlir_text)
        input_file = f_in.name

    try:
        # Build command as a list to avoid shell injection
        cmd = [nkipy_opt]
        if print_ir_after_all:
            cmd.append('--mlir-print-ir-after-all')
        if print_debuginfo:
            cmd.append('--mlir-print-debuginfo')
        if print_generic:
            cmd.append('--mlir-print-op-generic')
        cmd.extend(_pass_to_arg(p) for p in passes)
        cmd.append(input_file)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            cmd_str = ' '.join(cmd)
            error_msg = f"nkipy-opt failed with return code {result.returncode}\n"
            error_msg += f"Command: {cmd_str}\n"
            error_msg += f"Stdout:\n{result.stdout}\n"
            error_msg += f"Stderr:\n{result.stderr}"
            raise RuntimeError(error_msg)

        # Print stderr output if requested or when print_ir_after_all is enabled.
        # MLIR outputs IR dumps and pass diagnostics to stderr.
        # Always return just the clean module (stdout) so output can be parsed.
        if result.stderr and (print_stderr or print_ir_after_all):
            if print_ir_after_all:
                print("// === IR After Each Pass ===")
            else:
                print("// === stderr output ===")
            print(result.stderr)
            if print_ir_after_all:
                print("// === End IR Dumps ===\n")
            else:
                print("// === end stderr ===\n")

        return result.stdout

    finally:
        # Clean up temporary file
        if os.path.exists(input_file):
            os.unlink(input_file)


def apply_complete_knob_pipeline(
    mlir_module: str,
    target: str = "trn2",
    print_ir_after_all: bool = False,
    dump_dir: str = None,
    stop_after=None,
    print_debuginfo: bool = False,
    print_generic: bool = False,
) -> str:
    """
    Apply the complete knob-driven compilation pipeline in a single pass.

    This avoids switching between Python bindings and nkipy-opt, running all
    passes through nkipy-opt in sequence:

    Phase 0: Arithmetic Preparation
     1. remove-redundant-zero-fill: Remove linalg.fill(0) before matmul (NISA auto-zeros PSUM)
     2. prepare-arithmetic: Convert div to mul+reciprocal (NISA has no divide)

    Phase 1: Layout Inference, Partition Dim Canonicalization, and Tiling (on tensor IR)
     3. infer-layout: Infer tiling, placement, and partition_dim for unannotated ops
     4. canonicalize-partition-dim: Insert transposes to ensure partition_dim=0 everywhere
     5. assign-linalg-op-ids: Assign unique IDs to linalg ops (incl. new transposes)
     6. knob-driven-tiling: Rewrite linalg ops to tiled loops using transform dialect
     7. apply-and-strip-transforms: Apply the generated transforms, then erase
        the transform module (so downstream passes — including the Python
        linalg->NISA phase — see no transform-dialect ops).
     8. canonicalize-loop-step: Normalize loop steps to 1

    Phase 2: Bufferization
     9. one-shot-bufferize: Convert tensors to memrefs
    10. canonicalize: Clean up memref operations

    Phase 3: Memory Space Annotation + Reshape Canonicalization
    11. eliminate-uninitialized-copies: Remove copies from uninitialized buffers
    12. canonicalize: Clean up dead subview chains
    13. annotate-memory-space: Apply memory space attributes
    14. canonicalize-reshape: Classify expand/collapse_shape by mem_space and partition_dim
    15. eliminate-same-memspace-copy: Remove redundant SBUF->SBUF copies
    16. canonicalize: Clean up dead allocs

    Phase 4: Memref Finalization
    17. legalize-layout: Transform SBUF tensors to physical 4D layout
    18. canonicalize: Clean up after layout legalization
    19. simplify-linalg: Decompose high-rank transposes, canonicalize trivial-broadcast generics
    20. insert-spill-reload: Insert spill/reload for SBUF overflow
    21. insert-memref-dealloc: Insert memref.dealloc at allocation scope end
    22. cse: Common subexpression elimination
    23. canonicalize: DCE for unused subviews and cleanup

    Note: nkipy.annotate ops are removed in annotate-memory-space (pass 13).
    Note: The prior NISA-lowering steps (linalg-to-nisa, resolve-custom-ops,
    prepare-for-nki) are currently stripped. They will be reimplemented in
    Python using the public nki wheel as part of open-sourcing.

    Args:
        mlir_module: MLIR module text with tensor operations and knob annotations
        target: Hardware target (default "trn2")
        print_ir_after_all: If True, print IR after each pass
        dump_dir: If provided, save intermediate MLIR files after each pass to this directory
        stop_after: Controls how many passes to run. Can be:
            - None: run all passes (default)
            - int: stop after pass N (1-indexed)
            - str: stop after the first occurrence of the named pass.
              For passes that appear multiple times (e.g. "canonicalize"),
              use "name:N" to stop at the Nth occurrence (1-indexed).
        print_debuginfo: If True, include source locations in output (--mlir-print-debuginfo)
        print_generic: If True, print ops in generic form (--mlir-print-op-generic)

    Returns:
        Fully transformed MLIR module with NISA operations
    """
    passes = [
        # Phase 0: Arithmetic preparation (pre-tiling)
        # Remove zero fills before matmul — NISA matmul auto-zeros PSUM, so
        # linalg.fill(0) feeding into matmul outs is redundant. Must run before
        # tiling/bufferization to avoid generating unnecessary nisa.memset.
        'remove-redundant-zero-fill',                                           # 1
        # Convert linalg.div to linalg.mul + linalg.reciprocal since NISA
        # tensor_tensor_arith doesn't support DIVIDE
        'prepare-arithmetic',                                                   # 2

        # Phase 1: Layout inference and partition_dim canonicalization
        # InferLayout infers tiling, placement (mem_space), and partition_dim for
        # elementwise ops that lack explicit annotations, by propagating from
        # annotated neighbors
        'infer-layout',                                                         # 3
        # CanonicalizePartitionDim inserts transposes to ensure partition_dim=0
        # everywhere. Must run after infer-layout (so partition_dim is propagated)
        # and before assign-linalg-op-ids (so new transposes get op IDs)
        'canonicalize-partition-dim',                                            # 4
        # AssignLinalgOpIds assigns unique nkipy.op_id to each linalg op
        # (including transposes inserted above)
        'assign-linalg-op-ids',                                                 # 5
        # KnobDrivenTiling generates Transform dialect IR; the fused pass
        # applies it and then erases the transform module so downstream
        # (including the Python linalg->NISA phase) sees no transform-dialect
        # ops in the IR.
        'knob-driven-tiling',                                                   # 6
        'apply-and-strip-transforms',                                           # 7
        # CanonicalizeLoopStep normalizes loop steps to 1 (e.g., for %i = 0 to 512 step 128)
        # This simplifies index expressions from %i*128/128 to just %i
        'canonicalize-loop-step',                                               # 8

        # Phase 2: Bufferization
        'one-shot-bufferize="bufferize-function-boundaries allow-unknown-ops"', # 9
        'canonicalize',                                                         # 10

        # Phase 3: Memory Space Annotation + Reshape Canonicalization
        # Eliminate copies from uninitialized allocations (e.g., PSUM accumulator init)
        # Must run after bufferization, before annotate-memory-space
        'eliminate-uninitialized-copies',                                        # 11
        'canonicalize',  # Clean up dead subview chains from eliminated copies   # 12
        'annotate-memory-space',                                                 # 13
        # CanonicalizeReshape: classify expand/collapse_shape by mem_space and
        # partition_dim. HBM reshapes and SBUF non-pdim reshapes stay as views.
        # SBUF partition dim splits get alloc+copy (NISA has no modulo).
        # Returned expand_shape views of func args and direct returns of func
        # args get alloc+copy (NISA needs separate output allocations).
        'canonicalize-reshape',                                                  # 14
        # Eliminate redundant SBUF->SBUF copies (when data is already in SBUF)
        # This is needed after SBUF promotion of elementwise ops — if an input
        # is already in SBUF (e.g., from a previous matmul), we don't need to copy it again
        'eliminate-same-memspace-copy',                                          # 15
        'canonicalize',  # Clean up dead allocs and subviews from eliminated copies  # 16

        # Phase 4: NISA Lowering
        # LegalizeLayout transforms SBUF tensors from 2D to 4D physical layout
        # Runs here to inspect IR after bufferization
        'legalize-layout',                                                       # 17
        'canonicalize',                                                          # 18
        # Simplify linalg ops before NISA lowering: decompose high-rank
        # transposes to loops of 2D, collapse >2D SBUF transpose to 2D,
        # canonicalize trivial-broadcast generics to named ops.
        # Runs before insert-spill-reload so any SBUF temps it creates
        # are accounted for in spill/reload memory budgeting.
        'simplify-linalg',                                                       # 19
        # Insert spill/reload for SBUF memory pressure.  Runs after legalize-layout
        # so SBUF allocs are already in physical per-partition layout and their
        # total byte size equals the per-partition SBUF consumption.
        f'insert-spill-reload="target={target}"',                                # 20
        'insert-memref-dealloc',  # Insert memref.dealloc ops at allocation scope end  # 21
        'cse',  # Common subexpression elimination                               # 22
        'canonicalize',  # DCE for unused subviews and cleanup                   # 23

        # Phase 5: NISA lowering (Python) — reimplementation of the deleted C++
        # linalg-to-nisa / resolve-custom-ops / prepare-for-nki passes using
        # the `nki` wheel's Python bindings. Marked as Python-phase so the
        # driver below dispatches to `linalg_to_nisa_py` instead of nkipy-opt.
        'py:linalg-to-nisa',                                                     # 24
    ]

    # Slice passes if stop_after is provided
    if stop_after is not None:
        if isinstance(stop_after, int):
            passes = passes[:stop_after]
        elif isinstance(stop_after, str):
            # Support "name:N" to select the Nth occurrence (1-indexed).
            # "py:<name>" is also recognized — disambiguate by checking if
            # the tail after the final ":" is an integer.
            name = stop_after
            nth = 1
            if ':' in stop_after:
                head, tail = stop_after.rsplit(':', 1)
                if tail.isdigit():
                    name, nth = head, int(tail)
            occurrence = 0
            found_idx = None
            for i, p in enumerate(passes):
                # Strip `py:` prefix for matching so users can request the
                # same pass by either `py:linalg-to-nisa` or `linalg-to-nisa`.
                raw = p[len('py:'):] if p.startswith('py:') else p
                base_name = raw.split('=')[0].split('"')[0].strip()
                req_name = name[len('py:'):] if name.startswith('py:') else name
                if base_name == req_name:
                    occurrence += 1
                    if occurrence == nth:
                        found_idx = i
                        break
            if found_idx is None:
                available = [
                    (p[len('py:'):] if p.startswith('py:') else p)
                    .split('=')[0].split('"')[0].strip()
                    for p in passes
                ]
                raise ValueError(
                    f"Pass '{stop_after}' not found in pipeline. "
                    f"Available passes: {available}"
                )
            passes = passes[:found_idx + 1]
        else:
            raise TypeError(f"stop_after must be int, str, or None, got {type(stop_after)}")

    return _run_passes_with_python_dispatch(
        mlir_module,
        passes,
        target=target,
        print_ir_after_all=print_ir_after_all,
        dump_dir=dump_dir,
        print_debuginfo=print_debuginfo,
        print_generic=print_generic,
    )


def _run_passes_with_python_dispatch(
    mlir_module: str,
    passes: list[str],
    target: str,
    print_ir_after_all: bool,
    dump_dir: str | None,
    print_debuginfo: bool,
    print_generic: bool,
) -> str:
    """Run a pass list, batching consecutive nkipy-opt passes and dispatching
    any `py:<name>` entries to their Python implementation.

    Having a single driver keeps `dump_dir` numbering coherent across the
    C++/Python boundary: every pass — whether it runs in nkipy-opt or in
    Python — writes the same `NN_<pass>.mlir` artifact.
    """
    current = mlir_module

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "00_input.mlir"), 'w') as f:
            f.write(str(current))

    batch: list[str] = []
    batch_start_idx = 1

    def flush_batch(next_idx: int) -> None:
        nonlocal current, batch
        if not batch:
            return
        if dump_dir:
            # Run each pass separately when dumping so we save per-pass IR.
            for j, p in enumerate(batch):
                current = run_nkipy_opt_passes(
                    current, [p], print_ir_after_all,
                    print_debuginfo=print_debuginfo, print_generic=print_generic,
                )
                simple_name = p.split('=')[0].split('"')[0].strip()
                filename = f"{batch_start_idx + j:02d}_{simple_name}.mlir"
                with open(os.path.join(dump_dir, filename), 'w') as f:
                    f.write(current)
        else:
            current = run_nkipy_opt_passes(
                current, batch, print_ir_after_all,
                print_debuginfo=print_debuginfo, print_generic=print_generic,
            )
        batch = []

    for i, pass_name in enumerate(passes, start=1):
        if pass_name.startswith('py:'):
            flush_batch(i)
            py_name = pass_name[len('py:'):]
            current = _run_python_pass(
                py_name, current, target=target, print_generic=print_generic,
            )
            if dump_dir:
                filename = f"{i:02d}_{py_name}.mlir"
                with open(os.path.join(dump_dir, filename), 'w') as f:
                    f.write(current)
            batch_start_idx = i + 1
        else:
            if not batch:
                batch_start_idx = i
            batch.append(pass_name)

    flush_batch(len(passes) + 1)
    return current


def _run_python_pass(
    name: str, mlir_text: str, target: str, print_generic: bool = False,
) -> str:
    """Dispatch a `py:<name>` pass to its Python implementation."""
    if name == 'linalg-to-nisa':
        # Imported lazily because the NKI wheel and upstream `mlir` are only
        # required for this pass; tests that stop before phase 5 do not need
        # either installed.
        from .linalg_to_nisa_py import linalg_to_nisa
        return linalg_to_nisa(mlir_text, target=target, print_generic=print_generic)
    raise ValueError(f"Unknown Python pass: {name!r}")


# Export the main interface
__all__ = [
    'get_nkipy_opt_path',
    'run_nkipy_opt_passes',
    'apply_complete_knob_pipeline',
]
