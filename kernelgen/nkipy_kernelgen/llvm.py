import os
import ctypes
import ml_dtypes
import numpy as np

from mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
    InsertionPoint,
    FloatAttr,
    IntegerAttr,
    F32Type,
    IntegerType,
    MemRefType,
    FunctionType,
    TypeAttr,
)
from mlir.dialects import tensor, arith, linalg

from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
)

from nkipy_kernelgen.utils import (
    get_func_inputs_outputs,
    find_func_in_module,
    get_bitwidth_from_type,
    ctype_map,
    np_supported_types,
    np_type_to_str,
    get_np_struct_type,
    create_output_struct,
    extract_out_np_arrays_from_out_struct,
    ranked_memref_to_numpy,
)

# Import nkipy dialect for registration
from nkipy_kernelgen._mlir.dialects import nkipy as nkipy_d


def extract_and_clean_func_from_module(mlir_module_str: str):
    """
    Extract func.func operation from MLIR module and remove nkipy annotations.
    
    This utility is particularly useful for MLIR modules that have been through
    knob-driven-tiling or other passes that add transform dialect IR and nkipy
    annotations that need to be stripped before LLVM execution.
    
    Steps performed:
    1. Strip #nisa.mem<...> memory spaces from memref type syntax (text pre-processing)
    2. Parse the MLIR module
    3. Find the func.func operation (ignoring transform.named_sequence)
    4. Remove nkipy.annotate operations
    5. Remove nkipy.op_id and memory_space operation attributes
    6. Strip any remaining memory spaces from memref types programmatically
    7. Create a clean MLIR module with just the function
    
    Args:
        mlir_module_str: MLIR module as a string, potentially containing
                         transform.named_sequence and nkipy annotations
    
    Returns:
        tuple: (clean_mlir_str, function_name)
            - clean_mlir_str: MLIR module string with only the cleaned func.func
            - function_name: The name of the extracted function
    
    Raises:
        ValueError: If no func.func operation is found in the module
        
    Example:
        clean_mlir, func_name = extract_and_clean_func_from_module(tiled_mlir)
        runner = LLVMModule(clean_mlir, func_name)
    """
    import re
    # Strip memory-space attributes from memref type syntax before parsing.
    # We accept two forms and drop both:
    #   1. #nisa.mem<...>         — post-annotate-memory-space NISA dialect form
    #   2. N : i32                — raw IntegerAttr form that
    #                                nkipy's MemSpaceEnumAttr prints as
    #
    # The MLIR parser can't handle the NISA-dialect attribute without the
    # dialect registered. More importantly, a non-zero integer memspace
    # becomes `!llvm.ptr<N>` during memref-to-llvm lowering, and the
    # runtime helpers (`@free` etc.) only accept `!llvm.ptr` in address
    # space 0 — leaving the memspace on would trip the LLVM verifier
    # with `operand type mismatch … '!llvm.ptr<N>' != '!llvm.ptr'` when
    # lowering `memref.dealloc` on an HBM/SBUF buffer.
    mlir_module_str = re.sub(r',\s*#nisa\.mem<[^>]+>', '', mlir_module_str)
    mlir_module_str = re.sub(r',\s*\d+\s*:\s*i32>', '>', mlir_module_str)

    with Context() as ctx:
        # Register nkipy dialect to handle nkipy operations
        nkipy_d.register_dialect(ctx)

        # Allow unregistered dialects temporarily to parse the module
        ctx.allow_unregistered_dialects = True

        # Parse the MLIR module and clean it in-place (preserving all
        # module-level declarations like memref.global that the function
        # may reference).
        new_module = Module.parse(mlir_module_str, ctx)

        # Inline reference_impl regions from nkipy ops (e.g. nkipy.gather)
        # so the LLVM JIT only sees standard linalg/tensor ops.
        # The inline pass also folds tensor.extract(to_tensor(memref)) →
        # memref.load(memref) patterns left after inlining into post-
        # bufferization IR.  Canonicalize then folds remaining
        # to_buffer(to_tensor(x)) chains.
        from nkipy_kernelgen.transforms.nkipy_opt import run_nkipy_opt_passes
        inlined_str = run_nkipy_opt_passes(
            str(new_module),
            ["inline-nkipy-reference", "canonicalize"],
        )
        new_module = Module.parse(inlined_str, ctx)

        # Strip the transform.with_named_sequence module attribute if present
        module_op = new_module.operation
        if "transform.with_named_sequence" in module_op.attributes:
            del module_op.attributes["transform.with_named_sequence"]

        # Walk module body: find the func, and collect ops to erase
        # (transform sequences, nkipy.annotate, etc.)
        actual_func_name = None
        ops_to_erase = []

        def walk_and_mark(op):
            if op.name == "nkipy.annotate":
                ops_to_erase.append(op)
            if "nkipy.op_id" in op.attributes:
                del op.attributes["nkipy.op_id"]
            if "memory_space" in op.attributes:
                del op.attributes["memory_space"]
            for region in op.regions:
                for block in region:
                    for nested_op in block:
                        walk_and_mark(nested_op)

        for op in new_module.body.operations:
            op_name = op.operation.name
            if op_name == "func.func" and actual_func_name is None:
                actual_func_name = str(op.attributes["sym_name"]).strip('"')
                walk_and_mark(op.operation)
            elif op_name == "transform.named_sequence":
                ops_to_erase.append(op.operation)

        if actual_func_name is None:
            raise ValueError("Could not find func.func operation in MLIR module")

        # Erase marked operations
        for op in ops_to_erase:
            op.erase()

        # For CPU simulation: Zero-fill all tensor.empty operations
        _zero_fill_empty_tensors_ir(new_module)

        # For CPU simulation: Zero-fill all memref.alloc operations
        # (needed for post-bufferization IR, e.g. --stop=5+)
        _zero_fill_alloc_memrefs_ir(new_module)

        # Get clean MLIR string
        clean_mlir = str(new_module)
    
    return clean_mlir, actual_func_name


def _zero_fill_empty_tensors_ir(module: Module):
    """
    Walk the IR and replace tensor.empty operations with zero-filled tensors.
    
    For each tensor.empty:
    1. Create a zero constant of the appropriate element type
    2. Create a linalg.fill operation to fill the tensor with zero
    3. Replace all uses of tensor.empty with the filled tensor
    
    This ensures CPU simulation matches target ASIC behavior (empty tensors are zero).
    """
    # Collect all tensor.empty operations to process
    empty_ops = []
    
    def collect_empty_ops(op):
        if op.name == "tensor.empty":
            empty_ops.append(op)
        for region in op.regions:
            for block in region:
                for nested_op in block:
                    collect_empty_ops(nested_op)
    
    # Walk the module to collect all tensor.empty ops
    for op in module.body.operations:
        collect_empty_ops(op.operation)
    
    # Process each tensor.empty operation
    for empty_op in empty_ops:
        # Get the result type (should be a tensor type)
        result = empty_op.results[0]
        tensor_type = result.type
        
        # Extract element type from the tensor type
        try:
            elem_type = tensor_type.element_type
        except:
            # If we can't get element type, skip this operation
            continue
        
        # Get location from the empty op
        loc = empty_op.location
        
        # Create zero constant based on element type
        with loc, InsertionPoint.at_block_begin(empty_op.operation.block):
            # Determine zero value based on type
            if str(elem_type).startswith('f'):
                # Float type - create 0.0
                zero_const = arith.ConstantOp(elem_type, FloatAttr.get(elem_type, 0.0))
            elif str(elem_type).startswith('i'):
                # Integer type - create 0
                zero_const = arith.ConstantOp(elem_type, IntegerAttr.get(elem_type, 0))
            else:
                # Unknown type, skip
                continue
        
        # Move the insertion point right after tensor.empty  
        with loc, InsertionPoint(empty_op):
            new_empty = tensor.EmptyOp(list(tensor_type.shape), tensor_type.element_type, loc=loc)

            fill_op = linalg.FillOp([tensor_type], [zero_const.result], [new_empty.result], loc=loc)

            region = fill_op.regions[0]
            if len(region.blocks) == 0:
                block = region.blocks.append(elem_type, elem_type)
                with InsertionPoint(block):
                    linalg.YieldOp([block.arguments[0]], loc=loc)
            
            # Replace all uses of the original tensor.empty with linalg.fill result
            result.replace_all_uses_with(fill_op.results[0])


def _zero_fill_alloc_memrefs_ir(module: Module):
    """
    Walk the IR and zero-fill all memref.alloc operations.

    After bufferization (e.g., --stop=5+), tensor.empty becomes memref.alloc.
    Unlike tensor.empty (semantically undefined), memref.alloc produces truly
    uninitialized memory on CPU which may contain garbage/NaN. This function
    inserts linalg.fill operations right after each memref.alloc to
    zero-initialize the buffer for correct CPU simulation.
    """
    # Collect all memref.alloc operations
    alloc_ops = []

    def collect_alloc_ops(op):
        if op.name == "memref.alloc":
            alloc_ops.append(op)
        for region in op.regions:
            for block in region:
                for nested_op in block:
                    collect_alloc_ops(nested_op)

    for op in module.body.operations:
        collect_alloc_ops(op.operation)

    # Process each memref.alloc: insert linalg.fill right after it
    for alloc_op in alloc_ops:
        result = alloc_op.results[0]
        memref_type = result.type

        try:
            elem_type = memref_type.element_type
        except:
            continue

        loc = alloc_op.location

        # Find the next operation after this alloc in its parent block.
        # We insert the fill before the next op (effectively after the alloc).
        parent_block = alloc_op.operation.block
        found_alloc = False
        next_op = None
        for block_op in parent_block:
            if found_alloc:
                next_op = block_op
                break
            if block_op.operation == alloc_op.operation:
                found_alloc = True

        if next_op is None:
            continue

        # Insert zero constant + linalg.fill before next_op (= after alloc)
        with loc, InsertionPoint(next_op):
            # Create zero constant based on element type
            if str(elem_type).startswith('f'):
                zero_const = arith.ConstantOp(
                    elem_type, FloatAttr.get(elem_type, 0.0)
                )
            elif str(elem_type).startswith('i'):
                zero_const = arith.ConstantOp(
                    elem_type, IntegerAttr.get(elem_type, 0)
                )
            else:
                continue

            fill_op = linalg.FillOp(
                [], [zero_const.result], [result], loc=loc
            )

            region = fill_op.regions[0]
            if len(region.blocks) == 0:
                fill_block = region.blocks.append(elem_type, elem_type)
                with InsertionPoint(fill_block):
                    linalg.YieldOp([fill_block.arguments[0]], loc=loc)


class LLVMModule:
    def __init__(self, mod, top_func_name, ext_libs=None):
        # Copy the module to avoid modifying the original one
        with Context() as ctx:
            # Register the nkipy dialect to handle nkipy.annotate ops
            nkipy_d.register_dialect(ctx)
            
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs

            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)

            # Run through lowering passes
            pm = PassManager.parse(
                # "builtin.module("
                # # used for lowering tensor.empty
                # "empty-tensor-to-alloc-tensor,"
                # # translate tensor dialect (virtual) to memref dialect (physical)
                # "one-shot-bufferize{bufferize-function-boundaries},"
                # # used for lowering memref.subview
                # "expand-strided-metadata,"
                # # common lowering passes
                # "func.func(convert-linalg-to-affine-loops),lower-affine"
                # ")"
                "builtin.module("
                "one-shot-bufferize{bufferize-function-boundaries=1},"
                "func.func(convert-linalg-to-loops),"
                "func.func(lower-affine)"
                ")"
            )

            pm.run(self.module.operation)
            # self.intermediate_module = self.module.operation.clone()

            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()

            # https://github.com/llvm/llvm-project/issues/52945
            # Final lowering
            pm = PassManager.parse(
                "builtin.module("
                "func.func(convert-scf-to-cf),"
                "func.func(arith-expand),"
                "expand-strided-metadata,"
                "lower-affine,"
                "convert-math-to-llvm,"
                "convert-arith-to-llvm,"
                "finalize-memref-to-llvm,"
                "convert-func-to-llvm,"
                "convert-cf-to-llvm,"
                "reconcile-unrealized-casts"
                ")"
            )
            pm.run(self.module.operation)

            # Add shared library for MLIR runner utils (provides memrefCopy etc.)
            # Resolve the LLVM lib directory: LLVM_INST env var takes priority,
            # then llvm-config --libdir, then no shared libs.
            llvm_lib_dir = None
            if os.getenv("LLVM_INST"):
                llvm_lib_dir = os.path.join(os.getenv("LLVM_INST"), "lib")
            else:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["llvm-config", "--libdir"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        llvm_lib_dir = result.stdout.strip()
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            if llvm_lib_dir is not None:
                shared_libs = [
                    os.path.join(llvm_lib_dir, "libmlir_runner_utils.so"),
                    os.path.join(llvm_lib_dir, "libmlir_c_runner_utils.so"),
                ]
            else:
                shared_libs = []

            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )

    # pylint: disable=too-many-branches
    def __call__(self, *args):
        """
        Reference:
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/python/execution_engine.py
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/Integration/Dialect/SparseTensor/python/test_SpMM.py
        """
        input_types = self.in_types
        arg_ptrs = []
        new_args = []
        assert len(args) == len(
            input_types
        ), f"# of input arguments mismatch, got {len(args)} but expected {len(input_types)}"

        # 1. Construct argument pointers
        for arg, (target_in_type, shape, is_memref) in zip(args, input_types):
            if not is_memref:  # scalar
                if isinstance(arg, int):
                    if target_in_type != "i32":
                        raise RuntimeError(
                            f"Input type mismatch: {target_in_type} vs i32. Please use NumPy array"
                            " to wrap the data to avoid possible result mismatch"
                        )
                    bitwidth = get_bitwidth_from_type(target_in_type)
                    signed = "i" if target_in_type.startswith("i") else "ui"
                    dtype = ctype_map[f"{signed}{bitwidth}"]
                    c_int_p = dtype * 1
                    arg_ptrs.append(c_int_p(arg))

                elif isinstance(arg, float):
                    if target_in_type != "f32":
                        raise Warning(
                            f"Input type mismatch: {target_in_type} vs f32. Please use NumPy array"
                            " to wrap the data to avoid possible result mismatch"
                        ).warn()
                    if target_in_type == "f16":
                        c_float_p = ctypes.c_int16 * 1
                        arg = np.float16(arg).view(np.int16)
                    elif target_in_type == "bf16":
                        c_float_p = ctypes.c_int16 * 1
                        arg = ml_dtypes.bfloat16(arg).view(np.int16)
                    elif target_in_type == "f32":
                        c_float_p = ctypes.c_float * 1
                    else:  # f64
                        c_float_p = ctypes.c_double * 1
                    arg_ptrs.append(c_float_p(arg))

                else:
                    raise RuntimeError(
                        "Unsupported input type. Please use NumPy array to wrap the data if other"
                        " data types are needed as inputs."
                    )

            else:  # memref
                if not arg.flags["C_CONTIGUOUS"]:
                    raise RuntimeError(
                        "The input data is not contiguous. Please use np.ascontiguousarray to change the layout first."
                    )
                if not isinstance(arg.dtype, np.dtypes.VoidDType):
                    np_type = np_type_to_str(arg.dtype)
                    if np_type != target_in_type:
                        import warnings
                        warnings.warn(
                            f"Input type mismatch: {np_type} vs {target_in_type}",
                            RuntimeWarning,
                        )

                if target_in_type in np_supported_types:
                    target_np_type = np_supported_types[target_in_type]
                    if arg.dtype != target_np_type:
                        # avoid changing the address of the original array
                        arg = arg.astype(target_np_type)
                else:
                    raise RuntimeError(
                        f"Unsupported input type: {target_in_type}, "
                        f"please use a supported type or wrap the scalar as an array"
                    )
                arg_ptrs.append(
                    ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
                )
            new_args.append(arg)

        # 2. Construct return pointers
        # Need to verify the return variable is not the same as the input
        result_types = self.out_types
        # Returns as arguments: no return value from the top function
        if len(result_types) == 0:
            self.execution_engine.invoke(self.top_func_name, *arg_ptrs)
            for arg, new_arg, (target_in_type, shape, is_memref) in zip(
                args, new_args, input_types
            ):
                if is_memref:
                    arg[:] = new_arg
            return
        # Return inner variables: return one or more values allocated inside kernel
        # For two or more return values, llvm.emit_c_interface will return a struct
        # Therefore, for functions that return values, we need to separate two cases:
        # 1. return one value: no need to create a struct
        # 2. return two or more values: need to create a struct
        # In any case, we prepare a pointer of pointer to the return object
        # which is ready to be passed to the invoke function.
        if len(result_types) == 1:  # exactly one return value
            result_type, shape, is_memref = result_types[0]
            if is_memref:
                # After bufferization, tensors (including rank-0 tensors like tensor<f32>)
                # become memrefs (including rank-0 memrefs like memref<f32>)
                if result_type in ctype_map:
                    dtype = ctype_map[result_type]
                elif result_type.startswith("i") or result_type.startswith("ui"):
                    bitwidth = get_bitwidth_from_type(result_type)
                    dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
                else:
                    raise RuntimeError("Unsupported return type")
                # Create an empty memref descriptor (rank-0 for scalars, rank-N for tensors)
                return_desc = make_nd_memref_descriptor(len(shape), dtype)()
                return_ptr = ctypes.pointer(ctypes.pointer(return_desc))
            else:  # bare scalar
                if result_type in ctype_map:
                    dtype = ctype_map[result_type]
                else:
                    signed = "i" if result_type.startswith("i") else "ui"
                    bitwidth = get_bitwidth_from_type(result_type)
                    dtype = ctype_map[f"{signed}{bitwidth}"]

                dtype_p = dtype * 1
                # -1/-1.0 is a placeholder
                return_ptr = dtype_p(-1 if not result_type in {"f32", "f64"} else 1.0)
                
        else:  # multiple return values
            # we assume all return values are memrefs
            out_memref_descs = []
            for elt_res_type, elt_shape, is_memref in result_types:
                if not is_memref:
                    raise RuntimeError(
                        "When returning multiple values, we only support all tensors/memrefs."
                    )
                if elt_res_type in ctype_map:
                    dtype = ctype_map[elt_res_type]
                elif elt_res_type.startswith("i") or elt_res_type.startswith("ui"):
                    bitwidth = get_bitwidth_from_type(elt_res_type)
                    dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
                else:
                    raise RuntimeError("Unsupported return type")
                # Create an empty tensor
                return_desc = make_nd_memref_descriptor(len(elt_shape), dtype)()
                out_memref_descs.append(return_desc)
            # Create a struct
            out_struct = create_output_struct(out_memref_descs)
            return_ptr = ctypes.pointer(ctypes.pointer(out_struct))

        # 3. Invoke the function and return the result
        if len(result_types) == 1:
            result_type, shape, is_memref = result_types[0]
            if is_memref:
                # INVOKE - memref return (including rank-0 memrefs)
                self.execution_engine.invoke(self.top_func_name, return_ptr, *arg_ptrs)
                ret = ranked_memref_to_numpy(return_ptr[0][0])
                if result_type == "f16":
                    ret = np.array(ret, dtype=np.int16).view(np.float16)
                elif result_type == "bf16":
                    ret = np.array(ret, dtype=np.int16).view(ml_dtypes.bfloat16)

                # For rank-0 tensors, extract the scalar value
                if len(shape) == 0:
                    ret = ret.item()
            else:
                # INVOKE - bare scalar return
                self.execution_engine.invoke(self.top_func_name, *arg_ptrs, return_ptr)
                ret = return_ptr[0]
                if result_type == "f16":
                    ret = np.int16(ret).view(np.float16)
                elif result_type == "bf16":
                    ret = np.int16(ret).view(ml_dtypes.bfloat16)
        else:  # multiple returns, assume all memref
            # INVOKE
            self.execution_engine.invoke(self.top_func_name, return_ptr, *arg_ptrs)
            ret_raw_np = extract_out_np_arrays_from_out_struct(
                return_ptr, len(result_types)
            )
            # pylint: disable=redefined-variable-type
            ret = []
            for np_arr, (res_type, _, _) in zip(ret_raw_np, result_types):
                if res_type == "f16":
                    ret_i = np.array(np_arr, dtype=np.int16).view(np.float16)
                elif res_type == "bf16":
                    ret_i = np.array(np_arr, dtype=np.int16).view(ml_dtypes.bfloat16)
                else:
                    ret_i = np_arr
                ret.append(ret_i)
        return ret


if __name__ == "__main__":
    # Minimal example: parse MLIR from string, JIT-run it, and compare with NumPy.
    mlir_src = r"""
    module {
      func.func @top(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %init = tensor.empty() : tensor<4x4xf32>
        %C = linalg.generic
          { indexing_maps = [ affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)> ],
            iterator_types = ["parallel", "parallel"] }
          ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
          outs(%init : tensor<4x4xf32>) {
            ^bb0(%a: f32, %b: f32, %acc: f32):
              %sum = arith.addf %a, %b : f32
              linalg.yield %sum : f32
          } -> tensor<4x4xf32>
        return %C : tensor<4x4xf32>
      }
    }
    """

    # Build module and engine
    with Context():
        mod = Module.parse(mlir_src)
    runner = LLVMModule(mod, "top")

    # Inputs and NumPy reference
    A = np.random.rand(4, 4).astype(np.float32)
    B = np.random.rand(4, 4).astype(np.float32)
    ref = A + B

    # Run and compare
    out = runner(A.copy(), B.copy())
    ok = np.allclose(out, ref, rtol=1e-5, atol=1e-6)
    
    print("Match with NumPy:", ok)
    if not ok:
        print("MLIR:", out)
        print("NumPy:", ref)
