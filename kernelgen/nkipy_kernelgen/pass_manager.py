"""
MLIR pass management utilities.

This module provides utilities for applying MLIR transformation passes to modules.
"""


def apply_passes(mlir_module, passes):
    """
    Apply MLIR passes and/or custom Python transformations to a module and return the transformed module.
    
    This function allows you to apply a sequence of transformation passes to an MLIR module.
    You can specify passes in multiple formats and mix them together:
    
    Args:
        mlir_module: The MLIR module to transform
        passes: Can be:
            - A string: Complete pass pipeline (e.g., "builtin.module(func.func(...))")
            - A list containing:
                * Pass name strings (e.g., "linalg-generalize-named-ops")
                * Callable functions that take and return a module
            - A single callable function

    Returns:
        The transformed MLIR module

    Examples:
        >>> from nkipy_kernelgen import apply_passes
        >>>
        >>> # Method 1: Using a list of pass names
        >>> transformed = apply_passes(module, ["linalg-generalize-named-ops", "linalg-fuse-elementwise-ops"])
        >>>
        >>> # Method 2: Using a complete pipeline string
        >>> transformed = apply_passes(module, "builtin.module(func.func(linalg-generalize-named-ops))")
    """
    from mlir import passmanager
    
    # Module-level passes that should not be nested in func.func
    MODULE_LEVEL_PASSES = {
        "one-shot-bufferize",
        "canonicalize",
        "cse",
        "symbol-dce",
        "inline",
    }
    
    # Note: convert-linalg-to-loops is a function-level pass
    
    def is_module_level_pass(pass_name):
        """Check if a pass should run at module level rather than function level."""
        # Extract the base pass name (without options)
        base_name = pass_name.split("{")[0].strip()
        return base_name in MODULE_LEVEL_PASSES
    
    def apply_accumulated_passes(module, func_passes, module_passes):
        """Apply accumulated function-level and module-level passes."""
        # Apply module-level passes FIRST (at module level)
        # This is critical for passes like one-shot-bufferize that need to run
        # before subsequent function-level passes like convert-linalg-to-loops
        if module_passes:
            pass_list_str = ",".join(module_passes)
            pass_pipeline = f"builtin.module({pass_list_str})"
            with module.context:
                pm = passmanager.PassManager.parse(pass_pipeline)
                pm.run(module.operation)
        
        # Apply function-level passes SECOND (nested in func.func)
        if func_passes:
            pass_list_str = ",".join(func_passes)
            pass_pipeline = f"builtin.module(func.func({pass_list_str}))"
            with module.context:
                pm = passmanager.PassManager.parse(pass_pipeline)
                pm.run(module.operation)
        
        return module
    
    # Handle single callable function
    if callable(passes):
        return passes(mlir_module)
    
    # Handle string (complete pipeline)
    if isinstance(passes, str):
        with mlir_module.context:
            pm = passmanager.PassManager.parse(passes)
            pm.run(mlir_module.operation)
        return mlir_module
    
    # Handle list of passes (mix of strings and callables)
    if isinstance(passes, list):
        # Separate passes into function-level and module-level
        func_passes = []
        module_passes = []
        
        for item in passes:
            if callable(item):
                # If we have accumulated MLIR passes, apply them first
                if func_passes or module_passes:
                    mlir_module = apply_accumulated_passes(mlir_module, func_passes, module_passes)
                    func_passes = []
                    module_passes = []
                
                # Apply the Python transformation
                mlir_module = item(mlir_module)
            elif isinstance(item, str):
                # Categorize as module-level or function-level pass
                if is_module_level_pass(item):
                    module_passes.append(item)
                else:
                    func_passes.append(item)
            else:
                raise TypeError(f"Pass must be a string or callable, got {type(item)}")
        
        # Apply any remaining MLIR passes
        if func_passes or module_passes:
            mlir_module = apply_accumulated_passes(mlir_module, func_passes, module_passes)
        
        return mlir_module
    
    raise TypeError(f"Passes must be a string, callable, or list, got {type(passes)}")
