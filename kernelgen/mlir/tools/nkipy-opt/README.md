# nkipy-opt - MLIR Optimizer Driver

`nkipy-opt` is a command-line tool for testing and running MLIR passes on nkipy dialect code. It is analogous to LLVM's `mlir-opt` tool but specifically configured for the nkipy project.

## Building

To build `nkipy-opt`, you need to first set up your environment and then build the mlir project.

### Prerequisites

1. Source the environment setup script to configure LLVM/MLIR paths:
```bash
source setup.sh
```

This will set up the required environment variables:
- `LLVM_DIR` - Path to LLVM CMake config
- `MLIR_DIR` - Path to MLIR CMake config
- `PATH` - Includes LLVM binaries

2. Ensure you have a compatible CMake version (3.20.0 or higher)

### Build Steps

```bash
cd mlir
rm -rf build  # Clean any existing build
mkdir build
cd build

# Configure with CMake
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=${LLVM_DIR} \
  -DMLIR_DIR=${MLIR_DIR}

# Build nkipy-opt
ninja nkipy-opt
```

The resulting binary will be placed in `mlir/build/bin/nkipy-opt`.

Alternatively, if Ninja is not available, you can use Make:
```bash
cmake .. -DLLVM_DIR=${LLVM_DIR} -DMLIR_DIR=${MLIR_DIR}
make nkipy-opt -j$(nproc)
```

## Usage

### Basic Usage

```bash
nkipy-opt [options] <input-file>
```

### Common Options

- `--help`: Display available options and passes
- `--show-dialects`: Show all registered dialects
- `--print-ir-after-all`: Print IR after each pass
- `--mlir-print-ir-after-change`: Only print IR after a pass if it changed
- `--mlir-timing`: Display timing information for passes

### Running Passes

To run a specific pass:

```bash
nkipy-opt --memref-dce input.mlir
```

To chain multiple passes:

```bash
nkipy-opt --pass-pipeline='builtin.module(memref-dce,canonicalize)' input.mlir
```

### Available Nkipy Passes

- `--memref-dce`: Remove MemRefs that are never loaded from

### Example Workflows

#### 1. View Available Passes

```bash
nkipy-opt --help
```

#### 2. Run Dead Code Elimination

```bash
nkipy-opt --memref-dce example.mlir -o output.mlir
```

#### 3. Run with Timing Information

```bash
nkipy-opt --memref-dce --mlir-timing example.mlir
```

#### 4. Print IR After Each Pass

```bash
nkipy-opt --memref-dce --print-ir-after-all example.mlir
```

#### 5. Run Standard MLIR Passes

Since `nkipy-opt` registers all standard MLIR passes, you can also use standard passes:

```bash
nkipy-opt --canonicalize --cse example.mlir
```

## Input File Format

Input files should be valid MLIR text format. Example:

```mlir
module {
  func.func @example(%arg0: memref<10xf32>) -> f32 {
    %0 = memref.alloc() : memref<10xf32>
    %c0 = arith.constant 0 : index
    %1 = memref.load %arg0[%c0] : memref<10xf32>
    return %1 : f32
  }
}
```

## Integration with Build System

The tool is automatically built as part of the nkipy-kg project when you build the mlir subdirectory.

## Troubleshooting

If you encounter build errors:

1. Ensure MLIR and LLVM are properly installed and found by CMake
2. Check that all required dialects are registered
3. Verify the include paths are correct in CMakeLists.txt

For runtime errors:

1. Use `--help` to see all available options
2. Use `--show-dialects` to verify the nkipy dialect is registered
3. Enable verbose output with `--mlir-print-ir-after-all`
