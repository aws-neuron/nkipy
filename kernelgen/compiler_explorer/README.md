# Compiler Explorer Integration for NKIPy

This directory contains the configuration to run [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer) locally with NKIPy as a compiler backend.

## Quick Start

```bash
# Run the setup script (clones CE, installs deps, configures, and starts)
./setup.sh
```

Then open http://localhost:10240 in your browser.

## Usage in Compiler Explorer

1. Select **Python** as the language (left panel)
2. Select **NKIPy MLIR** as the compiler (right panel dropdown)
3. Write your traced kernel:

```python
import numpy as np
from nkipy_kernelgen import trace, knob

@trace(input_specs=[
    ((128, 128), "f32"),
    ((128, 128), "f32"),
])
def matmul_kernel(A, B):
    C = np.matmul(A, B)
    knob.knob(C, mem_space="Sbuf", tile_size=[64, 64, 64])
    return C
```

4. The right panel shows the compiled MLIR output

## Compiler Options

Add these in the "Compiler options" box to control output:

| Option | Description |
|--------|-------------|
| `--stop=0` | Trace only — initial MLIR before any passes |
| `--stop=2` | After `prepare-arithmetic` (div → mul+reciprocal) |
| `--stop=7` | After `apply-and-strip-transforms` (tiling applied) |
| `--stop=10` | After bufferization + canonicalize |
| `--stop=16` | After memory space annotation + cleanup |
| `--stop=17` | After layout legalization |
| `--stop=24` | After NISA lowering (same as omitting `--stop`) |
| `--sim` | Run BIR simulation and verify against NumPy reference |
| `--sim --stop=N` | Run LLVM JIT simulation on intermediate IR at stop point |
| `--hw` | Compile to NEFF and execute on Trainium hardware (requires device) |
| `--target=trn1\|trn2\|trn3` | Target hardware (default: `trn2`) |
| `--raw` | Clean MLIR output without `.loc`/`.file` annotations |

## Example Workflow

To debug how tiling transforms your kernel:

1. Write your kernel in the left panel
2. Set compiler options to `--stop=0` to see the initial traced MLIR
3. Change to `--stop=7` to see loops introduced by tiling
4. Change to `--stop=10` to see bufferized memref IR
5. Change to `--stop=16` for memory space annotations
6. Remove `--stop` to see the final NISA MLIR

## Files

```
compiler_explorer/
├── nkipy_compiler.py      # Main compiler wrapper
├── nkipy_ce_wrapper.sh    # Shell wrapper for CE
├── setup.sh               # Setup and run script
├── README.md              # This file
├── config/
│   ├── nkipy.local.properties   # CE config (Python language)
│   ├── c.nkipy.properties       # Alternative: custom language
│   └── example.nkipy            # Example kernel
```

## Manual Setup

If you prefer to set up manually:

```bash
# 1. Clone Compiler Explorer
git clone https://github.com/compiler-explorer/compiler-explorer.git
cd compiler-explorer

# 2. Install dependencies
npm install

# 3. Copy config (edit paths first!)
cp /path/to/NKIPyKernelGen/compiler_explorer/config/nkipy.local.properties etc/config/

# 4. Edit the config to fix the wrapper path
vim etc/config/nkipy.local.properties

# 5. Start
npm run dev
```

## Troubleshooting

**"No @trace decorated function found"**
- Ensure your function has the `@trace(input_specs=[...])` decorator

**Import errors**
- The wrapper sets PYTHONPATH automatically
- Check that nkipy_kernelgen is installed or accessible

**Node.js version error**
- Compiler Explorer requires Node.js 18+
- Use `nvm` to manage Node versions

## Limitations

- Compiler Explorer shows source → single output (one `--stop` level at a time)
- For side-by-side comparison of passes, use the dumped artifacts with `diff` or `vimdiff`
