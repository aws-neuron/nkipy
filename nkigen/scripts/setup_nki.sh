#!/bin/bash
#
# Environment setup for NKIPyKernelGen. Source this — do not run it.
#
#   source scripts/setup_nki.sh
#
# What it does:
#   - Points the build system at a pre-built LLVM+MLIR install (the C++
#     passes in mlir/ link against libMLIR*.a and use its TableGen).
#   - Adds LLVM's bin/ to PATH and lib/ to LD_LIBRARY_PATH so `nkipy-opt`
#     and friends can find mlir-tblgen at build time and shared libs at
#     runtime.
#   - Adds LLVM's `mlir_core` Python package to PYTHONPATH — the Python
#     linalg→NISA phase uses upstream MLIR bindings to re-parse IR, and
#     those bindings must match the C++ MLIR version nkipy-opt was built
#     against.
#
# What it does NOT do:
#   - Activate a virtualenv. Activate your venv first, then source this.
#   - Install LLVM. See "Installing LLVM" below.
#
# ---------------------------------------------------------------------------
# Installing LLVM
# ---------------------------------------------------------------------------
# `nkipy-opt` (the C++ passes) needs a full MLIR SDK install: headers,
# static libraries, CMake config files, and `mlir-tblgen`. None of the
# MLIR PyPI wheels (`mlir-core`, `mlir-python-bindings`, `nki`) ship the
# C++ side — they're Python bindings only.
#
# Three options, in order of convenience:
#
#   1. Download a pre-built tarball (recommended).  Set LLVM_TARBALL_URL
#      to a URL that returns a tar.gz of the install prefix and re-source
#      this script; it will fetch+extract on first use.  Expected layout
#      inside the tarball: bin/, lib/, include/, lib/cmake/{llvm,mlir}/,
#      python_packages/mlir_core/.
#
#   2. Distro packages (e.g. `apt install mlir-20-tools libmlir-20-dev`
#      on recent Ubuntu).  Set LLVM_INSTALL_PREFIX to the resulting path
#      (something like /usr/lib/llvm-20).  Warning: distro MLIR may not
#      match the commit `nkipy-opt` was developed against — API drift
#      between LLVM versions can break the C++ build.
#
#   3. Build from source.  ~45 min on 8 cores, ~20 GB build dir.
#        git clone https://github.com/llvm/llvm-project
#        cd llvm-project && git checkout <see REVISION in prebuilt tarball>
#        cmake -S llvm -B build -G Ninja \
#              -DLLVM_ENABLE_PROJECTS=mlir \
#              -DCMAKE_BUILD_TYPE=Release \
#              -DCMAKE_INSTALL_PREFIX=/opt/llvm-mlir \
#              -DLLVM_ENABLE_ASSERTIONS=ON \
#              -DLLVM_INSTALL_UTILS=ON \
#              -DMLIR_ENABLE_BINDINGS_PYTHON=ON
#        ninja -C build install
# ---------------------------------------------------------------------------

# Activate the project virtualenv if not already in one.
# Override by exporting NKIPY_VENV before sourcing this script.
: "${NKIPY_VENV:=${HOME}/nkipy-opensource-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${NKIPY_VENV}/bin/activate" ]; then
    source "${NKIPY_VENV}/bin/activate"
fi

# ---------------------------------------------------------------------------
# Install Python dependencies (first-time setup)
# ---------------------------------------------------------------------------
# Run once after creating the venv, or whenever dependencies change.
# Subsequent sources of this script skip installation (pip is fast to
# check already-installed packages, but we avoid the overhead entirely).
_nki_marker="${NKIPY_VENV}/.nki_deps_installed"
if [ -n "${VIRTUAL_ENV:-}" ] && [ ! -f "${_nki_marker}" ]; then
    echo "Installing Python dependencies (first time)..."
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com 2>/dev/null
    pip install -q numpy pytest pytest-timeout PyYAML 'black==25.9.0' ml_dtypes \
        neuronx-cc==2.* nki && \
    touch "${_nki_marker}"
fi

# Where the LLVM SDK lives. Override by exporting LLVM_INSTALL_PREFIX
# before sourcing this script.
: "${LLVM_INSTALL_PREFIX:=/opt/llvm-mlir}"

# Optional: URL to a pre-built LLVM tarball. If set and LLVM_INSTALL_PREFIX
# doesn't exist yet, fetch + extract on first use. Unset by default.
: "${LLVM_TARBALL_URL:=}"

# Compiler toolchain used to build the C++ passes. Clang 14+ works; we
# pin clang-22 on the reference dev image.
: "${CC:=clang-22}"
: "${CXX:=clang++-22}"

export LLVM_INSTALL_PREFIX LLVM_TARBALL_URL CC CXX

echo "Setting up NKIPyKernelGen build environment..."

# ---------------------------------------------------------------------------
# Fetch LLVM tarball on first use (opt-in via LLVM_TARBALL_URL).
# ---------------------------------------------------------------------------
if [ ! -d "${LLVM_INSTALL_PREFIX}" ] && [ -n "${LLVM_TARBALL_URL}" ]; then
    echo "LLVM not found at ${LLVM_INSTALL_PREFIX}; downloading from"
    echo "  ${LLVM_TARBALL_URL}"
    parent="$(dirname "${LLVM_INSTALL_PREFIX}")"
    mkdir -p "${parent}"
    tmp="$(mktemp -t llvm-mlir.XXXXXX.tar.gz)"
    if curl -fL --progress-bar -o "${tmp}" "${LLVM_TARBALL_URL}"; then
        tar -xzf "${tmp}" -C "${parent}"
        rm -f "${tmp}"
    else
        echo "ERROR: failed to download LLVM tarball" >&2
        rm -f "${tmp}"
    fi
fi

# ---------------------------------------------------------------------------
# Export paths that the CMake build and the Python phase consume.
# ---------------------------------------------------------------------------
export LLVM_INST="${LLVM_INSTALL_PREFIX}"
export LLVM_DIR="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm"
export MLIR_DIR="${LLVM_INSTALL_PREFIX}/lib/cmake/mlir"
export PATH="${LLVM_INSTALL_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${LLVM_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Put nkipy-opt (built under build/bin) on PATH once built.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${REPO_ROOT}/build/bin:${PATH}"

# Upstream MLIR Python bindings (mlir.ir, mlir.dialects.*). The Python
# linalg→NISA phase parses IR in an upstream context before handing it
# to the NKI context, so the bindings need to match the C++ MLIR build.
LLVM_PY="${LLVM_INSTALL_PREFIX}/python_packages/mlir_core"
if [ -d "${LLVM_PY}" ]; then
    case ":${PYTHONPATH:-}:" in
        *":${LLVM_PY}:"*) ;;
        *) export PYTHONPATH="${LLVM_PY}:${PYTHONPATH:-}" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Sanity checks.
# ---------------------------------------------------------------------------
if [ -d "${LLVM_INSTALL_PREFIX}" ]; then
    rev="$(cat "${LLVM_INSTALL_PREFIX}/REVISION" 2>/dev/null || echo unknown)"
    echo "  LLVM install : ${LLVM_INSTALL_PREFIX} (rev ${rev})"
else
    echo "  WARNING: ${LLVM_INSTALL_PREFIX} not found. nkipy-opt will not build." >&2
    echo "           See comments at the top of this script for install options." >&2
fi

if [ -d "${LLVM_PY}" ]; then
    echo "  Python bindings: ${LLVM_PY}"

    # Check that the active Python version has a matching MLIR native lib.
    # The LLVM install ships .so files for specific CPython versions (e.g.
    # _mlir.cpython-311-*.so).  If the venv uses a different version,
    # `from mlir import ir` will fail at runtime with a confusing ImportError.
    _py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null)"
    if [ -n "${_py_ver}" ]; then
        _mlir_so="${LLVM_PY}/mlir/_mlir_libs/_mlir.cpython-${_py_ver}-"*".so"
        if ! ls ${_mlir_so} >/dev/null 2>&1; then
            _avail="$(ls "${LLVM_PY}"/mlir/_mlir_libs/_mlir.cpython-*.so 2>/dev/null \
                      | sed 's/.*cpython-\([0-9]*\).*/\1/' | tr '\n' ' ')"
            echo "  WARNING: Active Python is ${_py_ver} but MLIR native libs exist for: ${_avail}" >&2
            echo "           Recreate your venv with a matching Python version, e.g.:" >&2
            echo "             python3.${_avail%% *: -1} -m venv ~/nkipy-opensource-venv" >&2
        fi
    fi
else
    echo "  WARNING: ${LLVM_PY} not found. The Python linalg→NISA phase will fail." >&2
fi

echo "  CC / CXX      : ${CC} / ${CXX}"
echo "Done. Build with 'pip install -e .' from the repo root."
