# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Project root and source dirs
ROOT_DIR = Path(__file__).resolve().parent
CSRC_DIR = ROOT_DIR / "csrc"

# Collect all .cpp sources recursively (relative paths)
sources = [str(s.relative_to(ROOT_DIR)) for s in CSRC_DIR.rglob("*.cpp")]

# Compiler flags
CXX_FLAGS = {
    "cxx": [
        "-O3",  # optimize
        "-g",  # keep debug symbols
        "-Wall",  # enable warnings
        "-Werror",  # treat warnings as errors
    ]
}

# Neuron Runtime paths (allow override via env)
NRT_INCLUDE = os.environ.get("NRT_INCLUDE", "/opt/aws/neuron/include")
NRT_LIB = os.environ.get("NRT_LIB", "/opt/aws/neuron/lib")

# Torch library path
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")


# Custom BuildExtension to prefer Ninja if available
class BuildExtWithNinja(BuildExtension):
    def __init__(self, *args, **kwargs):
        # Enable Ninja if installed
        kwargs["use_ninja"] = True
        super().__init__(*args, **kwargs)


ext_modules = [
    CppExtension(
        name="spike_torch._C",
        sources=sources,
        include_dirs=[
            str(ROOT_DIR),
            str(CSRC_DIR),
            str(CSRC_DIR / "include"),
            NRT_INCLUDE,
        ],
        library_dirs=[
            NRT_LIB,
            torch_lib_path,
        ],
        libraries=["nrt"],
        runtime_library_dirs=[
            NRT_LIB,
            torch_lib_path,
        ],
        extra_compile_args=CXX_FLAGS,
        extra_link_args=[
            f"-Wl,-rpath,{NRT_LIB}",
            f"-Wl,-rpath,{torch_lib_path}",
        ],
    )
]

setup(
    name="spike_torch",
    packages=find_packages(
        where="src",
        include=["spike_torch", "spike_torch.*"],
    ),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithNinja},
    zip_safe=False,  # binary extensions not zip safe
)
