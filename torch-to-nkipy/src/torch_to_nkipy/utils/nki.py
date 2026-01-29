# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for registering custom NKI operators"""

import logging
from typing import Callable, Dict, List, Set

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import unset_fake_temporarily


class NKIKernelWrapper:
    """Wrapper around NKI kernel to preserve launch grid and compiler args."""

    def __init__(self, func, compiler_args="", alias_map={}):
        self.func = func
        self.grid = []
        self.compiler_args = compiler_args
        self.alias_map = alias_map

    def __getitem__(self, grid):
        if not isinstance(grid, (tuple, list)):
            grid = [grid]
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        return None


class NKIOpRegistry:
    """
    Registry for custom NKI kernels.

    This class implements a registry pattern to map PyTorch custom operators
    to user-specific NKI kernel implementations.
    """

    # Maps custom op names to kernels
    _nki_name_map: Dict[str, Callable] = {}
    _processed_nki_kernel_hash: Set[str] = set()

    @classmethod
    def canonicalize_custom_op_name(cls, custom_op_name: str) -> str:
        legit_custom_op_name = f"torch.ops.{custom_op_name.replace('::', '.')}.default"
        return legit_custom_op_name

    @classmethod
    def register(
        cls, custom_op_name: str, compiler_args: str = "", alias_map={}
    ) -> NKIKernelWrapper:
        legit_custom_op_name = cls.canonicalize_custom_op_name(custom_op_name)
        assert legit_custom_op_name not in cls._nki_name_map, (
            f"{custom_op_name} is already registered in NKI op registry!"
        )

        if len(alias_map) > 0:
            logging.warning(
                "CUSTOM NKI OP WITH IO ALIASING IS FRAGILE, USE AT YOUR OWN RISK. "
            )
            logging.warning(
                "CUSTOM NKI OP WITH IO ALIASING IS FRAGILE, USE AT YOUR OWN RISK. "
            )
            logging.warning(
                "CUSTOM NKI OP WITH IO ALIASING IS FRAGILE, USE AT YOUR OWN RISK. "
            )

        def actual_decorator(nki_kernel_func: Callable):
            wrapped_kernel = NKIKernelWrapper(nki_kernel_func, compiler_args, alias_map)
            cls._nki_name_map[legit_custom_op_name] = wrapped_kernel
            return wrapped_kernel

        return actual_decorator

    @classmethod
    def is_registered(cls, custom_op_name: str, canonicalize: bool = False) -> bool:
        legit_custom_op_name = custom_op_name
        if canonicalize:
            legit_custom_op_name = cls.canonicalize_custom_op_name(custom_op_name)
        return legit_custom_op_name in cls._nki_name_map

    @classmethod
    def get_nki_kernel(
        cls, custom_op_name: str, canonicalize: bool = False
    ) -> Callable:
        legit_custom_op_name = custom_op_name
        if canonicalize:
            legit_custom_op_name = cls.canonicalize_custom_op_name(custom_op_name)
        assert cls.is_registered(legit_custom_op_name, canonicalize=False), (
            f"Custom op {custom_op_name} is not registered in NKi op registry!"
        )
        return cls._nki_name_map[legit_custom_op_name]

    @classmethod
    def add_processed_kernel_hash(cls, hash_str: str) -> None:
        cls._processed_nki_kernel_hash.add(hash_str)

    @classmethod
    def is_processed(cls, hash_str: str) -> bool:
        return hash_str in cls._processed_nki_kernel_hash

    @classmethod
    def reset_procesed_kernels(cls) -> None:
        cls._processed_nki_kernel_hash.clear()


# Get a NKI kernel's launch grid during aten op lowering
def populate_grid(op: Callable, args: List) -> None:
    with unset_fake_temporarily():
        with torch.no_grad():
            try:
                real_args = [
                    torch.empty(size=arg.meta["val"].shape, dtype=arg.meta["val"].dtype)
                    if isinstance(arg, fx.Node)
                    else arg
                    for arg in args
                ]
                op(*real_args)
            except Exception:
                # We don't care what is generated here
                pass


# Generate a hash str for a NKI kernel given its name, arguments, and launch grid
def get_nki_kernel_hash(
    name: str, args_str: str, grid_str: str, compiler_args: str = ""
) -> str:
    return str(hash(name + args_str + grid_str + compiler_args)).replace("-", "_")
