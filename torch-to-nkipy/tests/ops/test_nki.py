# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import pytest
import torch
from base import NKIPyTestBase

from torch_to_nkipy import NKIOpRegistry

# Simple matrix add kernel for testing


# Version 1: no grid, no kwargs
@NKIOpRegistry.register("mylib::add_custom_op_no_grid_no_kwarg")
def nki_tensor_add_kernel_no_grid_no_kwarg(a_input, b_input):
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    ix = nl.arange(128)[:, None]
    iy = nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    nl.store(c_output[ix, iy], value=c_tile)
    return c_output


@torch.library.custom_op("mylib::add_custom_op_no_grid_no_kwarg", mutates_args=())
def nki_add_no_grid_no_kwarg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nki_tensor_add_kernel_no_grid_no_kwarg(a, b)


@nki_add_no_grid_no_kwarg.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


# Version 2: with grid, no kwargs
@NKIOpRegistry.register("mylib::add_custom_op_with_grid_no_kwarg")
def nki_tensor_add_kernel_with_grid_no_kwarg(a_input, b_input):
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512
    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    nl.store(c_output[ix, iy], value=c_tile)
    return c_output


@torch.library.custom_op("mylib::add_custom_op_with_grid_no_kwarg", mutates_args=())
def nki_add_with_grid_no_kwarg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    grid = (a.shape[0] // 128, a.shape[1] // 512)
    return nki_tensor_add_kernel_with_grid_no_kwarg[grid](a, b)


@nki_add_with_grid_no_kwarg.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


# Version 3: with grid, with kwargs
@NKIOpRegistry.register("mylib::add_custom_op_with_grid_with_kwarg")
def nki_tensor_add_kernel_with_grid_with_kwarg(
    a_input, b_input, bias=0.0, add_bias=False
):
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512
    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    if add_bias:
        c_tile = c_tile + bias
    nl.store(c_output[ix, iy], value=c_tile)
    return c_output


@torch.library.custom_op("mylib::add_custom_op_with_grid_with_kwarg", mutates_args=())
def nki_add_with_grid_with_kwarg(
    a: torch.Tensor, b: torch.Tensor, bias: float, add_bias: bool
) -> torch.Tensor:
    grid = (a.shape[0] // 128, a.shape[1] // 512)
    return nki_tensor_add_kernel_with_grid_with_kwarg[grid](a, b, bias, add_bias)


@nki_add_with_grid_with_kwarg.register_fake
def _(a: torch.Tensor, b: torch.Tensor, bias: float, add_bias: bool) -> torch.Tensor:
    return torch.empty_like(a)


# Version 4: with IO aliasing
@NKIOpRegistry.register("mylib::add_custom_op_io_alias", alias_map={0: 0})
def nki_tensor_add_kernel_io_alias(a_input: nt.mutable_tensor, b_input):
    ix = nl.arange(128)[:, None]
    iy = nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    nl.store(a_input[ix, iy], value=c_tile)
    return a_input


@torch.library.custom_op("mylib::add_custom_op_io_alias", mutates_args=())
def nki_add_io_alias(a_input: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
    return nki_tensor_add_kernel_io_alias(a_input, b_input)


@nki_add_io_alias.register_fake
def _(a_input: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a_input)


# Version 5: more IO aliasing
@NKIOpRegistry.register("mylib::more_io_alias", alias_map={0: 0, 1: 1})
def nki_kernel_more_io_alias(a_input: nt.mutable_tensor, b_input: nt.mutable_tensor):
    ix = nl.arange(128)[:, None]
    iy = nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    c_res = nl.ndarray(a_input.shape, a_input.dtype, buffer=nl.hbm)
    nl.store(a_input[ix, iy], value=c_tile)
    nl.store(b_input[ix, iy], value=c_tile)
    nl.store(c_res[ix, iy], value=c_tile)
    return a_input, b_input, c_res


@torch.library.custom_op("mylib::more_io_alias", mutates_args=())
def nki_more_alias(
    a_input: torch.Tensor, b_input: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return nki_kernel_more_io_alias(a_input, b_input)


@nki_more_alias.register_fake
def _(
    a_input: torch.Tensor, b_input: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(a_input),
        torch.empty_like(a_input),
        torch.empty_like(a_input),
    )


class TestNki(NKIPyTestBase):
    @pytest.mark.parametrize("shape", [(128, 512)])
    def test_nki_no_grid_no_kwarg(self, shape):
        def test_func(a, b, c):
            add_res = nki_add_no_grid_no_kwarg(a, b)
            res = add_res + c
            return res

        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)
        arg_2 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b, c):
            return a + b + c

        self.run_test_on_device(
            test_func, (arg_0, arg_1, arg_2), cpu_ref_func=cpu_ref_func
        )

    @pytest.mark.parametrize(
        "shape", [(128, 512), (128, 1024), (1024, 512), (256, 2048)]
    )
    def test_nki_with_grid_no_kwarg(self, shape):
        def test_func(a, b, c):
            add_res = nki_add_with_grid_no_kwarg(a, b)
            res = add_res + c
            return res

        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)
        arg_2 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b, c):
            return a + b + c

        self.run_test_on_device(
            test_func, (arg_0, arg_1, arg_2), cpu_ref_func=cpu_ref_func
        )

    @pytest.mark.parametrize(
        "shape", [(128, 512), (128, 1024), (1024, 512), (256, 2048)]
    )
    @pytest.mark.parametrize("bias", [0.0, 5.0])
    @pytest.mark.parametrize("add_bias", [True, False])
    def test_nki_with_grid_with_kwarg(self, shape, bias, add_bias):
        def test_func(a, b, c):
            add_res = nki_add_with_grid_with_kwarg(a, b, bias, add_bias)
            res = add_res + c
            return res

        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)
        arg_2 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b, c):
            res = a + b + c
            if add_bias:
                res += bias
            return res

        self.run_test_on_device(
            test_func, (arg_0, arg_1, arg_2), cpu_ref_func=cpu_ref_func
        )

    @pytest.mark.parametrize("shape", [(128, 512)])
    def test_nki_io_alias(self, shape):
        def test_func(a, b):
            # a is updated in-place, after running the function a = a + b
            a = nki_add_io_alias(a, b)
            c = a + b
            return c

        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b):
            a[:] = a + b  # modify a in-place to match IO aliasing behavior
            res = a + b
            return res

        self.run_test_on_device(test_func, (arg_0, arg_1), cpu_ref_func=cpu_ref_func)

    @pytest.mark.parametrize("shape", [(128, 512)])
    def test_nki_io_alias2(self, shape):
        def test_func(a, b):
            # a and b is updated in-place
            # after running the function (a, b) = (a + b, a + b)
            a, b, c = nki_more_alias(a, b)
            return c

        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b):
            res = a + b
            a[:] = res  # modify a in-place to match IO aliasing behavior
            b[:] = res  # modify b in-place to match IO aliasing behavior
            return res

        self.run_test_on_device(test_func, (arg_0, arg_1), cpu_ref_func=cpu_ref_func)

    @pytest.mark.xfail(
        reason="NKI kernel only (no additional ops) triggers compiler bug."
    )
    @pytest.mark.parametrize("shape", [(128, 512)])
    def test_nki_io_alias_direct_return(self, shape):
        def test_func(a, b):
            a = nki_add_io_alias(a, b)
            return a

        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b):
            a[:] = a + b  # modify a in-place to match IO aliasing behavior
            return a

        self.run_test_on_device(test_func, (arg_0, arg_1), cpu_ref_func=cpu_ref_func)

    def test_nki_kernel_caching(self):
        def test_func(a, b, c):
            add_res = nki_add_with_grid_with_kwarg(a, b, 1.0, False)
            add_res_2 = nki_add_with_grid_with_kwarg(add_res, b, 5.0, True)
            add_res_3 = nki_add_with_grid_with_kwarg(a, add_res_2, 13.0, True)
            add_res_4 = nki_add_with_grid_with_kwarg(add_res_3, b, 1.0, False)
            res = add_res_4 + c
            return res

        shape = (128, 512)
        arg_0 = torch.randn(size=shape, dtype=torch.float32)
        arg_1 = torch.randn(size=shape, dtype=torch.float32)
        arg_2 = torch.randn(size=shape, dtype=torch.float32)

        def cpu_ref_func(a, b, c):
            add_res = a + b
            add_res_2 = add_res + b + 5.0
            add_res_3 = a + add_res_2 + 13.0
            add_res_4 = add_res_3 + b
            res = add_res_4 + c
            return res

        self.run_test_on_device(
            test_func, (arg_0, arg_1, arg_2), cpu_ref_func=cpu_ref_func
        )
        assert len(NKIOpRegistry._processed_nki_kernel_hash) == 3
