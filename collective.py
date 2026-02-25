from nkipy.core.trace import NKIPyKernel
from nkipy.runtime.device_tensor import DeviceTensor
import nkipy.distributed.collectives as cc
from nkipy.runtime.device_kernel import DeviceKernel
import numpy as np
from typing import Tuple

trace = NKIPyKernel.trace(backend="hlo")

def to_nested_tuple(nested_list):
    return tuple(
        to_nested_tuple(item) if isinstance(item, list) or isinstance(item, tuple) else item
        for item in nested_list
    )

# TODO: refactor using torch gloo collective instead of compile a neff?

compiled_all_gather_dict = {}
def all_gather(data, all_gather_dim, replica_groups: Tuple[Tuple[int]], is_neuronpy: bool):
    if len(replica_groups[0]) == 1:
        return data
    # convert any list into tuple
    replica_groups = to_nested_tuple(replica_groups)
    if is_neuronpy:
        return cc.all_gather(data, all_gather_dim, replica_groups)
    else:
        # use
        key = (data.shape, data.dtype, all_gather_dim, len(replica_groups), len(replica_groups[0]))
        if key not in compiled_all_gather_dict:
            kernel = DeviceKernel.compile_and_load(
                kernel=trace(all_gather),
                data=data,
                all_gather_dim=all_gather_dim,
                replica_groups=replica_groups,
                is_neuronpy=True,
                name=f"all_gather_{key}",
            )
            compiled_all_gather_dict[key] = kernel
        else:
          kernel = compiled_all_gather_dict[key]
        shape = list(data.shape)
        shape[all_gather_dim] *= len(replica_groups[0])
        res = np.zeros_like(data, shape=shape)
        res = DeviceTensor.from_numpy(res)
        kernel(
          inputs={
              "data": DeviceTensor.from_numpy(data),
          },
          outputs={
              "output0": res,
          }
        )
        return res.numpy()


compiled_reduce_scatter_dict = {}
def reduce_scatter(
    data, reduce_scatter_dim, replica_groups: Tuple[Tuple[int]], is_neuronpy: bool
):
    if len(replica_groups[0]) == 1:
        return data
    # convert any list into tuple
    replica_groups = to_nested_tuple(replica_groups)
    if is_neuronpy:
        return cc.reduce_scatter(data, reduce_scatter_dim, replica_groups)
    else:
        key = (data.shape, data.dtype, reduce_scatter_dim, len(replica_groups), len(replica_groups[0]))
        if key not in compiled_reduce_scatter_dict:
            kernel = DeviceKernel.compile_and_load(
                kernel=trace(reduce_scatter),
                data=data,
                reduce_scatter_dim=reduce_scatter_dim,
                replica_groups=replica_groups,
                is_neuronpy=True,
                name=f"reduce_scatter_{key}",
            )
            compiled_reduce_scatter_dict[key] = kernel
        else:
            kernel = compiled_reduce_scatter_dict[key]
        shape = list(data.shape)
        shape[reduce_scatter_dim] //= len(replica_groups[0])
        res = np.zeros_like(data, shape=shape)
        res = DeviceTensor.from_numpy(res)
        kernel(
            inputs={
                "data": DeviceTensor.from_numpy(data),
            },
            outputs={
                "output0": res,
            },
        )
        return res.numpy()


compiled_all_reduce_dict = {}
def all_reduce(
    data, replica_groups: Tuple[Tuple[int]], is_neuronpy: bool
):
    if len(replica_groups[0]) == 1:
        return data
    # convert any list into tuple
    replica_groups = to_nested_tuple(replica_groups)
    if is_neuronpy:
        return cc.all_reduce(data, replica_groups)
    else:
        key = (data.shape, data.dtype, len(replica_groups), len(replica_groups[0]))
        if key not in compiled_all_reduce_dict:
            kernel = DeviceKernel.compile_and_load(
                kernel=trace(all_reduce),
                data=data,
                replica_groups=replica_groups,
                is_neuronpy=True,
                name=f"all_reduce_{key}",
            )
            compiled_all_reduce_dict[key] = kernel
        else:
            kernel = compiled_all_reduce_dict[key]
        res = np.zeros_like(data)
        res = DeviceTensor.from_numpy(res)
        kernel(
            inputs={
                "data": DeviceTensor.from_numpy(data),
            },
            outputs={
                "output0": res,
            },
        )
        return res.numpy()

compiled_all_to_all_dict = {}
def all_to_all(
    data, split_dimension, concat_dimension, replica_groups: Tuple[Tuple[int]], is_neuronpy: bool
):
    if len(replica_groups[0]) == 1:
        return data
    # convert any list into tuple
    replica_groups = to_nested_tuple(replica_groups)
    if is_neuronpy:
        return cc.all_to_all(
            data=data,
            split_dimension=split_dimension,
            concat_dimension=concat_dimension,
            replica_groups=replica_groups,
        )
    else:
        key = (data.shape, data.dtype, len(replica_groups), len(replica_groups[0]))
        if key not in compiled_all_to_all_dict:
            kernel = DeviceKernel.compile_and_load(
                kernel=trace(all_to_all),
                data=data,
                split_dimension=split_dimension,
                concat_dimension=concat_dimension,
                replica_groups=replica_groups,
                is_neuronpy=True,
                name=f"all_to_all_{key}",
            )
            compiled_all_to_all_dict[key] = kernel
        else:
            kernel = compiled_all_to_all_dict[key]
        res = np.zeros_like(data)
        res = DeviceTensor.from_numpy(res)
        kernel(
            inputs={
                "data": DeviceTensor.from_numpy(data),
            },
            outputs={
                "output0": res,
            },
        )
        return res.numpy()
