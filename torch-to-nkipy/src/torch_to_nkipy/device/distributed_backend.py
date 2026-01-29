# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllToAllOptions,
    GatherOptions,
    ProcessGroup,
    ReduceOptions,
    ScatterOptions,
)


def neuron_barrier(device_id, global_device_id, global_device_count):
    """No-op barrier - Neuron operations are synchronous."""
    pass


def _create_neuron_process_group(prefix_store, rank, size, timeout):
    return ProcessGroupNeuron(prefix_store, rank, size, timeout)


def _register_neuron_backend():
    dist.Backend.register_backend(
        "nkipy", _create_neuron_process_group, devices="nkipy"
    )


def rendezvous_handler():
    pass


_register_neuron_backend()

dist.register_rendezvous_handler("nkipy", rendezvous_handler)


def _ret_work(ret):
    # TODO: Need to implement better version of Work. This basic version works
    # when there is single stream. When we have multi-stream execution,
    # this needs to be implemented.
    fut = torch.futures.Future()
    fut.set_result(ret)
    return torch._C._distributed_c10d._create_work_from_future(fut)


class ProcessGroupNeuron(ProcessGroup):
    """ProcessGroup for Neuron device. See ProcessGroup for doc.

    Here we are implementing only a Python subclass. For implementing a
    C++/Python extension, see
    https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html.
    """

    def __init__(self, prefix_store, rank, size, timeout):
        super().__init__(rank, size)

    def getBackendName(self):  # noqa N802
        return "nkipy"

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

    @property
    def group_name(self):
        return self._group_name

    def allreduce(self, tensors, all_reduce_options):
        raise NotImplementedError(__class__.allreduce)

    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: AllgatherOptions,
    ):
        raise NotImplementedError(__class__._allgather_base)

    def allgather(self, output_tensors_list, input_tensors, opts=None):
        raise NotImplementedError(__class__.allgather)

    def allgather_coalesced(self, output_tensors_list, input_tensors, opts=None):
        raise NotImplementedError(__class__.allgather_coalesced)

    def broadcast(self, tensors, opts):
        raise NotImplementedError(__class__.broadcast)

    def reduce_scatter(self, output_tensors, input_tensors_list, opts):
        raise NotImplementedError(__class__.reduce_scatter)

    def reduce_scatter_coalesced(self, output_tensors, input_tensors_list, opts):
        raise NotImplementedError(__class__.reduce_scatter_coalesced)

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts):
        raise NotImplementedError(__class__._reduce_scatter_base)

    def barrier(self, opts):
        neuron_barrier(0, self.rank(), self.size())
        return _ret_work(None)

    def reduce(self, tensors: list[torch.Tensor], opts: ReduceOptions):
        raise NotImplementedError(__class__.reduce)

    def allreduce_coalesced(self, *args):
        raise NotImplementedError(__class__.allreduce_coalesced)

    def alltoall(
        self,
        output_tensor_list: list[torch.Tensor],
        input_tensor_list: list[torch.Tensor],
        opts: AllToAllOptions,
    ):
        raise NotImplementedError(__class__.alltoall)

    def alltoall_base(self, output, input, output_split_sizes, input_split_sizes, opts):
        raise NotImplementedError(__class__.alltoall_base)

    def gather(
        self,
        output_tensors_list: list[list[torch.Tensor]],
        input_tensor_list: list[torch.Tensor],
        opts: GatherOptions,
    ):
        raise NotImplementedError(__class__.gather)

    def scatter(
        self,
        output_tensor_list: list[torch.Tensor],
        input_tensors_list: list[list[torch.Tensor]],
        opts: ScatterOptions,
    ):
        raise NotImplementedError(__class__.scatter)

    def recv_anysource(self, *args):
        raise NotImplementedError

    def monitored_barrier(self, *args):
        raise NotImplementedError

    def Options(self, *args):  # noqa N802
        raise NotImplementedError
