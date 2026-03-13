"""Async/await interface for Spike runtime operations."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Optional, Sequence
from ._spike import (
    NonBlockExecResult,
    NonBlockTensorReadResult,
    NonBlockTensorWriteResult,
    NrtModel,
    NrtTensor,
    NrtTensorSet,
    Spike,
)

import asyncio
import numpy as np


class SpikeAsyncSelector:
    """Event selector for async operations."""

    def __init__(self, spike: Spike) -> None:
        self._spike: Spike = spike

    def select(self, timeout: float | None) -> list[NonBlockResult]:
        """Poll for completed operations."""
        res = self._spike.try_poll()
        if res is None:
            return []
        else:
            return [res]


class SpikeAsyncFuture(asyncio.Future[Any]):
    """Future that can be waited on synchronously."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(SpikeAsyncFuture, self).__init__(*args, **kwargs)

    def wait(self) -> Any:
        """Wait for the future to complete (blocking)."""
        return self._loop.run_until_complete(self)


class SpikeAsyncTask(asyncio.Task[Any]):
    """Task that can be waited on synchronously."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(SpikeAsyncTask, self).__init__(*args, **kwargs)

    def wait(self) -> Any:
        """Wait for the task to complete (blocking)."""
        return self._loop.run_until_complete(self)


class SpikeAsyncEventLoop(asyncio.BaseEventLoop):
    """Custom event loop for Spike async operations."""

    def __init__(self, selector: SpikeAsyncSelector) -> None:
        super().__init__()
        self._selector: SpikeAsyncSelector = selector

        def task_factory(
            loop: asyncio.AbstractEventLoop,
            coro: Coroutine[Any, Any, Any],
            **kwargs: Any
        ) -> SpikeAsyncTask:
            return SpikeAsyncTask(coro, loop=loop, **kwargs)

        self.set_task_factory(task_factory)

        self._tensor_ops: dict[int, SpikeAsyncFuture] = {}
        self._exec_ops: dict[int, SpikeAsyncFuture] = {}

    def create_future(self) -> SpikeAsyncFuture:
        """Create a future for this event loop."""
        return SpikeAsyncFuture(loop=self)

    def register_tensor_op(self, req_id: int, fut: SpikeAsyncFuture) -> None:
        """Register a tensor operation future."""
        assert req_id not in self._tensor_ops
        self._tensor_ops[req_id] = fut

    def register_exec_op(self, req_id: int, fut: SpikeAsyncFuture) -> None:
        """Register an execution operation future."""
        assert req_id not in self._exec_ops
        self._exec_ops[req_id] = fut

    def _process_events(self, event_list: list[NonBlockResult]) -> None:
        """Process completed events."""
        for event in event_list:
            match event:
                case NonBlockTensorWriteResult(id=id, err=err):
                    fut = self._tensor_ops[id]
                    if err is None:
                        fut.set_result(None)
                    else:
                        fut.set_exception(err)

                    del self._tensor_ops[id]

                case NonBlockTensorReadResult(id=id, data=data, err=err):
                    fut = self._tensor_ops[id]
                    if err is None:
                        fut.set_result(data)
                    else:
                        fut.set_exception(err)

                    del self._tensor_ops[id]

                case NonBlockExecResult(id=id, err=err):
                    fut = self._exec_ops[id]
                    if err is None:
                        fut.set_result(None)
                    else:
                        fut.set_exception(err)

                    del self._exec_ops[id]


class SpikeAsync:
    """Async interface for Spike runtime."""

    def __init__(self, verbose_level: int = 0) -> None:
        self.spike: Spike = Spike(verbose_level=verbose_level)
        self.spike.init_nonblock()

        self._selector: SpikeAsyncSelector = SpikeAsyncSelector(self.spike)
        self._loop: SpikeAsyncEventLoop = SpikeAsyncEventLoop(self._selector)

        self._default_stream: SpikeStream | None = None

    def _wrapper(
        self,
        closure: Callable[[], SpikeAsyncFuture],
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None,
        stream: SpikeStream | None
    ) -> SpikeAsyncFuture | SpikeAsyncTask:
        """Wrap an operation with dependency and stream management."""
        # Try to use default stream if no stream is explicitly specified
        if stream is None:
            stream = self._default_stream

        # If no explicit dependency, see if there is any stream induced dependency
        if deps is None or len(deps) == 0:
            if stream is None:
                # No stream, fast path
                return closure()
            elif stream._last_fut is None:
                # There is stream but no stream induced dependency
                # Fast path, but also put the current op to the stream
                fut = closure()
                stream._last_fut = fut
                return fut

        # If there is a stream induced dependency, add it to explicit dependency
        if stream is not None and stream._last_fut is not None:
            if deps is None:
                deps = []
            else:
                deps = list(deps)  # Make mutable copy

            deps.append(stream._last_fut)

        # Heavy path for having any dependency
        async def internal_async():
            for dep in deps:
                await dep
            return await closure()

        fut = self._loop.create_task(internal_async())

        if stream is not None:
            stream._last_fut = fut

        return fut

    def create_stream(self) -> SpikeStream:
        """Create a new stream for operation sequencing."""
        return SpikeStream(self)

    def create_tensor_set(
        self,
        tensor_map: dict[str, NrtTensor]
    ) -> NrtTensorSet:
        """Create a tensor set from a dictionary of tensors.

        Convenience method for spike.create_tensor_set().
        """
        return self.spike.create_tensor_set(tensor_map)

    def load_model(
        self,
        neff_file: str,
        core_id: int = 0,
        cc_enabled: bool = False,
        rank_id: int = 0,
        world_size: int = 1
    ) -> NrtModel:
        """Load a model from a NEFF file.

        Convenience method for spike.load_model().
        """
        return self.spike.load_model(neff_file, core_id, cc_enabled, rank_id, world_size)

    def allocate_tensor(
        self,
        size: int,
        core_id: int = 0,
        name: str | None = None
    ) -> NrtTensor:
        """Allocate a tensor on a NeuronCore.

        Convenience method for spike.allocate_tensor().
        """
        return self.spike.allocate_tensor(size, core_id, name)

    def tensor_write(
        self,
        tensor: NrtTensor,
        data: Any,
        offset: int = 0,
        size: int = 0,
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncFuture | SpikeAsyncTask:
        """Write to a tensor asynchronously."""

        def internal() -> SpikeAsyncFuture:
            fut = self._loop.create_future()
            if size > 0:
                req_id = self.spike.tensor_write_nonblock(tensor, data, size, offset)
            else:
                req_id = self.spike.tensor_write_nonblock(tensor, data, offset)
            self._loop.register_tensor_op(req_id, fut)
            return fut

        return self._wrapper(internal, deps, stream)

    def tensor_write_batched_start(
        self,
        batch_id: int,
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncFuture | SpikeAsyncTask:
        """Start a batched tensor write asynchronously."""

        def internal() -> SpikeAsyncFuture:
            fut = self._loop.create_future()
            req_id = self.spike.tensor_write_nonblock_batched_start(batch_id)
            self._loop.register_tensor_op(req_id, fut)
            return fut

        return self._wrapper(internal, deps, stream)

    def tensor_read_batched_start(
        self,
        batch_id: int,
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncFuture | SpikeAsyncTask:
        """Start a batched tensor read asynchronously."""

        def internal() -> SpikeAsyncFuture:
            fut = self._loop.create_future()
            req_id = self.spike.tensor_read_nonblock_batched_start(batch_id)
            self._loop.register_tensor_op(req_id, fut)
            return fut

        return self._wrapper(internal, deps, stream)

    def tensor_read(
        self,
        tensor: NrtTensor,
        dest: Any = None,
        offset: int = 0,
        size: int = 0,
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncFuture | SpikeAsyncTask:
        """Read from a tensor asynchronously."""

        def internal() -> SpikeAsyncFuture:
            fut = self._loop.create_future()
            if dest is None:
                req_id = self.spike.tensor_read_nonblock(tensor, offset, size)
            else:
                req_id = self.spike.tensor_read_nonblock(tensor, dest, offset, size)
            self._loop.register_tensor_op(req_id, fut)
            return fut

        return self._wrapper(internal, deps, stream)

    def execute(
        self,
        model: NrtModel,
        inputs: NrtTensorSet,
        outputs: NrtTensorSet,
        ntff_name: str | None = None,
        save_trace: bool = False,
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncFuture | SpikeAsyncTask:
        """Execute a model asynchronously."""

        def internal() -> SpikeAsyncFuture:
            fut = self._loop.create_future()
            req_id = self.spike.execute_nonblock(
                model, inputs, outputs, ntff_name, save_trace
            )
            self._loop.register_exec_op(req_id, fut)
            return fut

        return self._wrapper(internal, deps, stream)

    def custom_op(
        self,
        coro: Coroutine[Any, Any, Any],
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncTask:
        """Submit a custom coroutine as an operation."""
        # Try to use default stream if no stream is explicitly specified
        if stream is None:
            stream = self._default_stream

        # If no explicit dependency, see if there is any stream induced dependency
        if deps is None or len(deps) == 0:
            if stream is None:
                # No stream, fast path
                return self._loop.create_task(coro)
            elif stream._last_fut is None:
                # There is stream but no stream induced dependency
                # Fast path, but also put the current op to the stream
                fut = self._loop.create_task(coro)
                stream._last_fut = fut
                return fut

        # If there is a stream induced dependency, add it to explicit dependency
        if stream is not None and stream._last_fut is not None:
            if deps is None:
                deps = []
            else:
                deps = list(deps)  # Make mutable copy

            deps.append(stream._last_fut)

        # Heavy path for having any dependency
        async def internal_async() -> Any:
            for dep in deps:
                await dep
            return await coro

        fut = self._loop.create_task(internal_async())

        if stream is not None:
            stream._last_fut = fut

        return fut

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        deps: Sequence[SpikeAsyncFuture | SpikeAsyncTask | Coroutine[Any, Any, Any]] | None = None,
        stream: SpikeStream | None = None
    ) -> SpikeAsyncTask:
        """Submit a coroutine to the event loop."""
        return self.custom_op(coro, deps, stream)

    @staticmethod
    async def all(futs: Sequence[SpikeAsyncFuture | SpikeAsyncTask]) -> list[Any]:
        """Wait for all futures to complete."""
        res: list[Any] = []
        for fut in futs:
            res.append(await fut)
        return res


class SpikeStream:
    """Stream for sequencing operations."""

    def __init__(self, spike_async: SpikeAsync) -> None:
        self._last_fut: SpikeAsyncFuture | SpikeAsyncTask | None = None
        self._spike_async: SpikeAsync = spike_async
        self._prev_stream: SpikeStream | None = None

    def __enter__(self) -> SpikeStream:
        """Enter context manager, making this the default stream."""
        self._prev_stream = self._spike_async._default_stream
        self._spike_async._default_stream = self
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        """Exit context manager, restoring previous default stream."""
        self._spike_async._default_stream = self._prev_stream
        self._prev_stream = None

    def wait(self) -> Any:
        """Wait for all operations in this stream to complete."""
        if self._last_fut is not None:
            return self._last_fut.wait()

    def record_event(self) -> SpikeAsyncFuture | SpikeAsyncTask | None:
        """Record an event at the current point in the stream."""
        return self._last_fut

    def wait_event(self, event: SpikeAsyncFuture | SpikeAsyncTask) -> None:
        """Wait for an event from another stream."""

        async def internal(fut: SpikeAsyncFuture | SpikeAsyncTask | None) -> None:
            if fut is not None:
                await fut
            await event

        self._last_fut = self._spike_async.custom_op(internal(self._last_fut))
