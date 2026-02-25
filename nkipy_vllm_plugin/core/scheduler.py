# SPDX-License-Identifier: Apache-2.0
import logging
from collections import deque
from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    SchedulerOutput = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support continuous 
    batching respecting Neuron constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)

        # Requests are temporarily moved to this queue so that the base
        # scheduler does not see them. This lets us ensure that the set of
        # requests scheduled have at least one common warmup shape.
        self.holdback_queue: deque[Request] = deque()


class ContinuousBatchingNeuronScheduler(NeuronScheduler):
    """ Support of continuous batching """

    # inherited from V1 base scheduler but mypy needs to know the type
    running: list[Request]

    def __init__(self, *args, **kwargs) -> None:
        # Initialize NeuronScheduler
        super().__init__(*args, **kwargs)

    def schedule(self) -> "SchedulerOutput":
        """This override adds constraints and then delegates most of the work
        to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after.
        """

        # First purge the full waiting queue into our holdback queue, preserving
        # priority
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # Check if new requests can be scheduled.
        while self.holdback_queue:
            if self.can_schedule(self.holdback_queue[0]):
                # Add request to the waiting queue
                self.waiting.append(self.holdback_queue.popleft())
            else:
                # Otherwise, we simply stop here so that the scheduler
                # can work with the batch we have
                break

        # Schedule Prefill and Decode separately
        if len(self.waiting) > 0:
            # For prefill, hide current decodes from the scheduler
            running_holdback = self.running
            self.running = []
            logger.debug(
                f"Scheduling a prefill step of {len(self.waiting)} requests, holding back {len(self.holdback_queue)} "
                "requests")
        else:
            running_holdback = []
            logger.debug(
                f"Scheduling a decode step of {len(self.running)} requests")

        # delegate to super of NeuronScheduler: base V1 Scheduler
        outputs = super(NeuronScheduler, self).schedule()

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())

        return outputs

    def can_schedule(self, request) -> bool:
        max_prompt_batch_size = 1
        _max_context_len = self.scheduler_config.max_model_len

        # running and waiting queues are both empty -> start new batch
        start_new_batch = len(self.running) + len(self.waiting) == 0
        # check that there is space in the current decode batch
        cond1 = len(self.running) + len(
            self.waiting) < self.max_num_running_reqs
        # check that there is space in the prefill batch
        cond2 = len(self.waiting) < max_prompt_batch_size
        # TODO: add context length checks
        return start_new_batch or (cond1 and cond2)
