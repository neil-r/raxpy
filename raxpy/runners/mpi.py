"""
    Provides Message Passing Interface (MPI) implementaion of
    parallel, distributed experiment execution logic.
"""

from typing import Optional
from mpi4py import MPI
from dataclasses import dataclass, field
import time

from .coordinator import (
    WorkerPool,
    WorkerContext,
    coordinate,
    CompletedTaskEvent,
)
from .task_provider import BatchExperimentTaskProvider
from .. import generate_design, function_spec


EVENT_TYPE_REGISTER = "register"  # sent from worker to coordinator
EVENT_TYPE_TASK_RESPONSE = "response"  # sent from worker to coordinator
EVENT_TYPE_TASK = "task"  # sent from coordinator to worker
EVENT_TYPE_DONE = "done"  # sent from coordinator to worker


@dataclass
class Event:
    from_rank: int
    type: str
    data: object = None


def perform_distributed_experiment(f, n_points, post_process):
    """
    Executes a distributed experiment. Must be called from
    within a MPI Process.  The MPI Process with rank-0 acts
    as the coordinator that designs the experiment and delegates
    function executions to the other MPI Processes. The coordinator
    assigns tasks for each point within the experiment tobe
    executed by f.

    The non-ranked zero MPI Processes act as workers; they
    should call this functions with the same values as the coordinator.
    They first notify the coordinator MPI process of rank-0 that they
    exist and then execute tasks from the coordinator.

    Once every point from the design is executed, post_process
    is called with the first argument being the designed experiment and
    the second arugment with the list of the results for each point in the
    design.
    """
    input_space = function_spec.extract_input_space(f)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tag = 11
    coordinator_rank = 0
    if rank == coordinator_rank:
        worker_pool = MPIWorkerPool(comm=comm, tag=tag)
        doe = generate_design(input_space, n_points=n_points)

        task_provider = BatchExperimentTaskProvider(design=doe)
        coordinate(task_provider, worker_pool)
        worker_pool.shutdown()
        post_process(task_provider.design, task_provider.results)
    else:
        # send registration message
        comm.send(
            Event(from_rank=rank, type=EVENT_TYPE_REGISTER),
            dest=coordinator_rank,
            tag=tag,
        )

        # execute tasks sent to this worker
        event: Event = comm.recv(source=coordinator_rank, tag=tag)

        while event is not None:
            if event.type == EVENT_TYPE_TASK:
                results = f(**event.data["input"])
                print(f"Executed f with {event.data}, got result: {results}")

                comm.send(
                    Event(
                        from_rank=rank,
                        type=EVENT_TYPE_TASK_RESPONSE,
                        data=results,
                    ),
                    dest=coordinator_rank,
                    tag=tag,
                )
            elif event.type == EVENT_TYPE_DONE:
                break
            event = comm.recv(source=coordinator_rank, tag=tag)


def _get_comm():
    comm = MPI.COMM_WORLD
    return comm


@dataclass
class MPIWorkerPool(WorkerPool):
    comm: MPI.Intracomm = field(default_factory=_get_comm)
    tag: int = 11
    _status: Optional[MPI.Status] = None

    def _process(self):
        event: Optional[Event] = None

        if self._status is None:
            self._status = MPI.Status()
            # self.comm.Iprobe(tag=self.tag, status=self._status)

        if self.comm.iprobe(tag=self.tag):
            event = self.comm.recv(tag=self.tag)
            self._status = None
            if event.type == EVENT_TYPE_REGISTER:
                worker_context = WorkerContext(event.from_rank)
                self.unassigned_workers.append(worker_context)
                self.worker_id_map[event.from_rank] = worker_context
                event = None
            elif event.type == EVENT_TYPE_TASK_RESPONSE:
                worker_context = self.worker_id_map[event.from_rank]
                self.unassigned_workers.append(worker_context)
                event = CompletedTaskEvent(
                    worker_context, event.data, worker_context.active_task
                )
            else:
                worker_context = self.worker_id_map[event.from_rank]
            worker_context.heartbeat_time = time.time()
            print(event)
        else:
            pass  # print(f"No response to process, status {self._status.count}")

        return event

    def send_to_worker(self, worker_context: WorkerContext, data):
        worker_context.active_task = data
        self.comm.send(
            Event(
                from_rank=self.comm.Get_rank(), type=EVENT_TYPE_TASK, data=data
            ),
            dest=worker_context.id,
            tag=self.tag,
        )

    def shutdown(self):
        for worker_id in self.worker_id_map.keys():
            self.comm.send(
                Event(from_rank=self.comm.Get_rank(), type=EVENT_TYPE_DONE),
                dest=worker_id,
                tag=self.tag,
            )
