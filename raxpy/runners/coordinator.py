from typing import Optional, List, Dict
from dataclasses import dataclass, field
import time


@dataclass
class WorkerContext:
    id: int
    assignment_time: Optional[float] = None
    heartbeat_time: Optional[float] = None
    active_task: Optional[object] = None


@dataclass
class CompletedTaskEvent:
    worker_context: WorkerContext
    task_result: object
    task: object


@dataclass
class WorkerPool:
    unassigned_workers: List[WorkerContext] = field(default_factory=list)
    worker_id_map: Dict[str, WorkerContext] = field(default_factory=dict)

    def has_unassigned_workers(self) -> bool:
        """
        Checks to see if any workers are not working a task.
        """
        return len(self.unassigned_workers) > 0

    def delegate_task(self, task):
        """
        Delegates a task by getting a worker not assigned a
        task and assigning them the task by sending the task to them.
        """
        worker_context = self.unassigned_workers.pop()
        worker_context.active_task = task
        worker_context.assignment_time = time.time()

        self.send_to_worker(worker_context, task)

    def process_event(self) -> Optional[object]:
        """
        Checks for events from workers to process.
        """
        return self._process()

    def _process(self):
        raise NotImplementedError("Abstract class")

    def send_to_worker(self, worker_context, data):
        """
        Sends data to a worker.
        """
        raise NotImplementedError("Abstract class")

    def shutdown(self):
        """
        Shutdowns by, if applicable, notifing workers that no more
        is needed to be done.
        """
        pass  # default behavior do nothing


def coordinate(task_provider, worker_pool: WorkerPool):
    done = False
    # assign tasks to workers
    while not done:
        # check workers for completed events
        while True:
            event = worker_pool.process_event()
            if event is None:
                break
            else:
                if isinstance(event, CompletedTaskEvent):
                    task_provider.process_task_result(
                        event.task, event.task_result
                    )

        # assign task to worker
        while worker_pool.has_unassigned_workers():
            task = task_provider.next()
            if task is not None:
                worker_pool.delegate_task(task)
            else:
                if not task_provider.is_waiting_for_results():
                    done = True
                else:
                    pass
                break
    worker_pool.shutdown()
