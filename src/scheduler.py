"""
src/scheduler.py

Lightweight DAG task scheduler with UI-ready progress reporting.

Design
------
Tasks are defined with explicit dependency names and resource types ("gpu"/"cpu").
The scheduler resolves the graph, runs independent tasks in parallel, and gates
GPU tasks through GpuGate so VRAM is never over-committed.

UI contract
-----------
Every state change emits a ``ProgressEvent`` to the caller-supplied
``progress_callback``.  The UI only needs to subscribe to this single channel —
it receives task status, per-task progress, per-task messages, and pipeline-wide
overall_fraction.  No polling required.

Cancellation
------------
Pass a ``threading.Event`` as ``cancel_event``.  Set it from any thread (e.g.
a UI "Cancel" button).  The scheduler stops launching new tasks and running
tasks receive the same event so they can exit their inner loops cleanly.

Checkpointing
-------------
If all of a task's declared output files already exist and are newer than all
declared input files, the task is skipped automatically.  Re-running the
pipeline after a partial failure picks up exactly where it left off.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable


# ── Public types ──────────────────────────────────────────────────────────────

class TaskStatus(Enum):
    PENDING   = auto()   # not yet examined
    WAITING   = auto()   # deps not done yet
    RUNNING   = auto()   # currently executing
    DONE      = auto()   # completed successfully
    SKIPPED   = auto()   # outputs already up-to-date
    FAILED    = auto()   # raised an exception
    CANCELLED = auto()   # cancelled before it could run


@dataclass
class ProgressEvent:
    """
    Single progress notification emitted by the scheduler.

    UI can bind directly to these fields:
      task_name       → which task bar to update
      fraction        → per-task 0.0-1.0 progress bar fill
      overall_fraction→ pipeline-wide progress bar fill
      message         → status text under the bar
      status          → colour/icon selector (RUNNING=blue, DONE=green, ...)
      error           → non-None when status == FAILED
    """
    task_name:        str
    current:          int
    total:            int
    message:          str        = ""
    fraction:         float      = 0.0
    overall_fraction: float      = 0.0
    status:           TaskStatus = TaskStatus.RUNNING
    error:            str | None = None


# Callback type the UI (or CLI printer) implements
ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class Task:
    """
    One unit of work in the pipeline DAG.

    Parameters
    ----------
    name     : unique identifier used in deps lists and progress events
    fn       : callable — must accept keyword args
                  progress_callback(current, total, message)
                  cancel_event: threading.Event
               Additional args/kwargs are passed via task_args / task_kwargs.
    deps     : names of tasks that must be DONE or SKIPPED before this starts
    resource : "gpu" acquires GpuGate before calling fn; "cpu" runs freely
    inputs   : files that must exist (checked before fn is called)
    outputs  : files produced (used for checkpointing + overall_fraction weight)
    weight   : relative time cost — used to compute overall_fraction
    task_args   : positional args forwarded to fn
    task_kwargs : keyword args forwarded to fn (merged with scheduler-injected ones)
    """
    name:        str
    fn:          Callable
    deps:        list[str]  = field(default_factory=list)
    resource:    str        = "cpu"          # "gpu" | "cpu"
    inputs:      list[Path] = field(default_factory=list)
    outputs:     list[Path] = field(default_factory=list)
    weight:      float      = 1.0
    task_args:   tuple      = field(default_factory=tuple)
    task_kwargs: dict       = field(default_factory=dict)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class Scheduler:
    """
    Runs a set of ``Task`` objects as a directed acyclic graph.

    Thread model
    ------------
    - One thread per running task (bounded by ``max_workers``).
    - GPU tasks additionally hold ``GpuGate`` for their duration.
    - The main ``run()`` call blocks until the graph is fully resolved or
      cancelled; it is itself thread-safe so the UI can call it from a
      background QThread / asyncio task.

    UI hook — status snapshot
    -------------------------
    ``scheduler.snapshot()`` returns a ``PipelineSnapshot`` that the UI can
    poll (or receive via signal) for a complete picture of all tasks.
    """

    def __init__(
        self,
        gpu_gate=None,
        progress_callback: ProgressCallback | None = None,
        cancel_event:      threading.Event | None  = None,
        max_workers:       int = 6,
    ):
        from .resources import GPU_GATE
        self._gpu_gate        = gpu_gate if gpu_gate is not None else GPU_GATE
        self._progress_cb     = progress_callback
        self._cancel          = cancel_event or threading.Event()
        self._max_workers     = max_workers

        self._tasks:    dict[str, Task]       = {}
        self._status:   dict[str, TaskStatus] = {}
        self._progress: dict[str, float]      = {}
        self._messages: dict[str, str]        = {}
        self._errors:   dict[str, str]        = {}
        self._results:  dict[str, Any]        = {}
        self._lock      = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, task: Task) -> "Scheduler":
        """Register a task. Returns self for chaining."""
        self._tasks[task.name]    = task
        self._status[task.name]   = TaskStatus.PENDING
        self._progress[task.name] = 0.0
        self._messages[task.name] = "Pending"
        return self

    def cancel(self) -> None:
        """Signal all running tasks to stop. Safe to call from any thread."""
        self._cancel.set()

    def result(self, task_name: str) -> Any:
        """Return the return value of a completed task."""
        return self._results.get(task_name)

    def run(self) -> bool:
        """
        Execute the DAG.  Blocks until done or cancelled.
        Returns True if every task succeeded (or was skipped).
        """
        pending      = set(self._tasks.keys())
        failed_deps: set[str] = set()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            in_flight: dict[str, Future] = {}

            while (pending or in_flight) and not self._cancel.is_set():

                # Collect finished futures
                newly_done = [n for n, f in in_flight.items() if f.done()]
                for name in newly_done:
                    fut = in_flight.pop(name)
                    if not fut.result():
                        failed_deps.add(name)

                # Find tasks ready to launch
                for name in list(pending):
                    task = self._tasks[name]
                    dep_statuses = [self._status.get(d, TaskStatus.PENDING)
                                    for d in task.deps]

                    if any(d in failed_deps for d in task.deps):
                        pending.discard(name)
                        failed_deps.add(name)
                        self._set_status(name, TaskStatus.CANCELLED)
                        self._emit(ProgressEvent(
                            task_name=name, current=0, total=1,
                            message="Cancelled — dependency failed",
                            status=TaskStatus.CANCELLED,
                        ))
                        continue

                    deps_done = all(
                        s in (TaskStatus.DONE, TaskStatus.SKIPPED)
                        for s in dep_statuses
                    )
                    if deps_done:
                        pending.discard(name)
                        self._set_status(name, TaskStatus.WAITING)
                        in_flight[name] = pool.submit(self._execute, task)

                if not newly_done and not any(
                    name in pending for name in self._tasks
                    if all(
                        self._status.get(d) in (TaskStatus.DONE, TaskStatus.SKIPPED)
                        for d in self._tasks[name].deps
                    )
                ):
                    time.sleep(0.05)

        # Mark remaining pending/waiting as cancelled
        for name, status in self._status.items():
            if status in (TaskStatus.PENDING, TaskStatus.WAITING):
                self._set_status(name, TaskStatus.CANCELLED)

        return not self._errors and not self._cancel.is_set()

    def snapshot(self) -> "PipelineSnapshot":
        """
        Thread-safe snapshot of full pipeline state.
        UI can call this at any time to refresh its display.
        """
        with self._lock:
            return PipelineSnapshot(
                tasks=dict(self._tasks),
                status=dict(self._status),
                progress=dict(self._progress),
                messages=dict(self._messages),
                errors=dict(self._errors),
                overall_fraction=self._overall_fraction(),
                is_running=any(
                    s == TaskStatus.RUNNING for s in self._status.values()
                ),
                is_cancelled=self._cancel.is_set(),
            )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _set_status(self, name: str, status: TaskStatus) -> None:
        with self._lock:
            self._status[name] = status

    def _overall_fraction(self) -> float:
        total_w = sum(t.weight for t in self._tasks.values())
        if total_w == 0:
            return 0.0
        done_w = sum(
            t.weight * self._progress.get(t.name, 0.0)
            for t in self._tasks.values()
        )
        return done_w / total_w

    def _emit(self, event: ProgressEvent) -> None:
        with self._lock:
            self._progress[event.task_name] = event.fraction
            self._messages[event.task_name] = event.message
            event.overall_fraction = self._overall_fraction()
        if self._progress_cb:
            self._progress_cb(event)

    def _should_skip(self, task: Task) -> bool:
        if not task.outputs:
            return False
        if not all(p.exists() for p in task.outputs):
            return False
        if not task.inputs:
            return True
        try:
            oldest_out = min(p.stat().st_mtime for p in task.outputs)
            newest_in  = max(
                (p.stat().st_mtime for p in task.inputs if p.exists()),
                default=0.0,
            )
            return oldest_out >= newest_in
        except OSError:
            return False

    def _make_task_progress(self, task: Task) -> Callable:
        """Returns a progress reporter bound to this task."""
        def _cb(current: int, total: int, message: str = "") -> None:
            if self._cancel.is_set():
                return
            frac = current / max(total, 1)
            self._emit(ProgressEvent(
                task_name=task.name,
                current=current,
                total=total,
                message=message,
                fraction=frac,
                status=TaskStatus.RUNNING,
            ))
        return _cb

    def _execute(self, task: Task) -> bool:
        """Run one task on a worker thread. Returns True on success."""
        if self._cancel.is_set():
            self._set_status(task.name, TaskStatus.CANCELLED)
            return False

        # Checkpoint
        if self._should_skip(task):
            self._set_status(task.name, TaskStatus.SKIPPED)
            self._emit(ProgressEvent(
                task_name=task.name, current=1, total=1,
                message="Skipped — outputs already up-to-date",
                fraction=1.0, status=TaskStatus.SKIPPED,
            ))
            return True

        # Acquire GPU slot if needed
        acquired_gpu = False
        if task.resource == "gpu":
            self._emit(ProgressEvent(
                task_name=task.name, current=0, total=1,
                message="Waiting for GPU slot...",
                fraction=0.0,
            ))
            self._gpu_gate.acquire()
            acquired_gpu = True

        self._set_status(task.name, TaskStatus.RUNNING)
        self._emit(ProgressEvent(
            task_name=task.name, current=0, total=1,
            message="Starting...", fraction=0.0,
        ))

        try:
            result = task.fn(
                *task.task_args,
                progress_callback=self._make_task_progress(task),
                cancel_event=self._cancel,
                **task.task_kwargs,
            )
            self._results[task.name] = result
            self._set_status(task.name, TaskStatus.DONE)
            self._emit(ProgressEvent(
                task_name=task.name, current=1, total=1,
                message="Done", fraction=1.0, status=TaskStatus.DONE,
            ))
            return True

        except Exception as exc:
            err = traceback.format_exc()
            with self._lock:
                self._errors[task.name] = err
            self._set_status(task.name, TaskStatus.FAILED)
            self._emit(ProgressEvent(
                task_name=task.name, current=0, total=1,
                message=f"Failed: {exc}",
                fraction=0.0, status=TaskStatus.FAILED, error=err,
            ))
            return False

        finally:
            if acquired_gpu:
                self._gpu_gate.release()


# ── Pipeline snapshot (for UI polling / signals) ──────────────────────────────

@dataclass
class PipelineSnapshot:
    """
    Complete pipeline state at one instant.
    Safe to pass across thread boundaries (all fields are plain Python objects).
    UI reads this to refresh its entire display in one go.
    """
    tasks:            dict[str, Task]
    status:           dict[str, TaskStatus]
    progress:         dict[str, float]
    messages:         dict[str, str]
    errors:           dict[str, str]
    overall_fraction: float
    is_running:       bool
    is_cancelled:     bool

    def summary(self) -> str:
        lines = [f"Pipeline  {self.overall_fraction*100:.0f}%"]
        for name, status in self.status.items():
            pct = self.progress.get(name, 0.0) * 100
            lines.append(f"  {name:30s} {status.name:10s} {pct:5.1f}%  "
                         f"{self.messages.get(name, '')}")
        return "\n".join(lines)
