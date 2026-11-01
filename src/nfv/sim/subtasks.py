"""The ForceVLA2 force prompt: a per-task subtask state machine driven by predicted progress.

ForceVLA2 (arXiv 2603.15169) cuts every task into ordered subtasks and appends the running
subtask's text to the instruction, ``Z_l = g([T_t ; T_f])`` — the only place force semantics
reach the language side. At run time the policy's predicted progress ``s`` drives the
machine: when ``s`` crosses a threshold the prompt advances to the next subtask string.

The subtask strings are NOT invented here: they are each task's ``SUBTASKS`` tuple in
``nfv/sim/tasks.py`` (the labels the scripted experts already record per step, via their
``PHASE_SUBTASK`` maps), so the prompt the policy sees at run time is the word it was trained
on. Only the subtasks some expert phase maps to are used (``task_subtasks``); every one of
those words is already in the course vocabulary (``nfv.model.text.VOCAB``):

    t0  reach / grasp / lift
    t1  approach / press / done
    t2  approach / wipe                       (SUBTASKS' "done" is never labelled)
    t3  approach / descend / search / insert  (SUBTASKS' "done" is never labelled)

The progress the policy learns (and the machine is driven by) is ``hindsight_progress`` —
per-subtask, computed from the recorded subtask labels — not the paper's eq. 10 label
``s_fv2``: see that function for the measured reason.

Choices the paper leaves open (written down here and in lesson 18):

* **Threshold with hysteresis.** The paper says the machine advances when ``s`` "exceeds 1"
  while bounding ``s`` to [0, 1] — an event that can never fire for a bounded regressor. We
  resolve it as: advance when ``s >= ADVANCE_AT`` (0.8), then DISARM; the machine re-arms
  only once ``s < REARM_BELOW`` (0.3). The paper's "reset s to 0" becomes "wait for the
  model's own s to fall": after the prompt changes, a model trained on per-subtask progress
  predicts a low ``s`` for the new subtask, which re-arms the machine. Without the dead band a
  noisy ``s`` hovering at 0.8 would skip several subtasks within one chunk.
* **Which s.** ``update`` is called once per EXECUTED chunk step (the executor runs
  ``n_exec`` = 8 of the 16 predicted steps), in order, with that step's predicted ``s``; the
  prompt used for the next re-plan is ``prompt(instruction)`` after those updates.
* **The last subtask is absorbing.** Crossing the threshold in the last subtask does nothing.

Usage::

    sm = SubtaskMachine.for_task("t2")
    text = sm.prompt("wipe the plate gently")     # "wipe the plate gently approach"
    for s in chunk_s[:n_exec]:
        sm.update(s)
"""

from __future__ import annotations

import numpy as np

ADVANCE_AT = 0.8  # predicted progress at/above this advances the prompt (when armed)
REARM_BELOW = 0.3  # after an advance, s must fall below this before the next advance


def task_subtasks(task: str) -> tuple[str, ...]:
    """The subtasks the task's expert actually LABELS, in order ("t0".."t3").

    Read from the expert's ``PHASE_SUBTASK`` map into ``TASKS[task].SUBTASKS``, so a subtask no
    demo ever carries is not part of the run-time machine: t2's and t3's trailing "done" are
    never labelled (the wipe / insert phase runs until the episode ends), so prompting it
    would show the policy a word it was never trained on.
    """
    from nfv.sim.experts import EXPERTS  # local import: the sim pulls MuJoCo
    from nfv.sim.tasks import TASKS

    names = TASKS[task].SUBTASKS
    ids = sorted(set(EXPERTS[task].PHASE_SUBTASK))
    return tuple(names[i] for i in ids)


def hindsight_progress(subtask_ids) -> np.ndarray:
    """Per-step progress label [T] in [0, 1] from one episode's subtask labels [T] (int).

    Hindsight, per subtask: inside each contiguous run of one label (length n, step k = 0..n-1)
    ``s = k / (n - 1)`` — 0 on the subtask's first step, 1 on its last (a 1-step run is 1).
    So the label starts below REARM_BELOW at every subtask entry and crosses ADVANCE_AT at 80 %
    of the subtask's duration: a model that regresses it well switches the prompt ~20 % of a
    subtask early, never late.

    Why not the recorded labels (measured with the scripted experts, 4 seeds per task, fed
    straight into ``SubtaskMachine``; "agree" = machine subtask == expert label per step):

    * ``s_fv2`` (paper eq. 10) restarts at every expert PHASE, not every subtask (t0/t1/t2 map
      two phases to one subtask), and its free-space factor ``exp(-2 l)`` is already 0.82 at
      l = 0.1 m, so the very first step reads s ≈ 0.8 and the machine advances at once:
      agree t0 0.23, t1 0.56, t2 0.61, t3 0.15.
    * ``s_lin`` rescaled to the current subtask's phases rises only while a phase's distance
      shrinks: t2's force-settling "engage" phase and t3's spiral "search" never pass ~0.5,
      so the machine never leaves "approach" (agree t2 0.72 — all of it in approach — t3 0.25).
    * this label: agree t0 0.90, t1 0.83, t2 0.93, t3 0.88 (the gap is the 20 %-early switch).
    """
    ids = np.asarray(subtask_ids)
    s = np.ones(len(ids), dtype=np.float32)
    start = 0
    for t in range(1, len(ids) + 1):
        if t == len(ids) or ids[t] != ids[start]:  # run [start, t) ends here
            n = t - start
            if n > 1:
                s[start:t] = np.arange(n, dtype=np.float32) / (n - 1)
            start = t
    return s


class SubtaskMachine:
    """Ordered subtask strings + the progress threshold with hysteresis (module docstring)."""

    def __init__(
        self,
        subtasks: tuple[str, ...],
        advance_at: float = ADVANCE_AT,
        rearm_below: float = REARM_BELOW,
    ):
        if not subtasks:
            raise ValueError("a subtask machine needs at least one subtask")
        if not 0.0 <= rearm_below < advance_at <= 1.0:
            raise ValueError("need 0 <= rearm_below < advance_at <= 1")
        self.subtasks = tuple(subtasks)
        self.advance_at = advance_at
        self.rearm_below = rearm_below
        self.reset()

    @classmethod
    def for_task(cls, task: str, **kw) -> SubtaskMachine:
        """The machine over the subtasks the task's expert labels (``task_subtasks``)."""
        return cls(task_subtasks(task), **kw)

    def reset(self) -> None:
        """Episode start: first subtask, armed."""
        self.index = 0
        self.armed = True

    @property
    def current(self) -> str:
        return self.subtasks[self.index]

    @property
    def is_last(self) -> bool:
        return self.index == len(self.subtasks) - 1

    def update(self, s: float) -> bool:
        """Feed one predicted progress value; returns True iff the prompt advanced."""
        s = float(s)
        if not self.armed:
            if s < self.rearm_below:
                self.armed = True  # the new subtask's progress has visibly restarted
            return False
        if s >= self.advance_at and not self.is_last:
            self.index += 1
            self.armed = False
            return True
        return False

    def prompt(self, instruction: str) -> str:
        """The force prompt: the instruction with the running subtask's word appended."""
        return f"{instruction} {self.current}"
