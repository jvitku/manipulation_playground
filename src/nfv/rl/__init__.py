"""Lesson 19 (`force-rl`): runtime residual TD3+BC on a frozen nano(Force)VLA.

Entry point for the lab task wrapper: :func:`nfv.rl.loop.run_residual_rl`.
"""

from nfv.rl.loop import evaluate, load_base, run_residual_rl
from nfv.rl.residual import DP_RES, F_RES, bc_weight_at

__all__ = ["DP_RES", "F_RES", "bc_weight_at", "evaluate", "load_base", "run_residual_rl"]
