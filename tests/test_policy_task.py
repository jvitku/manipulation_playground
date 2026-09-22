"""Stage 1 task, expert and dataset plumbing (torch-free; the models are tested by make s1)."""

import numpy as np

from fvb.policy.data import compute_norm, make_windows, split_episodes
from fvb.policy.task import ACT_DIM, FORCE_IDX, OBS_DIM, GantryTask, TaskParams


def _expert_episode(seed):
    task = GantryTask(TaskParams(), seed)
    obs, act, done, reason = [], [], False, None
    while not done:
        obs.append(task.observe())
        a = task.expert_action()
        act.append(a.astype(np.float32))
        done, reason = task.step(a)
    return {
        "obs": np.stack(obs),
        "action": np.stack(act),
        "success": reason == "success",
        "hole": task.hole_xy,
    }


def test_expert_succeeds_and_observation_hides_hole():
    eps = [_expert_episode(s) for s in (1, 2, 3)]
    assert all(e["success"] for e in eps)
    for e in eps:
        assert e["obs"].shape[1] == OBS_DIM and e["action"].shape[1] == ACT_DIM
        # first observation is identical across episodes: the hole offset is not observable
        assert np.allclose(e["obs"][0, :6], eps[0]["obs"][0, :6], atol=1e-9)


def test_windows_and_ablation():
    eps = [_expert_episode(s) for s in (4, 5)]
    tr, va = split_episodes(eps, 0.5, 0)
    norm = compute_norm(tr, use_force=False)
    assert np.all(norm.obs_std[FORCE_IDX] == 1e-6)  # zeroed channels get the std floor
    X, Y, M = make_windows(eps, H=10, K=5, use_force=False, norm=norm)
    assert X.shape[1:] == (10, OBS_DIM) and Y.shape[1:] == (5, ACT_DIM) and M.shape[1] == 5
    assert np.all(X[..., FORCE_IDX] == 0.0)
    assert M[-1].sum() == 1 and M[0].sum() == 5  # mask past the episode end
    assert len(X) == sum(len(e["obs"]) for e in eps)
