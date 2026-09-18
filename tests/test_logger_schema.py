"""PLAN §5 schema validation: keys, shapes, dtypes, finite values, monotonic time."""

import numpy as np
import pytest

from fvb.control.gantry import Gantry
from fvb.ft.compensate import LoadParams
from fvb.logging.episode import HF_SCHEMA, SCHEMA, EpisodeLogger, load, validate
from fvb.scenes.builder import SceneParams, build_model


def _episode(tmp_path, with_image=False):
    p = SceneParams()
    m, d = build_model(p)
    g = Gantry(m, d, p, 20.0, load=LoadParams(p.peg_mass, np.array([0, 0, -p.peg_len / 2])))
    g.settle(0.5)
    log = EpisodeLogger(config={"x": 1}, seed=3)
    for k in range(15):
        g.step([0, 0, -0.001 * k], log)
    if with_image:
        log._rows["image"] = [np.zeros((8, 8, 3), np.uint8) for _ in range(15)]
    log.success = True
    return log.save(tmp_path / "ep.npz")


def test_saved_episode_matches_schema(tmp_path):
    path = _episode(tmp_path)
    arrs, meta = load(path)
    validate(arrs)
    T = len(arrs["t"])
    for k, (shape, dtype) in SCHEMA.items():
        assert arrs[k].dtype == dtype, k
        assert arrs[k].shape[0] == T, k
        for got, want in zip(arrs[k].shape[1:], shape, strict=True):
            assert want is None or got == want, k
    for k, (shape, dtype) in HF_SCHEMA.items():
        assert arrs[k].dtype == dtype and arrs[k].shape[1:] == shape, k
    assert len(arrs["ft_raw_hf"]) == T * 25  # 20 Hz control, 2 ms physics
    assert np.all(np.diff(arrs["t"]) > 0) and np.all(np.diff(arrs["t_hf"]) > 0)
    assert arrs["success"].dtype == np.bool_ and bool(arrs["success"])
    assert all(np.isfinite(arrs[k]).all() for k in arrs if arrs[k].dtype.kind == "f")
    assert meta["seed"] == 3 and meta["config"] == {"x": 1} and "mujoco" in meta["versions"]
    assert meta["T"] == T


def test_optional_image_key(tmp_path):
    arrs, _ = load(_episode(tmp_path, with_image=True))
    validate(arrs)
    assert arrs["image"].dtype == np.uint8 and arrs["image"].shape == (15, 8, 8, 3)


def test_validate_rejects_bad_data(tmp_path):
    arrs, _ = load(_episode(tmp_path))
    bad = dict(arrs)
    bad["ft_raw"] = bad["ft_raw"][:, :5]
    with pytest.raises(ValueError):
        validate(bad)
    bad = dict(arrs)
    bad["t"] = bad["t"][::-1].copy()
    with pytest.raises(ValueError):
        validate(bad)
    bad = dict(arrs)
    bad["ft_comp"] = bad["ft_comp"].copy()
    bad["ft_comp"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate(bad)
    bad = dict(arrs)
    del bad["contact_wrench"]
    with pytest.raises(ValueError):
        validate(bad)


def test_unknown_key_rejected():
    log = EpisodeLogger(config={}, seed=0)
    with pytest.raises(KeyError):
        log.step(t=0.0, made_up_key=1.0)
