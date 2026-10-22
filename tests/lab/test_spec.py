import json

import pytest

from lab.tasks import registry
from lab.tasks.base import Param, TaskSpec


def _spec(task_id="spec_dummy", order=1, estimate=None):
    return TaskSpec(
        id=task_id,
        order=order,
        title="T",
        subtitle="s",
        params=[
            Param("lr", "LR", "float", 0.1, min=0.0, max=1.0),
            Param("steps", "Steps", "int", 1000, min=1, max=5000),
            Param("flag", "Flag", "bool", False),
            Param("mode", "Mode", "choice", "a", choices=["a", "b"]),
        ],
        eval_params=[],
        charts=[],
        viewer="t1",
        estimate=estimate or (lambda p: 0.0),
        train=lambda ctx: None,
        evaluate=lambda ctx: None,
        lesson="x.md",
    )


def test_validate_clamps_and_coerces():
    spec = _spec()
    out = registry.validate_params(spec, {"lr": "0.5", "steps": "99999", "flag": "true"})
    assert out == {"lr": 0.5, "steps": 5000, "flag": True, "mode": "a"}
    assert isinstance(out["steps"], int)
    assert registry.validate_params(spec, {"lr": -3})["lr"] == 0.0
    assert registry.validate_params(spec, {"flag": "false"})["flag"] is False
    with pytest.raises(ValueError):
        registry.validate_params(spec, {"bogus": 1})
    with pytest.raises(ValueError):
        registry.validate_params(spec, {"mode": "c"})


def test_estimate_uses_calib(tmp_path, monkeypatch):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"spec_dummy": {"steps_per_s": 100}}))
    monkeypatch.setattr(registry, "CALIB_PATH", path)
    spec = _spec(estimate=lambda p: p["steps"] / registry.calib("spec_dummy")["steps_per_s"])
    assert spec.estimate({"steps": 1000}) == 10.0
    assert registry.calib("missing") == {}
    path.unlink()
    assert registry.calib("spec_dummy") == {}


def test_all_specs_sorted():
    registry.register(_spec("zz_b", order=902))
    registry.register(_spec("zz_a", order=901))
    ids = [s.id for s in registry.all_specs() if s.id.startswith("zz_")]
    assert ids == ["zz_a", "zz_b"]
