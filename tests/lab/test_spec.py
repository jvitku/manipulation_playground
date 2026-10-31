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


def test_ladder_orders_unique_and_expected():
    import lab.tasks  # noqa: F401  (registers every built-in task)

    orders = [s.order for s in registry.all_specs()]
    assert len(orders) == len(set(orders)), "ladder orders must be unique"
    expected = {"ta-vla": 10, "td3": 11, "td3-bc": 12, "residual-rl": 13}
    for task_id, order in expected.items():
        assert registry.get(task_id).order == order, task_id


def test_which_selects_key_space():
    spec = _spec()
    spec.eval_params = [Param("episodes", "Episodes", "int", 5, min=1, max=50)]
    with pytest.raises(ValueError):
        registry.validate_params(spec, {"episodes": 3}, which="train")
    assert registry.validate_params(spec, {"episodes": "3"}, which="eval") == {"episodes": 3}
    with pytest.raises(ValueError):
        registry.validate_params(spec, {"lr": 0.2}, which="eval")
    assert registry.validate_params(spec, {"episodes": 3, "lr": 0.2})["lr"] == 0.2


def test_register_rejects_name_in_both_lists():
    spec = _spec("zz_dup")
    spec.eval_params = [Param("lr", "LR", "float", 0.1)]
    with pytest.raises(ValueError):
        registry.register(spec)


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["lr", "steps"])
def test_non_finite_rejected(name, bad):
    with pytest.raises(ValueError):
        registry.validate_params(_spec(), {name: bad})
