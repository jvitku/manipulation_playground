"""TA-SmolVLA torque token (VLA image with the smolvla_base weights cached; GPU)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)


@pytest.fixture(scope="module")
def setup():
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from fvb.vla.ta_smolvla import make_ta_smolvla

    try:
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to("cuda")
    except Exception as e:  # noqa: BLE001  (no network / cache)
        pytest.skip(f"smolvla_base unavailable: {e}")
    pre, _ = make_pre_post_processors(policy.config, pretrained_path="lerobot/smolvla_base")
    make_ta_smolvla(policy)
    return policy, pre


def _batch(policy, torque):
    cfg = policy.config
    b = {}
    for k, v in cfg.input_features.items():
        b[k] = torch.rand(1, *v.shape) if "image" in k else torch.randn(1, *v.shape)
    b["action"] = torch.randn(1, cfg.chunk_size, *cfg.output_features["action"].shape)
    b["action_is_pad"] = torch.zeros(1, cfg.chunk_size, dtype=torch.bool)
    b["task"] = ["put the bolt in the hole"]
    b["observation.torque_hist"] = torque
    return b


def test_token_trains_and_changes_actions(setup):
    policy, pre = setup
    torch.manual_seed(0)
    t1 = torch.randn(1, 150)
    batch = pre(_batch(policy, t1))
    batch["observation.torque_hist"] = t1.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    g = policy.model.torque_mlp[0].weight.grad
    assert torch.isfinite(loss) and g is not None and g.abs().sum() > 0

    policy.eval()
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")
    outs = []
    for t in (t1, -5 * t1):
        b = dict(batch)
        b["observation.torque_hist"] = t.cuda()
        with torch.no_grad():
            outs.append(policy.predict_action_chunk(b, noise=noise))
    assert (outs[0] - outs[1]).abs().max() > 1e-4  # the token reaches the actions
    assert policy.model._torque is None  # cleared after every call
