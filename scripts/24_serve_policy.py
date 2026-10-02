#!/usr/bin/env python
"""Serve a trained LeRobot policy (ACT, SmolVLA, ...) to the simulator over TCP (VLA image).

    python scripts/24_serve_policy.py --ckpt outputs/act_v1/checkpoints/last/pretrained_model

The simulator side (``fvb.vla.sort_io.run_episode`` + ``PolicyClient``) sends ``policy_obs``
dicts (uint8 HWC images, float32 vectors, the task string); this turns them into a batch of one,
runs the checkpoint's own pre-processor, ``select_action`` (which executes the action chunk
step by step) and post-processor, and returns the 7-D action.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="pretrained_model directory")
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--rename",
        default="",
        help="obs key renames as src=dst pairs, e.g. for smolvla_base: "
        "observation.images.front=observation.images.camera1,"
        "observation.images.wrist=observation.images.camera2",
    )
    ap.add_argument(
        "--ta",
        choices=("off", "ta", "ta_zero", "ta_norm"),
        default="off",
        help="TA-SmolVLA checkpoints (scripts/34_train_ta_smolvla.py): rebuild the torque token",
    )
    ap.add_argument(
        "--n-action-steps",
        type=int,
        default=0,
        help="> 0: execute only this many steps of each predicted chunk (re-plan more often)",
    )
    args = ap.parse_args()

    if args.ta != "off":
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "ta_train", Path(__file__).with_name("34_train_ta_smolvla.py")
        )
        ta_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ta_train)
        ta_train.patch(args.ta)

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from fvb.vla.transport import PolicyServer

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = args.device
    policy = (
        get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg).to(args.device).eval()
    )
    dev = {"device_processor": {"device": args.device}}
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides=dev,
        postprocessor_overrides=dev,
    )
    if args.n_action_steps > 0:
        policy.config.n_action_steps = args.n_action_steps
        policy.reset()
    wanted = set(policy.config.input_features)
    if args.ta != "off":  # the torque token reads it outside the config's input features
        wanted.add("observation.torque_hist")
    rename = dict(kv.split("=", 1) for kv in args.rename.split(",") if kv)

    def act(obs: dict) -> np.ndarray:
        batch = {}
        for k, v in obs.items():
            k = rename.get(k, k)
            if k == "task":
                batch[k] = [v]
            elif k not in wanted:
                continue
            elif k.startswith("observation.images."):
                t = torch.from_numpy(v).permute(2, 0, 1).float() / 255.0
                batch[k] = t.unsqueeze(0)
            else:
                batch[k] = torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0)
        with torch.inference_mode():
            a = post(policy.select_action(pre(batch)))
        return a.squeeze(0).float().cpu().numpy()

    print(f"serving {cfg.type} from {args.ckpt} on :{args.port}", flush=True)
    PolicyServer(act, reset=policy.reset, port=args.port).serve()


if __name__ == "__main__":
    main()
