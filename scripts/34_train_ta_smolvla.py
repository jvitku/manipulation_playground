#!/usr/bin/env python
"""V9: run ``lerobot-train`` with SmolVLA turned into TA-SmolVLA (VLA image).

    TA_MODE=ta      python scripts/34_train_ta_smolvla.py --policy.path=lerobot/smolvla_base ...
    TA_MODE=ta_zero ...   (same architecture, torque token input zeroed: the no-force twin)
    TA_MODE=off     ...   (plain SmolVLA, for reference)

Every ``SmolVLAPolicy`` built in this process gets the torque token (``fvb.vla.ta_smolvla``);
its weights are part of the model's state dict, so checkpoints carry them. The same patch is
applied by ``scripts/24_serve_policy.py --ta {ta,ta_zero}`` when serving.
"""

from __future__ import annotations

import os
import sys


def patch(mode: str) -> None:
    if mode == "off":
        return
    from lerobot.policies.smolvla import modeling_smolvla as m

    from fvb.vla.ta_smolvla import make_ta_smolvla

    orig = m.SmolVLAPolicy.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        make_ta_smolvla(self, zero_token=(mode == "ta_zero"))

    m.SmolVLAPolicy.__init__ = init


if __name__ == "__main__":
    patch(os.environ.get("TA_MODE", "ta"))
    from lerobot.scripts.lerobot_train import main

    sys.exit(main())
