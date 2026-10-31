"""nfv.model — the nano force-VLA family (lesson 16's nanoVLA core, flags for 17/18).

Entry points:
    ModelConfig   every course switch in one dataclass (nfv.model.config)
    build(cfg)    validate the config, return the Policy (loss / act / EMA)

Blocks (one file per plan-table row): vision (conv stem -> 72 tokens), text (vocabulary +
1-layer encoder), trunk (the 4-layer "VLM" over [image ; text]), expert (the action decoder),
flow (flow matching + Euler sampler), build (Policy assembly). Training lives in nfv.train.
"""

from nfv.model.config import ModelConfig, build

__all__ = ["ModelConfig", "build"]
