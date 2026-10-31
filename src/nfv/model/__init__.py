"""nfv.model — the nano force-VLA family (lesson 16's nanoVLA core, flags for 17/18).

Entry points:
    ModelConfig   every course switch in one dataclass (nfv.model.config)
    build(cfg)    validate the config, return the Policy (loss / act / EMA)

Blocks (one file per plan-table row): vision (conv stem -> 72 tokens), text (vocabulary +
1-layer encoder), trunk (the 4-layer "VLM" over [image ; text]), expert (the action decoder),
flow (flow matching + Euler sampler), build (Policy assembly). Training lives in nfv.train.
"""

# Import the build SUBMODULE before re-exporting the build FUNCTION: a submodule import
# anywhere (`from nfv.model.build import param_table`) sets the package attribute
# `nfv.model.build` to the MODULE, shadowing the function for any later
# `from nfv.model import build`. Loading the submodule here first and rebinding the name
# below keeps the function exported in every import order (later submodule imports are
# sys.modules cache hits and do not re-set the attribute).
import nfv.model.build  # noqa: F401  (rebound to the function just below)
from nfv.model.config import ModelConfig, build

__all__ = ["ModelConfig", "build"]
