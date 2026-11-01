#!/usr/bin/env python
"""Generate tests/nfv/fixtures/limoe_parity.npz from the REFERENCE JAX LIMoEBlock.

Run on the HOST (jax/flax are deliberately not in the fvb-lab image — plan §1 "parity by
fixture"), from the repo root:

    uv run --with "jax==0.4.38" --with "flax==0.10.2" --with numpy --with absl-py \
        python tests/nfv/gen_limoe_fixture.py

What it does:
  * imports the PINNED reference clone (third_party/forcevla @ 79de88f) and its VENDORED
    flaxformer (flaxformer is not on PyPI) by extending sys.path;
  * stubs two imports the reference drags in but never executes on this path:
    - ``openpi.shared.array_typing`` (imported by limoe_simple.py, unused there) — stubbing
      it avoids openpi's heavy dependency tree,
    - ``aqt.jax_legacy.jax.*`` (imported by flaxformer/components/dense.py; only live when
      ``use_aqt=True``, which the reference never sets) — the stub only has to satisfy the
      class-body annotations;
  * instantiates the reference ``LIMoEBlock(mlp_dim=64, num_experts=4, num_top_k=1,
    num_heads=2, out_dim=24)`` at width 32, in deterministic EVAL mode, on CPU;
  * runs a fixed 32-token input (1 batch x 32 tokens: 4 routing groups of 8, expert
    capacity max(round(8/4), 4) = 4) and dumps input + ALL params + output to
    tests/nfv/fixtures/limoe_parity.npz.

dtype pin (documented deviation): the reference computes router logits (RouterWeights
dtype=bfloat16) and expert inputs (MoeLayer dtype=bfloat16) in bfloat16 — a TPU-efficiency
choice. The course port is fp32 end-to-end (plan §1 fp32 discipline), so this generator pins
those two module dtypes to float32 to compare the MATH, not TPU rounding. Everything else is
verbatim reference code.

npz layout: "in:x" [1, 32, 32], "out:y" [1, 32, 24], "param:<flax/path>" for every leaf of
the 'params' collection, "meta:config" (json).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORCEVLA = REPO / "third_party" / "forcevla"
FIXTURE = REPO / "tests" / "nfv" / "fixtures" / "limoe_parity.npz"

CONFIG = {
    "d_model": 32,
    "mlp_dim": 64,
    "num_heads": 2,
    "out_dim": 24,
    "num_experts": 4,
    "num_top_k": 1,
    "expert_dim": 2048,  # flaxformer MlpBlock.intermediate_dim default, inherited by the ref
    "tokens": 32,
    "batch": 1,
    "seed_params": 0,
    "seed_input": 1,
    "dtype_pin": "float32 (reference uses bfloat16 for router logits + expert inputs)",
}


def _stub_modules() -> None:
    """Pre-register stub modules so the reference imports resolve without their dep trees."""
    # openpi.shared.array_typing — imported as `at` by limoe_simple.py, never used there.
    for name in ("openpi", "openpi.shared", "openpi.shared.array_typing"):
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark packages
        sys.modules.setdefault(name, mod)

    # aqt.jax_legacy.jax.{flax_layers,quant_config,quantization} — flaxformer dense.py
    # imports them at module level but only touches them when use_aqt=True (never here).
    # The quantization stub must carry the types used in dataclass ANNOTATIONS (evaluated
    # at class-body time): QuantOps.WeightParams / QuantOps.ActHParams, QuantType.AQT.
    class _QuantOps:
        WeightParams = object
        ActHParams = object

    class _QuantType:
        AQT = "aqt"

    flax_layers = types.ModuleType("aqt.jax_legacy.jax.flax_layers")
    quant_config = types.ModuleType("aqt.jax_legacy.jax.quant_config")
    quantization = types.ModuleType("aqt.jax_legacy.jax.quantization")
    quantization.QuantOps = _QuantOps
    quantization.QuantType = _QuantType
    for name in ("aqt", "aqt.jax_legacy", "aqt.jax_legacy.jax"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules.setdefault(name, pkg)
    sys.modules.setdefault("aqt.jax_legacy.jax.flax_layers", flax_layers)
    sys.modules.setdefault("aqt.jax_legacy.jax.quant_config", quant_config)
    sys.modules.setdefault("aqt.jax_legacy.jax.quantization", quantization)


def _load_limoe_simple():
    """Load the reference limoe_simple.py by file path (bypasses openpi/__init__.py)."""
    path = FORCEVLA / "src" / "openpi" / "models" / "limoe_simple.py"
    spec = importlib.util.spec_from_file_location("ref_limoe_simple", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_limoe_simple"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    sys.path.insert(0, str(FORCEVLA / "flaxformer"))  # vendored flaxformer package root
    _stub_modules()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict
    from flaxformer.architectures.moe import routing

    limoe = _load_limoe_simple()

    # dtype pin (see module docstring): float32 instead of the reference's bfloat16 for the
    # router weights and the MoeLayer expert path. functools.partial keeps the classes (and
    # therefore the flax auto-naming and isinstance checks) untouched.
    routing.RouterWeights = functools.partial(routing.RouterWeights, dtype=jnp.float32)
    limoe.MoeLayer = functools.partial(limoe.MoeLayer, dtype=jnp.float32)

    block = limoe.LIMoEBlock(
        mlp_dim=CONFIG["mlp_dim"],
        num_heads=CONFIG["num_heads"],
        out_dim=CONFIG["out_dim"],
        num_experts=CONFIG["num_experts"],
        num_top_k=CONFIG["num_top_k"],
    )
    x = jax.random.normal(
        jax.random.PRNGKey(CONFIG["seed_input"]),
        (CONFIG["batch"], CONFIG["tokens"], CONFIG["d_model"]),
        dtype=jnp.float32,
    )
    variables = block.init(jax.random.PRNGKey(CONFIG["seed_params"]), x, True)  # eval mode
    params = variables["params"]

    out = block.apply({"params": params}, x, True, mutable=["intermediates"])
    (y, _), _ = out  # LIMoEBlock returns (x, None); mutable apply wraps it in (out, vars)

    flat = flatten_dict(params)
    arrays = {
        "in:x": np.asarray(x, dtype=np.float32),
        "out:y": np.asarray(y, dtype=np.float32),
        "meta:config": np.array(json.dumps(CONFIG)),
    }
    for path, leaf in flat.items():
        arrays[f"param:{'/'.join(path)}"] = np.asarray(leaf, dtype=np.float32)

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size / 1e6:.2f} MB)")
    print(f"output y: shape {y.shape}, mean {float(np.mean(y)):+.6f}")
    for key in sorted(arrays):
        if key.startswith("param:"):
            print(f"  {key}  {arrays[key].shape}")


if __name__ == "__main__":
    main()
