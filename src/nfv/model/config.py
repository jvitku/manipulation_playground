"""ModelConfig: every switch of the force-VLA school in one dataclass (plan §4).

Every variant the course trains — nanoVLA, the ForceVLA routes, ForceVLA2, every ablation —
is a value of this config, never a subclass. The flags for lessons 17/18 are declared NOW so
the later modules only implement behaviour, never re-plumb configuration; ``build`` refuses
them with NotImplementedError until their lesson lands.

Architecture fields (all defaults = the plan's nanoVLA table, ≈ 7 M parameters):
    d_model 256 everywhere; two 96x96 cameras -> 72 vision tokens; vocabulary ≤ 64 words,
    ≤ 12 text tokens; trunk 4 layers x 4 heads, FFN 768; expert 4 decoder layers; chunk 16
    with 8 executed steps; 10 Euler steps at inference; state 8-D; action 4-D (dp xyz +
    gripper — the gantry).

Course flags (lessons 17-19; defaults = nanoVLA, i.e. everything off):
    force_in       where the wrench enters: none | prefix_linear | prefix_moe | concat_after |
                   fvlmoe | forcevla2 | prompt_bins (ForceVLA Table 3 + ForceVLA2 + ForceBench)
    moe            router of the fvlmoe fusion: top1_capacity (the reference port) | top1 | soft
    n_experts      experts in the fvlmoe MoE (paper: 4)
    balance_loss   add a load-balancing loss (the reference computes one but never applies it)
    expert_inputs  ForceVLA2 expert ablation: all | vision_only | none
    force_prompt   ForceVLA2: append the running subtask's words to the instruction
    head           position (Δp + gripper) | hybrid (+ f_target 6 + progress 1, lesson 18)
    force_noise    eval-time twin/FACT checks: off | zero | shuffle | gaussian
"""

from __future__ import annotations

from dataclasses import dataclass

from nfv.model.text import MAX_TEXT, VOCAB_SIZE

FORCE_IN = (
    "none",
    "prefix_linear",
    "prefix_moe",
    "concat_after",
    "fvlmoe",
    "forcevla2",
    "prompt_bins",
)
MOE = ("top1_capacity", "top1", "soft")
EXPERT_INPUTS = ("all", "vision_only", "none")
HEAD = ("position", "hybrid")
FORCE_NOISE = ("off", "zero", "shuffle", "gaussian")


@dataclass
class ModelConfig:
    """All knobs of the nano force-VLA family; see the module docstring for the legend."""

    # architecture (plan §4 table)
    d_model: int = 256
    cameras: int = 2
    img: int = 96
    vocab: int = VOCAB_SIZE
    max_text: int = MAX_TEXT
    trunk_layers: int = 4
    heads: int = 4
    ffn: int = 768
    expert_layers: int = 4
    chunk: int = 16  # H: actions per predicted chunk
    n_exec: int = 8  # executed steps per chunk before re-planning
    euler_steps: int = 10  # K: integration steps at inference
    d_state: int = 8  # env state vector width (env.py legend)
    act_dim: int = 4  # dp xyz + gripper (the gantry)
    # course flags (lessons 17-19; validated in build, implemented in their lesson)
    force_in: str = "none"
    moe: str = "top1_capacity"
    n_experts: int = 4
    balance_loss: bool = False
    expert_inputs: str = "all"
    force_prompt: bool = False
    head: str = "position"
    force_noise: str = "off"  # eval-time only: the twin / FACT checks


def _check(value: str, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {allowed}")


def build(cfg: ModelConfig):
    """Validate the config and return the assembled Policy (the only model constructor).

    Flags whose lesson has not landed yet raise NotImplementedError here, so a config that
    silently ignores its own switches cannot exist.
    """
    from nfv.model.build import Policy  # local import: build.py needs ModelConfig

    _check(cfg.force_in, FORCE_IN, "force_in")
    _check(cfg.moe, MOE, "moe")
    _check(cfg.expert_inputs, EXPERT_INPUTS, "expert_inputs")
    _check(cfg.head, HEAD, "head")
    _check(cfg.force_noise, FORCE_NOISE, "force_noise")
    if cfg.force_in != "none":
        raise NotImplementedError(f"force_in={cfg.force_in!r} lands in lesson 17/18")
    if cfg.force_prompt:
        raise NotImplementedError("force_prompt lands in lesson 18")
    if cfg.head != "position":
        raise NotImplementedError(f"head={cfg.head!r} lands in lesson 18")
    if cfg.head == "position" and cfg.act_dim != 4:
        raise ValueError("the position head on the gantry is 4-D: dp xyz + gripper")
    return Policy(cfg)
