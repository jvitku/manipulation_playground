"""SmolVLA (Shukor et al. 2025, arXiv 2506.01844) on the course gantry - lesson 20's large VLA.

Runs in the lab image's SEPARATE venv ``/opt/vla`` (Python 3.12, torch 2.7.1, lerobot 0.6.1,
mujoco 3.3.0; docker/Dockerfile.lab) - never in the lab's own python. Weights come offline from
the bind-mounted HF cache (HF_HOME=/workspace/outputs/hf_cache, HF_HUB_OFFLINE=1):
``lerobot/smolvla_base`` (450 M parameters) and the SmolVLM2-500M VLM it is built on.

The model as loaded (counts measured in this repo, see lab/tasks/task_nfv5.py's result.json):
    vision encoder (SigLIP) + connector + 16 VLM text layers   350 M   bf16, frozen
    action expert: 16 layers, width 720 (cross-attention every
    other layer into the VLM's K/V)                              98 M   bf16 (+ 1.6 M fp32)
    new-embodiment projections: state_proj [32 -> 960],
    action_in_proj [32 -> 720], action_out_proj [720 -> 32],
    action_time_mlp_in/out (time + action fusion)                1.6 M  fp32
The VLM reads the two images, the instruction and ONE state token (the prefix); the expert
denoises the action chunk (the suffix) by flow matching, attending to the prefix's K/V.

Adapting it to the course (the "new embodiment"):
    cameras   rgb_static -> observation.images.camera1, rgb_wrist -> camera2: uint8 [96, 96, 3]
              -> float [B, 3, 96, 96] in [0, 1]; SmolVLA's own prepare_images upsamples to
              512 x 512 (resize_with_pad, square -> no padding) and maps to [-1, 1] for SigLIP:
              64 image tokens per camera after the connector's pixel shuffle.
    text      the instruction + "\\n" (lerobot's NewLineTaskProcessorStep), SmolVLM's GPT-2
              tokenizer, right-padded to 48 tokens (smolvla_base's pad_language_to=max_length).
    state     the env's 8-d state [x, y, z, yaw, grip, vx, vy, vz], z-scored with the demo set's
              mean/std (SmolVLA's MEAN_STD mapping), zero-padded to max_state_dim = 32.
    action    dp xyz (m/step) + gripper: 4 dims, z-scored the same way, padded to max_action_dim
              = 32; the loss masks the padded dims away (lerobot's forward slices them off).
    chunk     chunk_size 16, n_action_steps 8 - the course standard (smolvla_base was trained
              with 50/50). Nothing in the weights depends on the chunk length: the suffix tokens
              carry no learned positional table (RoPE positions + sinusoidal flow time), so the
              base loads unchanged and the expert simply sees shorter chunks.
Normalisation (our own, not lerobot's processor pipeline - the base's stats belong to the
SO-100 arm it was pretrained on): ``norm_stats(ds)`` over the demo set's steps; std < 1e-8
(e.g. a constant yaw) is replaced by 1.

Three fine-tuning modes + one memory probe (``setup_mode``):
    head         only the new-embodiment projections train (the linear-probe baseline)
    lora         head + LoRA (nfv.lora, r / alpha) on the attention q/k/v/o projections of
                 BOTH the 16 VLM text layers and the 16 action-expert layers
    expert_full  lerobot's standard SmolVLA fine-tune (train_expert_only): VLM frozen, the whole
                 action expert + the projections train - the reference
    full         every parameter trainable (vision encoder included) - only MEASURED (does it
                 fit in 10 GB at the same batch?), never trained to the end

Precision (a lesson point: 450 M parameters is where it starts to matter). smolvla_base loads
the VLM AND the action expert in bf16 storage; trained as-is, AdamW would keep bf16 weights and
bf16 moments, and an update smaller than half a bf16 ulp of its weight (8-bit mantissa: ~0.4 %
of the weight's magnitude) rounds to nothing. So every TRAINABLE tensor is upcast to an fp32
master copy here (frozen weights stay bf16 - they are never updated), and the forward/backward
run under ``torch.autocast(bfloat16)`` on the RTX 3080 (Ampere has bf16 tensor cores; bf16 needs
no loss scaling, unlike fp16). The nano models of lessons 15-19 stayed plain fp32 - at ~5 M
parameters there was nothing to save. The expert_full arm measures how much of one peak-LR
AdamW step would have been lost to bf16 storage (``bf16_lost_update_frac``).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np
import torch

from nfv import lora as L

BASE = "lerobot/smolvla_base"
CHUNK, N_EXEC = 16, 8  # course standard (lessons 16-19)
STATE_DIM, ACTION_DIM = 8, 4  # env state [8]; dp xyz + gripper
CAM_KEYS = ("observation.images.camera1", "observation.images.camera2")  # static, wrist
TOK_KEY, MASK_KEY = "observation.language.tokens", "observation.language.attention_mask"
MODES = ("head", "lora", "expert_full", "full")
TRAIN_MODES = ("head", "lora", "expert_full")

_P = r"model\.vlm_with_expert\."
# new-embodiment projections: state in, action in/out, the time-action fusion MLP
HEAD_PATTERNS = (
    r"model\.state_proj\.(weight|bias)",
    r"model\.action_in_proj\.(weight|bias)",
    r"model\.action_out_proj\.(weight|bias)",
    r"model\.action_time_mlp_(in|out)\.(weight|bias)",
)
# LoRA targets: attention q/k/v/o of the 16 VLM text layers and of the 16 action-expert layers
LORA_TARGETS = (
    _P + r"vlm\.model\.text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
    _P + r"lm_expert\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
)
# parameter groups for trainable_report (first match wins)
GROUPS = {
    "lora": r"\.lora_[AB]$",
    "vision": r"\.vision_model\.",
    "connector": r"\.connector\.",
    "vlm_text": r"\.vlm\.",
    "expert": r"\.lm_expert\.",
    "projections": r"^model\.(state_proj|action_)",
}
# representative expert layers both B (LoRA) and C (expert_full) adapt: a self-attention q
# (720 -> 960), a cross-attention v (VLM K/V 320 -> 320) and a late o (960 -> 720)
SPECTRUM_LAYERS = (
    "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj",
    "model.vlm_with_expert.lm_expert.layers.7.self_attn.v_proj",
    "model.vlm_with_expert.lm_expert.layers.14.self_attn.o_proj",
)
# per-mode peak learning rates (pre-registered in outputs/lora_vla/PREREG.txt): expert_full =
# SmolVLA's own preset; LoRA a common LoRA default (B starts at 0, so it needs more than the
# preset); head = linear-probe style (few parameters, larger steps)
MODE_LR = {"head": 1e-3, "lora": 2e-4, "expert_full": 1e-4, "full": 1e-4}
BETAS, EPS, WD, CLIP = (0.9, 0.95), 1e-8, 1e-10, 10.0  # SmolVLA's optimizer preset
DECAY_FRAC = 0.025  # cosine ends at 2.5 % of the peak (SmolVLA: 1e-4 -> 2.5e-6)


# ------------------------------------------------------------------------------- the model
def load_policy(device: str = "cuda", chunk: int = CHUNK, n_exec: int = N_EXEC):
    """smolvla_base, re-configured for the gantry (2 cameras, 8-d state, 4-d action, chunk 16).

    Offline: needs HF_HOME pointing at the cache and HF_HUB_OFFLINE=1.
    """
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)) for k in CAM_KEYS},
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    cfg.chunk_size, cfg.n_action_steps = int(chunk), int(n_exec)
    cfg.empty_cameras = 0
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(BASE, config=cfg)
    return policy.to(device)


def setup_mode(policy, mode: str, r: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> dict:
    """Make exactly the mode's parameters trainable (fp32 master copies); returns the report.

    {"mode", "lora": inject_lora report | None, "trainable": trainable_report, "head": [names]}
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    lora_rep = None
    if mode == "head":
        L.freeze_all(policy)
        head = L.set_trainable(policy, HEAD_PATTERNS)
    elif mode == "lora":
        lora_rep = L.inject_lora(policy, LORA_TARGETS, r=r, alpha=alpha, dropout=dropout)
        head = L.set_trainable(policy, HEAD_PATTERNS)
    elif mode == "expert_full":  # exactly lerobot's as-loaded requires_grad set
        L.freeze_all(policy)
        head = L.set_trainable(policy, HEAD_PATTERNS)
        L.set_trainable(policy, [_P + r"lm_expert\..*"])
    else:  # full
        for p in policy.parameters():
            p.requires_grad_(True)
        head = L.set_trainable(policy, HEAD_PATTERNS)
    for p in policy.parameters():  # fp32 master weights for everything that will be updated
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return {"mode": mode, "lora": lora_rep, "head": head,
            "trainable": L.trainable_report(policy, GROUPS)}  # fmt: skip


# --------------------------------------------------------------------------------- the data
def norm_stats(ds) -> dict:
    """Mean/std of state [8] and action [4] (dp xyz + gripper) over every demo step."""
    st = np.concatenate([e["state"] for e in ds.episodes]).astype(np.float64)
    ac = np.concatenate(
        [np.concatenate([e["dp"], e["gripper"][:, None]], axis=1) for e in ds.episodes]
    ).astype(np.float64)

    def ms(x):
        s = x.std(0)
        return x.mean(0).tolist(), np.where(s < 1e-8, 1.0, s).tolist()

    sm, ss = ms(st)
    am, as_ = ms(ac)
    return {"state_mean": sm, "state_std": ss, "action_mean": am, "action_std": as_}


class Codec:
    """Course arrays <-> SmolVLA batch tensors (z-scoring, tokens, camera keys)."""

    def __init__(self, policy, stats: dict, device: str):
        self.device = device
        self.tok = policy.model.vlm_with_expert.processor.tokenizer
        self.max_len = int(policy.config.tokenizer_max_length)
        f = lambda k: torch.tensor(stats[k], dtype=torch.float32, device=device)  # noqa: E731
        self.s_mu, self.s_sd = f("state_mean"), f("state_std")
        self.a_mu, self.a_sd = f("action_mean"), f("action_std")
        self._cache: dict[str, tuple] = {}

    def tokens(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 48] int64 ids, [B, 48] bool mask (instruction + '\\n', right-padded)."""
        ids, masks = [], []
        for t in texts:
            if t not in self._cache:
                enc = self.tok(t + "\n", padding="max_length", max_length=self.max_len,
                               truncation=True, padding_side="right",
                               return_tensors="pt")  # fmt: skip
                self._cache[t] = (enc["input_ids"][0], enc["attention_mask"][0].bool())
            i, m = self._cache[t]
            ids.append(i)
            masks.append(m)
        return torch.stack(ids).to(self.device), torch.stack(masks).to(self.device)

    def obs_batch(self, rgb_s, rgb_w, state, texts) -> dict:
        """uint8 [B, 96, 96, 3] x2, state [B, 8], texts -> the policy's observation dict."""

        def img(u8):
            t = torch.as_tensor(np.ascontiguousarray(u8), device=self.device)
            return t.permute(0, 3, 1, 2).float() / 255.0  # [B, 3, 96, 96] in [0, 1]

        st = torch.as_tensor(np.asarray(state, np.float32), device=self.device)
        tok, mask = self.tokens(list(texts))
        return {CAM_KEYS[0]: img(rgb_s), CAM_KEYS[1]: img(rgb_w),
                "observation.state": (st - self.s_mu) / self.s_sd,
                TOK_KEY: tok, MASK_KEY: mask}  # fmt: skip

    def encode_actions(self, a: np.ndarray) -> torch.Tensor:
        """[B, H, 4] env units -> z-scored tensor."""
        t = torch.as_tensor(np.asarray(a, np.float32), device=self.device)
        return (t - self.a_mu) / self.a_sd

    def decode_actions(self, z: torch.Tensor) -> np.ndarray:
        """[..., >= 4] z-scored (padded) -> [..., 4] env units (float64 numpy)."""
        a = z[..., :ACTION_DIM].float() * self.a_sd + self.a_mu
        return a.cpu().numpy().astype(np.float64)


@dataclass
class Sampler:
    """Uniform chunk starts over every demo step (DemoDataset.sample_batch's rule) that also
    returns each sample's instruction - SmolVLA reads the text; the nano sampler did not."""

    ds: object  # nfv.data.dataset.DemoDataset (or a first-n view with .episodes/.index)
    chunk: int = CHUNK

    def __post_init__(self):
        self.lengths = np.array([e["dp"].shape[0] for e in self.ds.episodes])
        self.starts = np.array([(i, t) for i, n in enumerate(self.lengths) for t in range(n)],
                               dtype=np.int64)  # fmt: skip
        self.instr = [ep["instruction"] for ep in self.ds.index["episodes"][: len(self.lengths)]]

    def sample(self, b: int, rng: np.random.Generator) -> dict:
        picks = self.starts[rng.integers(0, len(self.starts), size=b)]
        H = self.chunk
        out = {"rgb_static": [], "rgb_wrist": [], "state": [], "action": [], "pad": [], "text": []}
        for i, t in picks:
            ep, n = self.ds.episodes[i], int(self.lengths[i])
            out["rgb_static"].append(ep["rgb_static"][t])
            out["rgb_wrist"].append(ep["rgb_wrist"][t])
            out["state"].append(ep["state"][t])
            v = min(H, n - t)
            a = np.zeros((H, ACTION_DIM), np.float32)
            a[:v, :3] = ep["dp"][t : t + v]
            a[:v, 3] = ep["gripper"][t : t + v]
            out["action"].append(a)
            m = np.ones(H, bool)
            m[:v] = False  # True = padded (lerobot's action_is_pad)
            out["pad"].append(m)
            out["text"].append(self.instr[i])
        return {k: (v if k == "text" else np.stack(v)) for k, v in out.items()}


def train_batch(codec: Codec, raw: dict) -> dict:
    b = codec.obs_batch(raw["rgb_static"], raw["rgb_wrist"], raw["state"], raw["text"])
    b["action"] = codec.encode_actions(raw["action"])
    b["action_is_pad"] = torch.as_tensor(raw["pad"], device=codec.device)
    return b


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine from peak to DECAY_FRAC x peak at the last step."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, steps - warmup)
    lo = peak * DECAY_FRAC
    return lo + 0.5 * (peak - lo) * (1 + math.cos(math.pi * min(1.0, p)))


def make_optimizer(policy, lr: float):
    params = [p for p in policy.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=BETAS, eps=EPS, weight_decay=WD), params


def train_step(policy, opt, params, batch, lr: float) -> float:
    for g in opt.param_groups:
        g["lr"] = lr
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach())


# --------------------------------------------------------------------------- measurements
@torch.no_grad()
def bf16_lost_fraction(before: dict[str, torch.Tensor], policy) -> float:
    """Share of the elements whose fp32 update since ``before`` would vanish in bf16 storage:
    (bf16(w_old) + Δ) rounded to bf16 == bf16(w_old), over the tensors in ``before``."""
    lost = tot = 0
    params = dict(policy.named_parameters())
    for n, w0 in before.items():
        w1 = params[n].detach()
        d = w1 - w0.to(w1.device)
        moved = d != 0
        b0 = w0.to(w1.device).bfloat16()
        same = (b0.float() + d).bfloat16() == b0
        lost += int((same & moved).sum())
        tot += int(moved.sum())
    return lost / max(1, tot)


@torch.no_grad()
def velocity(policy, batch: dict, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The flow-matching velocity v_t [B, H, 32] for fixed noise / time (fp32, no autocast) -
    the output the merge check compares."""
    out = {}
    h = policy.model.action_out_proj.register_forward_hook(lambda m, i, o: out.update(v=o))
    try:
        policy.forward(batch, noise=noise, time=t)
    finally:
        h.remove()
    return out["v"].float()


def spectrum(dw: torch.Tensor, r: int) -> dict:
    s = L.singular_values(dw)
    return {"shape": list(dw.shape), "sv": [round(float(x), 6) for x in s[:64]],
            "n_sv": int(s.numel()), "fro2": float((s.double() ** 2).sum()),
            "energy_top_r": L.energy_top_k(s, r),
            "energy_top_16": L.energy_top_k(s, 16)}  # fmt: skip


# ------------------------------------------------------------------------------- closed loop
class SmolVLAActor:
    """``act(obs, instruction)`` -> a [16, 4] chunk in env units (dp m/step, gripper 0..1).

    One prefix pass (VLM over images + text + state, K/V cached) and ``num_steps`` = 10 Euler
    steps of the expert per chunk, under bf16 autocast; the flow noise comes from the caller's
    generator, so a rollout is seed-deterministic.
    """

    def __init__(self, policy, codec: Codec):
        self.policy = policy.eval()
        self.codec = codec
        self.n_exec = int(policy.config.n_action_steps)
        self.chunk = int(policy.config.chunk_size)
        self.calls, self.time_s = 0, 0.0

    @torch.no_grad()
    def act(self, obs, instruction: str, generator: torch.Generator | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        b = self.codec.obs_batch(obs.rgb_static[None], obs.rgb_wrist[None], obs.state[None],
                                 [instruction])  # fmt: skip
        pol = self.policy
        images, img_masks = pol.prepare_images(b)
        state = pol.prepare_state(b)
        noise = torch.randn((1, self.chunk, pol.config.max_action_dim), generator=generator,
                            device=self.codec.device)  # fmt: skip
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = pol.model.sample_actions(images, img_masks, b[TOK_KEY], b[MASK_KEY], state,
                                         noise=noise)  # fmt: skip
        a = self.codec.decode_actions(z[0])  # [16, 4]
        a[:, 3] = np.clip(a[:, 3], 0.0, 1.0)
        self.calls += 1
        self.time_s += time.perf_counter() - t0
        return a


def adapter_param_names(policy) -> list[str]:
    return [n for n, p in policy.named_parameters() if p.requires_grad]


def full_ckpt_bytes(policy) -> int:
    """What a full checkpoint of the model would weigh as stored now (computed, not written)."""
    return L.state_bytes(p for _, p in policy.named_parameters())


def lora_spectra(policy, layers=SPECTRUM_LAYERS) -> dict:
    mods = L.lora_layers(policy)
    out = {}
    for n in layers:
        if n in mods:
            out[n] = spectrum(mods[n].delta_weight(), mods[n].r)
    return out


def match(names, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [n for n in names if rx.fullmatch(n)]
