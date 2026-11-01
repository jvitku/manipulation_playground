"""LoRA from scratch: a low-rank adapter for a frozen pretrained network (lesson 20).

Pure torch (no peft): the whole idea fits in one small module, and every line here is the
lesson. Used by ``nfv.vla_smolvla`` to fine-tune SmolVLA (~450 M parameters) on the course's
gantry tasks, and tested in ``tests/nfv/test_nfv_lora.py`` in the lab's own python.

The question
------------
Fine-tuning moves every pretrained weight matrix W to W + ΔW. For a 450 M-parameter VLA that
means 450 M trainable numbers, their gradients and two AdamW moments each - about 7 GB before a
single activation is stored. LoRA (Hu et al. 2021, arXiv 2106.09685) bets that the UPDATE needed
to adapt a pretrained model to a new task has low intrinsic rank (Aghajanyan et al. 2020, arXiv
2012.13255: fine-tuning succeeds inside a random subspace of a few hundred to a few thousand
dimensions), and therefore parameterises it as the product of two thin matrices:

    ΔW = (alpha / r) · B @ A            A [r, in],  B [out, r],  r << min(in, out)

    y  = W x + b + (alpha / r) · B (A x)            W, b frozen; only A and B train

Shape legend (one wrapped nn.Linear):
    x   [..., in]       input activations (any leading dims: batch, tokens, ...)
    W   [out, in]       the frozen pretrained weight, b [out] its frozen bias
    A   [r, in]         "down" projection: in -> r
    B   [out, r]        "up" projection:   r -> out
    y   [..., out]

Parameters per layer: r · (in + out) instead of in · out. For SmolVLA's VLM attention
(in = out = 960) at r = 16 that is 30 720 instead of 921 600 - 3.3 % of the layer.

Initialisation (the "bit-identical at step 0" property):
    A ~ kaiming-uniform (exactly what nn.Linear does to its own weight)
    B = 0          =>  ΔW = 0  =>  the wrapped layer computes EXACTLY the base output
Training therefore starts at the pretrained model, not near it. The gradients at step 0 show
why both halves are needed: with g = dL/dy [..., out],
    dL/dB = (alpha/r) · g^T (A x)     non-zero (A is random), so B moves first
    dL/dA = (alpha/r) · B^T g x^T     = 0 while B = 0; A starts learning once B has moved
(initialising both to zero would leave the adapter stuck at zero forever).

Why the alpha / r factor: with Adam each element of B moves by ~lr per step whatever r is, so
the size of ΔW = B @ A grows with r. Dividing by r (and choosing alpha once, typically 2r) keeps
the effective step size of ΔW roughly constant when you sweep r - the lesson's r = 4/8/16/32
knob changes capacity, not the learning rate.

Merging (inference at zero extra cost): after training, W' = W + (alpha/r) · B @ A is a plain
[out, in] matrix, so ``merge()`` returns an ordinary nn.Linear and the adapter disappears - the
merged network has exactly the base architecture and latency. In exact arithmetic the merge is
exact; in floating point it differs by rounding only (``tests/nfv/test_nfv_lora.py`` pins
< 1e-5 in fp32). Merged into a bf16 weight it rounds to bf16's 8-bit mantissa - one reason the
adapter checkpoint keeps A and B in fp32.

What this module provides
-------------------------
    LoRALinear(base, r, alpha, dropout)        the wrapper; .merge() -> nn.Linear
    inject_lora(model, targets, r, alpha, ...)  wrap every nn.Linear whose qualified name
                                                fully matches a regex; freeze the rest
    set_trainable(model, patterns)              re-enable chosen parameters (task heads)
    trainable_report(model, groups)             total / trainable / per-group counts
    trainable_state_dict(model)                 ONLY the trainable tensors (the adapter file)
    merge_lora(model)                           replace every LoRALinear by its merge, in place
    singular_values(dw), energy_top_k(s, k)     the spectrum of an update (is ΔW low-rank?)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """``base(x) + (alpha / r) · B(A(dropout(x)))`` with ``base`` frozen; see the module doc.

    The wrapper keeps the host code's view of the layer intact: ``.weight`` / ``.bias`` /
    ``.in_features`` / ``.out_features`` read through to the frozen base. That matters for real
    models - SmolVLA's attention code does ``x.to(layer.self_attn.q_proj.weight.dtype)`` before
    every projection, so a wrapper without ``.weight`` would crash the host at the first token.

    A and B live in fp32 (or ``adapter_dtype``) whatever the base dtype is: a bf16 base is fine,
    the adapter path upcasts its input and casts its contribution back to the base output's
    dtype (under ``torch.autocast`` the matmuls run in the autocast dtype anyway).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        adapter_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if int(r) < 1:
            raise ValueError(f"rank r must be >= 1, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # W [out, in] and b [out] stay pretrained
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r  # the alpha / r factor
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=dev,
                                               dtype=adapter_dtype))  # [r, in]  # fmt: skip
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=dev,
                                               dtype=adapter_dtype))  # [out, r] # fmt: skip
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # nn.Linear's own weight init
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # --- the host code's view of the layer (read through to the frozen base) ---
    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def n_adapter_params(self) -> int:
        """r · (in + out): the trainable numbers this layer adds."""
        return self.r * (self.in_features + self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)  # [..., out]   W x + b, frozen
        h = self.dropout(x).to(self.lora_A.dtype)  # [..., in]
        d = F.linear(F.linear(h, self.lora_A), self.lora_B)  # [..., r] -> [..., out]
        return y + (self.scale * d).to(y.dtype)  # B = 0 at init: exactly y

    def delta_weight(self) -> Tensor:
        """ΔW = (alpha / r) · B @ A  [out, in], in the adapter dtype; rank <= r."""
        return self.scale * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """A plain nn.Linear computing W + ΔW (the base's dtype/device): zero extra latency."""
        b = self.base
        m = nn.Linear(b.in_features, b.out_features, bias=b.bias is not None,
                      device=b.weight.device, dtype=b.weight.dtype)  # fmt: skip
        w = b.weight.to(self.lora_A.dtype) + self.delta_weight()  # [out, in], add in fp32
        m.weight.copy_(w.to(b.weight.dtype))
        if b.bias is not None:
            m.bias.copy_(b.bias)
        m.requires_grad_(False)
        return m

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha:g}, scale={self.scale:g}"


# ------------------------------------------------------------------------------ model surgery
def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """(parent module, attribute name) of the submodule at qualified ``name``."""
    head, _, leaf = name.rpartition(".")
    return (model.get_submodule(head) if head else model), leaf


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def set_trainable(model: nn.Module, patterns: Iterable[str]) -> list[str]:
    """Re-enable grads on every parameter whose qualified name fully matches a regex.

    Returns the parameter names it switched on. Raises if a pattern matches nothing - a typo in
    a regex must not silently leave a head frozen (the same guard lerobot puts on its own
    freezing patterns).
    """
    pats = [re.compile(p) for p in patterns]
    hit: list[str] = []
    used = set()
    for name, p in model.named_parameters():
        for i, pat in enumerate(pats):
            if pat.fullmatch(name):
                p.requires_grad_(True)
                hit.append(name)
                used.add(i)
                break
    missing = [pats[i].pattern for i in range(len(pats)) if i not in used]
    if missing:
        raise ValueError(f"set_trainable: patterns matched no parameter: {missing}")
    return hit


def inject_lora(
    model: nn.Module,
    target_patterns: Iterable[str],
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    freeze: bool = True,
) -> dict:
    """Wrap every nn.Linear whose qualified module name fully matches one of the regexes.

    ``freeze=True`` first freezes EVERY parameter of the model, so afterwards exactly the
    LoRA A/B tensors are trainable (add task heads back with :func:`set_trainable`). Modules
    already inside a LoRALinear (its ``.base``) are never wrapped twice. Raises ValueError when
    a pattern matches no nn.Linear.

    Returns the report: {"targets": [module names], "r", "alpha", "scale", "dropout",
    "n_layers", "lora_params", "per_layer": {name: {"in", "out", "params"}}}.
    """
    pats = [re.compile(p) for p in target_patterns]
    if freeze:
        freeze_all(model)
    wrapped_roots = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    names: list[str] = []
    used = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear):
            continue
        if any(name == w or name.startswith(w + ".") for w in wrapped_roots):
            continue  # the .base of an existing adapter
        for i, pat in enumerate(pats):
            if pat.fullmatch(name):
                names.append(name)
                used.add(i)
                break
    missing = [pats[i].pattern for i in range(len(pats)) if i not in used]
    if missing:
        raise ValueError(f"inject_lora: patterns matched no nn.Linear: {missing}")
    per_layer = {}
    for name in names:  # collected first: never mutate the module tree while iterating it
        parent, leaf = _parent(model, name)
        lora = LoRALinear(getattr(parent, leaf), r=r, alpha=alpha, dropout=dropout)
        setattr(parent, leaf, lora)
        per_layer[name] = {"in": lora.in_features, "out": lora.out_features,
                           "params": lora.n_adapter_params()}  # fmt: skip
    return {
        "targets": names, "r": int(r), "alpha": float(alpha), "scale": float(alpha) / int(r),
        "dropout": float(dropout), "n_layers": len(names),
        "lora_params": int(sum(v["params"] for v in per_layer.values())), "per_layer": per_layer,
    }  # fmt: skip


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear in ``model`` by its merged nn.Linear, in place; returns count."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        parent, leaf = _parent(model, name)
        setattr(parent, leaf, getattr(parent, leaf).merge())
    return len(names)


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


# -------------------------------------------------------------------------------- accounting
def trainable_report(model: nn.Module, groups: dict[str, str] | None = None) -> dict:
    """Parameter counts: {"total", "trainable", "trainable_frac", "groups": {label: {"total",
    "trainable"}}}. ``groups`` = ordered {label: regex searched in the parameter name}; each
    parameter counts in the FIRST matching group, else in "other". Shared (tied) parameters are
    counted once, like ``sum(p.numel() for p in model.parameters())``.
    """
    pats = [(lbl, re.compile(rx)) for lbl, rx in (groups or {}).items()]
    out: dict[str, dict[str, int]] = {lbl: {"total": 0, "trainable": 0} for lbl, _ in pats}
    out["other"] = {"total": 0, "trainable": 0}
    total = trainable = 0
    for name, p in model.named_parameters():  # named_parameters() de-duplicates tied tensors
        n = p.numel()
        total += n
        trainable += n if p.requires_grad else 0
        lbl = next((lbl for lbl, pat in pats if pat.search(name)), "other")
        out[lbl]["total"] += n
        out[lbl]["trainable"] += n if p.requires_grad else 0
    if not out["other"]["total"]:
        del out["other"]
    return {"total": total, "trainable": trainable,
            "trainable_frac": trainable / max(1, total), "groups": out}  # fmt: skip


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the tensors that train (LoRA A/B + unfrozen heads) - the whole adapter file."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def state_bytes(tensors: Iterable[Tensor]) -> int:
    """Bytes the tensors occupy as stored (numel x element size)."""
    return int(sum(t.numel() * t.element_size() for t in tensors))


# ----------------------------------------------------------------------------- the spectrum
@torch.no_grad()
def singular_values(dw: Tensor) -> Tensor:
    """Singular values (descending, fp32, on CPU) of an update matrix ΔW [out, in]."""
    return torch.linalg.svdvals(dw.detach().float().cpu())


def energy_top_k(s: Tensor, k: int) -> float:
    """Share of the squared Frobenius norm ||ΔW||_F^2 = sum_i s_i^2 carried by the top-k
    singular values. A LoRA update has rank <= r, so energy_top_k(s, r) == 1 up to rounding; a
    full fine-tune's update is low-rank in the lesson's sense if this is large for small k."""
    e = s.double() ** 2
    tot = float(e.sum())
    return float(e[:k].sum()) / tot if tot > 0 else 0.0
