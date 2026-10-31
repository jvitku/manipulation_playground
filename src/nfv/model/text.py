"""The course vocabulary and the tiny language encoder (embedding + 1 transformer layer).

The vocabulary is word-level and tiny by design: 96x96 simulated scenes with six instructions
do not need web knowledge (plan §4). ``VOCAB`` holds every word of every task instruction in
``nfv/sim/tasks.py`` plus the subtask words (every entry of every task's SUBTASKS) reserved
for the lesson-18 force prompt, which appends the running subtask's name to the instruction.
A unit test re-derives both sets from the live tasks module, so VOCAB cannot drift from the
source silently.

Tokenizer: lowercase, split on whitespace, look up (unknown words -> <unk>), pad with <pad>
to ``max_text`` (12) tokens. The <pad> positions are masked out of every attention read
downstream (the encoder below, the trunk, the expert's cross-attention).

Shape legend: B = batch, T = max_text (12), D = d_model, V = vocabulary size.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nfv.model.trunk import EncoderBlock

MAX_TEXT = 12  # tokens per instruction (the longest course instruction is 4 words)

# fmt: off
VOCAB: tuple[str, ...] = (
    "<pad>", "<unk>",
    # instruction words (every INSTRUCTIONS entry of tasks T0-T3)
    "pick", "up", "the", "cube",            # t0: "pick up the cube"
    "press", "red", "blue", "button",       # t1: "press the red/blue button"
    "wipe", "plate", "gently", "firmly",    # t2: "wipe the plate gently/firmly"
    "insert", "peg",                        # t3: "insert the peg"
    # subtask words (every SUBTASKS entry; reserved for the lesson-18 force prompt)
    "approach", "done", "reach", "grasp", "lift", "descend", "search",
)
# fmt: on
PAD, UNK = 0, 1
WORD_ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE <= 64, "the plan caps the course vocabulary at 64 words"


def tokenize(text: str, max_text: int = MAX_TEXT) -> np.ndarray:
    """One instruction -> int64 ids [max_text]: lowercase, split, look up, pad with <pad>."""
    words = text.lower().split()
    ids = [WORD_ID.get(w, UNK) for w in words][:max_text]
    return np.array(ids + [PAD] * (max_text - len(ids)), dtype=np.int64)


def tokenize_batch(texts: list[str], max_text: int = MAX_TEXT) -> torch.Tensor:
    """A batch of instructions -> int64 ids [B, max_text]."""
    return torch.from_numpy(np.stack([tokenize(t, max_text) for t in texts]))


class TextEncoder(nn.Module):
    """Word embedding + learned positions + one pre-norm transformer layer -> text tokens."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, vocab: int, max_text: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)  # [V, D] table
        self.pos = nn.Parameter(torch.randn(max_text, d_model) * 0.02)  # [T, D]
        self.block = EncoderBlock(d_model, n_heads, ffn_hidden)

    def forward(self, ids: torch.Tensor):
        """ids int64 [B, T] -> (tokens [B, T, D], key_mask [B, T] bool, attn [B, NH, T, T])."""
        key_mask = ids != PAD  # [B, T], True on real words
        x = self.embed(ids) + self.pos  # [B, T, D]
        x, attn = self.block(x, mask=key_mask[:, None, None, :])  # pads hidden as keys
        return x, key_mask, attn
