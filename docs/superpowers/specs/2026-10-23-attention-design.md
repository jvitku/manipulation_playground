# Attention school — four new Lab tasks (attention → transformer → cross-attention → find-the-digit)

Date: 2026-10-23. Requested in chat by the owner (jvitku); direction approved ("OK perfect"
to the trio, fourth task added in the same message).

## Intent

Four new lessons in the existing Lab (`lab/`, served at :8430) that teach the attention
mechanism and the transformer from the inside, as a bridge toward the TA-VLA lessons that
already use block attention. The owner's explicit requirements, verbatim in spirit:

- minimal architectures whose **inner wiring is fully drawn** — detailed SVG schematics of
  how the thing works inside, with every tensor shape labeled, the loss marked where it is
  computed, and the math in KaTeX (this applies to ALL FOUR tasks);
- **visualize the Q/K/V value vectors** (vecgrid) and how attention heads interact;
- watch **how training progresses** (the attention pattern forming live) and **how
  inference flows** step by step;
- for cross-attention specifically: what it is, how it is wired, how inference flows, how
  training works, and **where it is used and where it makes sense**;
- each lesson carries **tiny collapsible code snippets** showing the main concepts as real
  implementation — at least the training loop, the inference loop, and the core attention
  computation — syntax-highlighted, collapsed by default (added by the owner 2026-10-23).

## Goals — the four tasks

All four follow every existing Lab convention: `TaskSpec` registration with curated params,
calibrated runtime estimates, `RunContext` job contract, checkpoints + parameterized eval,
lesson HTML fragment with KaTeX + inline SVG schematic, vecgrid with the per-task toggle,
charts via `charts.js`. They form a visible sequence in the ladder.

### 1. `attention` — the mechanism alone (ladder order 6)

The purest setting: **key–value lookup**, the task that IS what Q/K/V means. A sequence of
(key, value) token pairs plus one query token; the model must output the value belonging to
the queried key. **Each pair is one fused input token** (key embedding + value embedding
summed), and the answer is read out at the query token's position — with a single attention
layer and no MLP, two separate tokens per pair would need an induction-style second layer
and the task would not be learnable (controller ruling). Architecture: token embeddings +
ONE self-attention layer (default 2 heads, tiny dims, d_k ~8–16) + linear readout;
cross-entropy loss on the predicted value token. No MLP block, no stacking — the lesson is
the attention computation itself.

- Train knobs: number of heads (1/2/4), d_k, number of pairs in the sequence, steps, lr, seed.
- Visualizations: vecgrid of every token's Q, K and V vectors per head; the QKᵀ score matrix
  before and after softmax as heatmaps per head; the attended-value mixture. **During
  training** the attention heatmap updates live, sharpening from uniform to a crisp pointer.
- Eval: an inference stepper on a concrete example — walk token by token through embed →
  Q/K/V projection → scores → softmax → weighted sum → readout, each stage visualized.
- Runtime: seconds to ~1 min.

### 2. `vit-mnist` — a real transformer trained end to end (ladder order 7)

Mini Vision Transformer on MNIST: 28×28 image cut into 7×7 patches → 16 patch tokens +
learned position embeddings + a CLS token; 2 standard transformer blocks (MHA + MLP,
pre-norm); CLS readout → 10 classes. Chosen over a text toy because image-patches-as-tokens
is exactly the input half of the VLA models the Lab builds toward.

- Train knobs: patch size (7/14), depth (1/2), heads (2/4), model dim, lr, steps,
  train-subset size, seed. Charts: train loss, val accuracy.
- Visualizations: the CLS token's attention overlaid on the actual digit image, per layer
  and head, updating live during training (noise → "looks at the stroke"); vecgrid of patch
  embeddings and the CLS token.
- Eval: pick test digits (or draw an index), show prediction + confidence + attention
  overlays per layer/head.
- Runtime: ~1–3 min default; the schematic is the full transformer block, shapes labeled.

### 3. `cross-attn` — encoder–decoder cross-attention (ladder order 8)

Toy date-format translation, char-level: varied source formats ("23 Oct 2026",
"Oct 23, 2026", "23/10/2026") → ISO "2026-10-23". Encoder: 1 self-attention block.
Decoder: 1 block of masked self-attention + cross-attention (Q from the decoder, K/V from
the encoder — the wiring is the lesson) + readout. Teacher forcing in training; greedy
decode at inference with a step-through.

- Visualizations: the **cross-attention alignment matrix** (output chars × source chars —
  which source character each output character reads), plus encoder self-attention and the
  decoder's masked self-attention; vecgrids of Q (decoder side) vs K/V (encoder side).
- Lesson covers where cross-attention lives in the wild: seq2seq/translation, Perceiver-style
  latent queries, VLMs, and **ACT from this repo's own Stage-1 pipeline** (action queries
  cross-attending into observation features), and its relation to TAVLAMini's block attention.
- Runtime: tens of seconds to ~2 min.

### 4. `find-digit` — a miniature VLA: query-conditioned visual attention (ladder order 9)

Input: a 28×56 canvas with two random MNIST digits side by side; a **query** naming a digit
class 0–9. The model answers **left / right / not present** (the owner's "pick the correct
one or declare there's no number"; left/right/absent is the builder's framing, approved as
the controller's call). Architecture: the task-2 ViT encoder over the canvas (4×8 = 32 patch
tokens) + a learned query-class embedding token + ONE cross-attention readout (query attends
over patch tokens) + 3-way head. Roughly ⅓ of training samples have the queried digit absent.
The whole model **trains jointly from scratch** — no dependency on a task-2 checkpoint (that
is what fits the stated runtime; controller ruling). The two digits on the canvas are always
of **distinct classes**, so left/right/absent is never ambiguous (controller ruling).

- Visualizations: the query's attention overlaid on the canvas — it should light up on the
  queried digit — per head, live during training and at eval; vecgrid of the query embedding
  next to patch K vectors.
- Eval knobs: choose the two digits and the query, absent cases, optional pixel noise.
- Lesson: this is the cross-attention pattern of a VLA in miniature — image tokens +
  command token → grounded answer.
- Runtime: ~1–4 min.

## Constraints

- **MNIST without torchvision**: fvb-train ships torch 2.5.1 only. Bake the 4 raw MNIST IDX
  files (~11 MB, ossci-datasets S3 mirror) into the `fvb-lab` image at build; parse with a
  small numpy IDX reader in lab code. No runtime network access, no torchvision.
- **One shared mini-transformer module** (e.g. `lab/tasks/minitf.py`): a single readable
  implementation of multi-head attention, the transformer block, and attention-weight
  capture, used by all four tasks — so the schematics describe one real implementation.
  Plain torch; no `nn.Transformer`, the code must be inspectable and match the drawings.
- **Live attention during training**: training jobs periodically emit a small (downsampled
  if needed) attention snapshot the frontend renders live, same polling model as metrics.
  Mechanism (metrics.jsonl side keys vs. a snapshot file in the run dir) is the builder's.
- **Quality targets for the default configs** (tune until met; the measured numbers then go
  into the lessons per the grounding rule): lookup accuracy ≥ 99 %; ViT-MNIST val accuracy
  ≥ 90 %; cross-attn exact-match ≥ 95 % on held-out dates with a visibly structured
  alignment matrix; find-digit accuracy ≥ 90 % with the attention overlay landing on the
  queried digit in correct cases.
- **Code snippets in lessons**: collapsed-by-default `<details>` blocks, syntax-highlighted
  with a small vendored highlighter (e.g. highlight.js core + python, added to
  `lab/web/vendor/` + VENDOR.md; no CDN at runtime). Snippets are **trimmed excerpts of the
  real implementation** (`minitf.py`, the task files, `lab/core/loop.py`), never free-floating
  pseudo-code, and a test guards against drift between snippet and source (mechanism is the
  builder's). The four new lessons get them in this wave; **retrofitting the nine existing
  lessons with the same snippet blocks is in scope as the final task** (owner said "for each
  experiment").
- Ladder renumbering: existing ta-vla/td3/td3-bc/residual-rl shift from orders 6/7/8/9 to
  10/11/12/13. Lessons, README tables and any docs that cite the old ladder numbers are
  updated as part of the renumbering (controller ruling: that is renumbering, not a change
  to those tasks).
- Lesson claims stay grounded: these four lessons teach textbook mechanisms, so grounding
  means correct math and shapes consistent with `minitf.py`, plus repo cross-references
  (ACT, TAVLAMini) that match the actual code in `src/fvb`.
- Branch `attention`; same review/testing bar as the first Lab wave (tests in `tests/lab`,
  run in the fvb-lab image).

## Non-goals / anti-goals

- No text-corpus language modeling, no tokenizers beyond char/toy vocabularies.
- Not a general interpretability tool: visualizations are bespoke per task, not a probe UI.
- Anti-goal: schematics that abstract the mechanism away (a box labeled "attention" is a
  failure — the drawing shows Q/K/V projections, the score matrix, softmax, the mixture).
- Anti-goal: anything slow enough to break the "watch it train" loop — defaults stay in the
  minutes range.

## Left to the builder

Exact lesson texts and schematic layouts; snapshot transport mechanism and cadence; exact
tiny-model dims; date-format grammar; MNIST subset sizes and the absent-case sampling; how
much of task 2's encoder code task 4 reuses; calibration methodology (same as wave 1).
