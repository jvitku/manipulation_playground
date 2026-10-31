# Attention School Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four new Lab tasks teaching attention from the inside (key–value lookup, mini-ViT on MNIST, encoder–decoder cross-attention, query-conditioned digit finding), with live attention views during training, inference steppers, and collapsible real-code snippets in every lesson.

**Architecture:** One shared readable mini-transformer module (`lab/tasks/minitf.py`, plain torch, attention weights always returned) powers all four tasks; MNIST ships as raw IDX files baked into the fvb-lab image; a new generic "live panel" renders attention snapshots emitted through the existing `ctx.log` metric rows; snippets are excerpts of the real code guarded by a drift test.

**Tech Stack:** Existing Lab stack (FastAPI, vanilla JS, uPlot, KaTeX, vecgrid) + vendored highlight.js 11.9.0.

**Spec:** `docs/superpowers/specs/2026-10-23-attention-design.md`

## Global Constraints

- Branch `attention`; never push/merge to `main` (owner's decision per PR).
- Plain torch only: no `nn.Transformer`, no torchvision; MNIST from baked IDX files.
- Frontend assets vendored under `lab/web/vendor/` + entry in `VENDOR.md`; no CDN at runtime.
- Every Lab convention from `lab/README.md` "Adding a task" applies to each task (TaskSpec, no duplicate param names, self-describing checkpoints — evaluate rebuilds from ckpt `meta` only, calib.json entry, lesson grounding rule, viewer JSON fallback).
- Quality targets at default params: lookup ≥ 99 %, ViT val ≥ 90 %, date exact-match ≥ 95 %, find-digit ≥ 90 %. Tune defaults until met; measured numbers go into lessons.
- Lesson snippets are trimmed excerpts of real implementation files, `<details>`-collapsed, highlighted; a test guards snippet↔source drift.
- Tests run in the fvb-lab image: `docker run --rm -u 1001:1001 -v /home/jvitku/workspace/manipulation_playground:/workspace -w /workspace -e HOME=/tmp fvb-lab:latest pytest tests/lab/<file> -q` (rebuild image via `make lab-build` after Dockerfile.lab changes). Ruff via the dev image (canonical ruff version).
- Attention snapshot rows stay small: matrices downsampled to ≤ 16×16, values rounded to 3 decimals, emitted at most ~60 times per run.

## Review Focus

1. Lesson pages mixing snippets + KaTeX + SVG: highlight.js must not catch KaTeX output or trip the `.lesson svg:not(.katex *)` scoping — pinned by Task 8's static test (snippet blocks contain no `$$`/`\(`) and controller browser check.
2. Metrics file bloat from live attention — pinned by Task 4's row-size test (< 8 KB per row).
3. find-digit data generator: ~⅓ absent, never two digits of the same class — pinned by Task 7 generator test.
4. cross-attn inference must be genuinely autoregressive (no teacher forcing leak) — pinned by Task 6 decode test (works from BOS alone on unseen input).
5. Server boot without MNIST files (unset `LAB_MNIST_DIR` on a dev box): task modules import lazily and the registry loads — pinned by Task 3's import test.

---

### Task 1: `lab/tasks/minitf.py` — the shared mini-transformer

**Files:**
- Create: `lab/tasks/minitf.py`
- Test: `tests/lab/test_minitf.py`

**Interfaces (Produces — all later tasks rely on these exact names):**
- `class MHA(nn.Module)` — `__init__(self, d_model: int, n_heads: int)`; `forward(self, x_q, x_kv=None, mask=None) -> tuple[Tensor, Tensor]` returning `(out [B,Tq,d_model], attn [B,H,Tq,Tk])`. `x_kv=None` means self-attention; `mask` is a bool tensor broadcastable to `[Tq,Tk]`, True = keep.
- `class Block(nn.Module)` — pre-norm `MHA` + 2-layer GELU MLP (hidden `4*d_model`); `forward(self, x, mask=None) -> tuple[Tensor, Tensor]` → `(x, attn)`.
- `class CrossBlock(nn.Module)` — pre-norm masked self-attn + cross-attn + MLP; `forward(self, x, memory, self_mask=None) -> tuple[Tensor, Tensor, Tensor]` → `(x, self_attn, cross_attn)`.
- `causal_mask(t: int) -> Tensor` (bool `[t,t]`, lower-triangular True).
- `attn_snapshot(attn: Tensor, max_dim: int = 16) -> list` — mean over batch, per-head nested lists, adaptive-average-pooled to ≤ max_dim per side, rounded 3 decimals. Shape `[H][Tq'][Tk']`.
- Code style: this file is quoted in lessons — short, commented at the "what each projection is" level, no cleverness.

- [ ] **Step 1: Write the failing tests** — `test_mha_shapes` (self-attn out/attn shapes, attn rows sum to 1), `test_mha_cross_shapes` (Tq≠Tk), `test_causal_mask_blocks_future` (attn weight on future positions is 0 when mask applied), `test_block_and_crossblock_shapes`, `test_attn_snapshot_downsamples` (input `[1,2,32,32]` → `[2][16][16]`, values rounded), `test_mha_learns_lookup` (a 200-step overfit of one fused-token lookup batch reaches loss < 0.1 — proves gradients flow).
- [ ] **Step 2: Run tests, verify they fail** (module missing).
- [ ] **Step 3: Implement `lab/tasks/minitf.py`.**
- [ ] **Step 4: Run tests to green; ruff on dev image.**
- [ ] **Step 5: Commit** `feat(lab): minitf - shared readable mini-transformer with attention capture`.

### Task 2: MNIST baked into the image + loader

**Files:**
- Modify: `docker/Dockerfile.lab`
- Create: `lab/tasks/mnist_data.py`
- Test: `tests/lab/test_mnist_data.py`

**Interfaces (Produces):**
- `load_mnist(split: str) -> tuple[np.ndarray, np.ndarray]` — split `"train"|"test"`, returns `(images uint8 [N,28,28], labels uint8 [N])`. Reads gzipped IDX from `$LAB_MNIST_DIR` (default `/opt/mnist`); `FileNotFoundError` with a message naming the env var when absent. Parsed with numpy only. Module imports cleanly without the files (lazy read).

- [ ] **Step 1: Dockerfile.lab — download the 4 files** from `https://ossci-datasets.s3.amazonaws.com/mnist/` (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`) into `/opt/mnist/` via `ADD` or curl in one layer; `make lab-build` succeeds.
- [ ] **Step 2: Write failing tests** — `test_load_mnist_shapes` (train = 60000, test = 10000, images 28×28 uint8, labels 0–9), `test_missing_dir_raises` (monkeypatched `LAB_MNIST_DIR` → FileNotFoundError naming it), `test_import_without_files` (import with bogus env var set does not raise).
- [ ] **Step 3: Implement `mnist_data.py`; tests green in the rebuilt image.**
- [ ] **Step 4: Commit** `feat(lab): bake MNIST IDX files into fvb-lab + numpy loader`.

### Task 3: Ladder renumbering 6–9 → 10–13

**Files:**
- Modify: `lab/tasks/task5_tavla.py` (order 6→10), `lab/tasks/task6_td3.py` (7→11), `lab/tasks/task7_td3bc.py` (8→12), `lab/tasks/task8_residual.py` (9→13)
- Modify: every `lab/web/lessons/*.html` and `lab/README.md` line citing the old numbers of those four tasks ("task 6 (ta-vla)" → "task 10 (ta-vla)" etc.; grep for `task 6`–`task 9` and the README ladder table)
- Test: extend `tests/lab/test_spec.py`

- [ ] **Step 1: Failing test** — `test_ladder_orders_unique_and_expected`: the registry's orders are exactly `{1..13} minus {6,7,8,9}` before the new tasks land — i.e. assert orders are unique and `ta-vla`=10, `td3`=11, `td3-bc`=12, `residual-rl`=13.
- [ ] **Step 2: Apply the renumber + reference edits; test green; grep shows no stale "task [6-9] (" references to the moved tasks.**
- [ ] **Step 3: Commit** `refactor(lab): shift ta-vla/td3/td3-bc/residual-rl to ladder 10-13`.

### Task 4: `attention` task (order 6) + the live-panel mechanism

**Files:**
- Create: `lab/tasks/task_attention.py`, `lab/web/lessons/attention.html`
- Modify: `lab/tasks/base.py` (new field), `lab/tasks/__init__.py`, `lab/web/app.js`, `lab/web/viewers.js`, `lab/tasks/calib.json`
- Test: `tests/lab/test_task_attention.py`

**Interfaces:**
- Consumes: Task 1's `MHA`, `attn_snapshot`.
- Produces: `TaskSpec.live: str = ""` on base.py — name of a `LabViewers`-registered renderer; while a run is attached and non-terminal, app.js keeps one container under the charts and calls the renderer with the newest metrics row containing the key `attn` (rows arrive through the existing since-polling; renderer contract `fn(row, container, ctx)` with in-place update like vecgrid). Later tasks reuse this.
- Task semantics (from spec): sequences of `n_pairs` fused (key+value-embedding sum) tokens + 1 query token; model = embeddings + one `MHA` + readout at the query position; CE loss on the value. Params: `heads` (choice 1/2/4, default 2), `d_k` (int 8–32, default 16), `n_pairs` (int 2–8, default 4), `steps`, `lr`, `seed`. Eval params: `eval_seed`, `eval_n_pairs`.
- Training emits `ctx.log(step, loss=…, acc=…)` plus, ~60× per run, `attn=attn_snapshot(...)` and `qkv` (vecgrid payload: per-token Q/K/V of a fixed probe example). Charts: loss (log_y), acc. `vecgrids=True`.
- Eval result schema (document in module docstring): the probe sequence (token labels), per-head Q/K/V matrices, scores pre/post softmax, the mixture vector, prediction + target, and per-stage arrays for the **inference stepper** viewer `attnstep` (stage slider: embed → Q/K/V → scores → softmax → mixture → readout).

- [ ] **Step 1: Failing tests** — `test_spec_shape` (registered, order 6, param names unique, live="attnlive", vecgrids True), `test_train_reaches_target` (default-ish tiny config: 1500 steps reach acc ≥ 0.99 — the spec target), `test_metrics_rows_small` (every metrics.jsonl line < 8 KB), `test_eval_roundtrip` (train → ckpt → evaluate through the real job code; result has the documented keys; model rebuilt from ckpt meta only), `test_estimate_uses_calib`.
- [ ] **Step 2: Implement task module + base.py `live` field + app.js live-panel hook + `attnlive` and `attnstep` viewers (heatmap cells via vecgrid palette; JSON fallback).**
- [ ] **Step 3: Tests green; `node --check lab/web/*.js`; ruff.**
- [ ] **Step 4: Write `attention.html`** (schematic: embeddings → W_Q/W_K/W_V with shapes → QKᵀ/√d_k → softmax → ΣαV → readout, loss marked; KaTeX; measured numbers from a real run; refer to neighbours by new ladder numbers).
- [ ] **Step 5: calib.json entry measured from a default run; estimate within ~2× wall.**
- [ ] **Step 6: Commit** `feat(lab): attention task - KV lookup with live Q/K/V and attention stepper`.

### Task 5: `vit-mnist` task (order 7)

**Files:**
- Create: `lab/tasks/task_vit.py`, `lab/web/lessons/vit-mnist.html`
- Modify: `lab/tasks/__init__.py`, `lab/web/viewers.js`, `lab/tasks/calib.json`
- Test: `tests/lab/test_task_vit.py`

**Interfaces:**
- Consumes: `Block`, `attn_snapshot`, `load_mnist`, the Task 4 live-panel (`live="attnlive"` reused as-is — payload adds `img` (28×28 uint8 list) + `grid` (patch grid dims) and the renderer overlays when present; extend `attnlive` accordingly, keep fallback).
- Model: patchify (`patch` 7 or 14) → linear embed + learned positions + CLS → `depth` × `Block` → CLS head → 10 classes. Params: `patch` (choice 7/14), `depth` (1/2, default 2), `heads` (2/4, default 2), `d_model` (default 64), `lr`, `steps`, `train_n` (subset, default 10000), `seed`. Eval params: `eval_seed`, `n_examples` (default 8).
- Charts: train loss (log_y), val acc (`val_acc` logged on periodic held-out eval). Live: CLS-row attention per layer/head reshaped to the patch grid over the current probe digit.
- Eval result: for each sampled test image — pixels, prediction, confidence, per-layer/per-head CLS attention maps; viewer `vitattn` (image + selectable layer/head overlay). `vecgrids=True` (patch/CLS embedding grids).

- [ ] **Step 1: Failing tests** — `test_spec_shape`, `test_train_reaches_target` (defaults sized so a shortened test config ≥ 0.85 on 1000 held-out; defaults themselves must hit the spec's ≥ 0.90 — assert in a marked-slow test or verify via the measured default run recorded in the lesson), `test_eval_roundtrip` (keys incl. attention map shapes = depth×heads×grid), `test_patchify_roundtrip` (patchify then unfold-back equals input).
- [ ] **Step 2: Implement; tests green; JS check; ruff.**
- [ ] **Step 3: Lesson** (schematic: image → patches → tokens (+CLS, +positions) → block internals → CLS readout, shapes labeled; measured default-run accuracy).
- [ ] **Step 4: calib entry; commit** `feat(lab): vit-mnist task - mini ViT with live CLS attention overlays`.

### Task 6: `cross-attn` task (order 8)

**Files:**
- Create: `lab/tasks/task_crossattn.py`, `lab/web/lessons/cross-attn.html`
- Modify: `lab/tasks/__init__.py`, `lab/web/viewers.js`, `lab/tasks/calib.json`
- Test: `tests/lab/test_task_crossattn.py`

**Interfaces:**
- Consumes: `Block` (encoder), `CrossBlock` (decoder), `causal_mask`, `attn_snapshot`, live panel.
- Data: generator of (source, target) date pairs; source formats `"23 Oct 2026"`, `"Oct 23, 2026"`, `"23/10/2026"` (uniform choice), target `"2026-10-23"`; char vocab + BOS/EOS/PAD; dates span 1970–2070, deterministic train/held-out split by date so exact-match is measured on unseen dates.
- Model: char embed + positions → 1 encoder `Block`; decoder = embed + positions → 1 `CrossBlock` → vocab head. Teacher forcing CE. `greedy_decode(model, src) -> str` builds from BOS only. Params: `d_model`, `heads`, `lr`, `steps`, `seed`. Eval: `eval_seed`, `n_examples`.
- Charts: loss (log_y), `exact` (held-out exact-match). Live: cross-attention matrix of a fixed probe pair.
- Eval result per example: source/target/decoded strings, correct flag, enc self-attn, dec self-attn, cross-attn matrices with char labels; viewer `xattn` (alignment heatmap with char ticks + a decode stepper: slider over output positions showing the cross-attention row used at each step).

- [ ] **Step 1: Failing tests** — `test_spec_shape`, `test_generator_split_disjoint`, `test_greedy_decode_autoregressive` (decode called with only `src`; output ends at EOS; monkeypatch asserts the decoder is never fed target chars), `test_train_reaches_target` (shortened config ≥ 0.9 exact on held-out; defaults must hit ≥ 0.95), `test_eval_roundtrip`.
- [ ] **Step 2: Implement; green; JS check; ruff.**
- [ ] **Step 3: Lesson** — the wiring (Q from decoder, K/V from encoder, drawn), where cross-attention is used (seq2seq, Perceiver latents, VLMs, **ACT in this repo** — cite the actual `src/fvb` Stage-1 code location — and the relation to TAVLAMini's block attention in task 10).
- [ ] **Step 4: calib entry; commit** `feat(lab): cross-attn task - date translation with alignment matrix`.

### Task 7: `find-digit` task (order 9)

**Files:**
- Create: `lab/tasks/task_finddigit.py`, `lab/web/lessons/find-digit.html`
- Modify: `lab/tasks/__init__.py`, `lab/web/viewers.js`, `lab/tasks/calib.json`
- Test: `tests/lab/test_task_finddigit.py`

**Interfaces:**
- Consumes: `Block`, `MHA` (the cross-attention readout), `load_mnist`, live panel, vit patchify approach (reimplement locally or import from task_vit if exported — exporting `patchify(imgs, patch) -> tokens` from `task_vit` is the cleaner call; do that).
- Data generator: canvas 28×56 = two MNIST digits of **distinct classes**; query class 0–9; label `0=left, 1=right, 2=absent` with P(absent) ≈ ⅓ (query sampled from the complement); optional gaussian pixel noise (eval knob).
- Model: patch tokens (7-px → 4×8 grid) + positions → `depth` × `Block` → learned query-class embedding token cross-attends via one `MHA` over patch tokens → 3-way head. Joint from scratch. Params: `depth` (1/2), `heads` (2/4), `d_model`, `lr`, `steps`, `train_n`, `seed`. Eval: `eval_seed`, `n_examples`, `noise` (float 0–0.5, default 0), `query` (choice "random"/0–9).
- Charts: loss (log_y), `val_acc`, `absent_acc`. Live: query→patch attention over the probe canvas.
- Eval result per example: canvas pixels, query, truth, prediction, per-head query attention maps (4×8); viewer `finddigit` (canvas + overlay + verdict chips). `vecgrids=True` (query embedding vs. sample patch K vectors).

- [ ] **Step 1: Failing tests** — `test_generator_invariants` (distinct classes; absent fraction 0.25–0.40 over 2000 samples; label matches construction), `test_spec_shape`, `test_train_reaches_target` (shortened ≥ 0.8; defaults ≥ 0.90), `test_eval_roundtrip` (attention map shape heads×4×8; absent case present among examples), `test_noise_param_applies` (noise > 0 changes pixels, bounded).
- [ ] **Step 2: Implement (+ export `patchify` from task_vit); green; JS check; ruff.**
- [ ] **Step 3: Lesson** — "a VLA in miniature": image tokens + command token → grounded answer; schematic shows the query's cross-attention readout; measured default accuracy.
- [ ] **Step 4: calib entry; commit** `feat(lab): find-digit task - query-conditioned visual attention`.

### Task 8: Snippet infrastructure + snippets in the four new lessons

**Files:**
- Create: `lab/web/vendor/highlight/` (highlight.js 11.9.0 core+python build: `highlight.min.js`, `github.min.css`, `github-dark.min.css`), `tests/lab/test_snippets.py`
- Modify: `lab/web/index.html`, `lab/web/app.js`, `lab/web/style.css`, `lab/web/vendor/VENDOR.md`, the four new lessons
- Test: `tests/lab/test_snippets.py` + extend `tests/lab/test_static.py` (vendor files served)

**Interfaces (Produces — Task 9 reuses verbatim):**
- Snippet block convention in lessons:
  `<details class="code"><summary>Training loop</summary><pre><code class="language-python" data-src="lab/tasks/minitf.py">…excerpt…</code></pre></details>`
  Collapsed by default; `data-src` names the repo file the excerpt comes from. Elision lines are exactly `# ...`.
- app.js: after lesson insert (and after `typesetMath`), run `hljs.highlightElement` on every `.lesson pre code`; dark/light via the two CSS files with `media` attributes gated like the existing theme handling.
- Drift test contract: for every `code[data-src]` in every lesson, each line that is non-blank and not `# ...` must appear, whitespace-stripped, as a stripped line of the named source file. Snippet blocks must not contain `$$` or `\(` (KaTeX safety).

- [ ] **Step 1: Vendor highlight.js (exact version 11.9.0, core+python custom build from the official CDN artifacts), VENDOR.md entry, index.html wiring, `.lesson details.code` styling (monospace, compact, theme-aware).**
- [ ] **Step 2: Failing drift test + static-serving test; then add snippets to the four new lessons** — each gets at least: the attention/model core (from `minitf.py` or the task file), the training loop (from the task's `train` / `lab/core/loop.py`), and the inference/decode path (from `evaluate` or `greedy_decode`).
- [ ] **Step 3: Tests green; JS check; browser-verifiable (controller does the visual pass).**
- [ ] **Step 4: Commit** `feat(lab): collapsible highlighted code snippets in lessons + drift guard`.

### Task 9: Retrofit snippets into the nine existing lessons + README

**Files:**
- Modify: `lab/web/lessons/{bc-mse,flow,embeddings,flow-bc,force-token,ta-vla,td3,td3-bc,residual-rl}.html`, `lab/README.md`
- Test: covered by `tests/lab/test_snippets.py` (it scans all lessons)

- [ ] **Step 1: Each existing lesson gets 2–3 snippet blocks** (training loop, inference/eval path, and the task's core model or loss), excerpted from that task's module / `lab/core/loop.py` / `src/fvb` files, following Task 8's convention exactly; drift test green across all 13 lessons.
- [ ] **Step 2: README ladder table: 13 rows with the new tasks and measured default runtimes; "Adding a task" gains one line each for `live` and the snippet convention.**
- [ ] **Step 3: Full lab suite in the image green; ruff; JS check.**
- [ ] **Step 4: Commit** `feat(lab): code snippets in all lessons; README ladder 13 tasks`.
