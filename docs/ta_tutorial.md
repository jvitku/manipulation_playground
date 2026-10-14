# Training diffusion / flow-matching policies for a TA-VLA — a hands-on tutorial

Code: `src/fvb/ta_tutorial/` (≈ 900 lines, read in this order: `generative.py`, `nets.py`,
`data.py`, `train.py`, then the tasks `toy.py` and `contact2d.py`). Experiments:
`scripts/40_ta_tutorial.py` (T1–T4, VLA image) and `scripts/42_ta_tutorial_t5.py` (T5, train
image). Results with all curves: the "TA-VLA tutorial" report (link in the progress journal,
entry `TUT`). Tests: `tests/test_ta_tutorial.py`.

## 1. What a TA-VLA is

A vision-language-action model (pi0, SmolVLA) has two parts that share one attention sequence:

```
prefix  [ image patches | instruction tokens | robot state ]      -> "VLM" weights   (encoder)
suffix  [ torque token | a_t, a_t+1, ..., a_t+H-1  (noisy) ]      -> "action expert" (decoder)
```

* The **prefix** is the observation. In SmolVLA it is a pretrained small VLM (SmolVLM2); here a
  3-layer CNN for 32×32 images, a word embedding and a linear layer for the state.
* The **suffix** is an *action chunk* of H future actions, plus the denoising time. The action
  expert has its own weights but attends to the prefix in every layer (block attention mask:
  prefix → prefix only; suffix → prefix + suffix). The prefix never sees the noisy actions, so
  its keys/values are computed once per observation and reused for every denoising step.
* **TA-VLA** (Zhang et al. 2025, arXiv 2509.07962) adds force: the last few frames of joint
  torques / force readings are flattened, passed through a 2-layer MLP and become **one token**.
  Their ablation: put it in the **decoder** (suffix), next to the actions — better than in the
  encoder (prefix) or as extra state. Optionally an **auxiliary loss** asks the expert to predict
  the *future* torques: `L = L_action + 0.1 · L_torque`.

`nets.TAVLAMini` is exactly this layout with `torque_pos ∈ {prefix, suffix}`, `zero_torque`
(same parameters, input zeroed: the no-force twin) and `aux_dim` (the auxiliary head).

## 2. Why the action head is generative

Demonstrations are multimodal: around an obstacle the expert goes left in half the episodes and
right in the other half; in a search it sweeps left or right first. A regression head trained with
MSE predicts the conditional **mean** — straight into the obstacle (T1: 100 % collisions). A
generative head learns the **distribution** and samples one mode.

### Flow matching (pi0, SmolVLA, TA-VLA) — `generative.FlowMatching`

Training step for one expert chunk `x0` (H × A, actions in [-1, 1]):

1. draw a noise level `t ~ Beta(1.5, 1)` (pi0's choice: more weight on noisy t) and noise `e ~ N(0, I)`;
2. mix: `x_t = t·e + (1 − t)·x0` — a straight line from the data (t = 0) to noise (t = 1);
3. the velocity of that line is `u = e − x0`; the network predicts `v = f(x_t, t, observation)`;
4. loss `|v − u|²`, averaged over the chunk (padding past the episode end masked).

Sampling: start from noise `x_1 = e`, take K Euler steps `x ← x − (1/K)·v(x, t)` from t = 1 to 0.
Different noise draws end in different modes. Because the regression target is an average over
all chunks passing through `x_t`, the field is straight where the data is unimodal (one step is
enough) and bends where it splits (more steps needed) — measured in T1 and T2.

### DDPM / DDIM (Diffusion Policy) — `generative.DDPM`

Same idea, different path: a cosine schedule `ᾱ_k`, `x_k = √ᾱ_k·x0 + √(1−ᾱ_k)·e`, the network
predicts the noise `e`; sampling with DDIM walks a subsequence of the 100 training timesteps,
clipping the implied `x0` to [-1, 1] at every step.

### The baseline — `generative.Regression`

The same network, fed zeros instead of `x_t`, trained with MSE on the chunk itself.

## 3. One training step of a TA-VLA, end to end

```
batch of B samples (episode e, step i):
  observation  image_i, instruction, state_i           -> prefix tokens
  torque       F_{i-9..i}  (10 frames × 2), z-scored    -> torque_mlp -> 1 token (suffix)
  target       x0 = a_{i..i+H-1}, mask (padding)        (actions already in [-1, 1])
  aux target   F_{i+1..i+8} z-scored                    (only with the auxiliary loss)

t ~ Beta(1.5, 1), e ~ N(0, I);   x_t = t·e + (1−t)·x0
prefix K/V  = VLM stream over the prefix (all layers)
suffix      = [torque token | MLP(action_in(x_t) ⊕ time_emb(t)) + positions]
            → expert stream, each layer attends to [prefix K/V ; suffix K/V]
v           = action_out(last H suffix outputs)
L           = mean_mask |v − (e − x0)|²  +  w · |aux_head(mean of action outputs) − future F|²
AdamW (lr 3e-4, wd 1e-4), warm-up 500 steps, cosine decay, grad-norm clip 1, EMA 0.999 of the weights
```

Inference (`train.ChunkPolicy`): encode the prefix once (`with model.cache(cond)`), integrate
10 Euler steps from noise, execute the first `n_exec` actions of the chunk open loop, observe,
re-plan. TA-VLA / SmolVLA execute part of each chunk for the same reason: a whole chunk is a
commitment to one mode (no left/right dithering), a short execution keeps the loop reactive to
contact.

## 4. The experiments

| | task | network | what it teaches |
|---|---|---|---|
| T1 | 2-D waypoint around an obstacle, two modes | MLP | regression averages modes; flow vs DDPM; training budget; sampling steps |
| T2 | point mass around an obstacle, closed loop | MLP, chunks | chunk length, demo count, sampling steps; loss vs success |
| T3 | 2-D peg-in-hole, hole known to ±3 cm, unknown surface height and stiffness | TAVLAMini (state + torque) | torque token: suffix vs prefix vs zeroed; auxiliary loss; demos |
| T4 | 4 language-conditioned tasks from 32×32 images: insert into the red / blue hole, press gently (2 N) / firmly (6 N) | TAVLAMini (image + language + state + torque) | the full TA-VLA layout; per-task value of force |
| T5 | the MuJoCo insertion skill (`SortInsertEnv`) | TAVLAMini (state + 60-D torque history) | the same recipe on a real contact simulation |

The contact world (`contact2d.py`) is built so that force is *necessary* for some decisions and
*not* for others: the impedance-controlled peg rests on the surface while the controller's spring
presses with `F = k·(pos − ref)`; surface height and stiffness are random per episode, so *when*
contact happens and *how hard* the peg presses are only visible in the force signal; on a 45°
chamfer the reaction force tilts toward the hole (Fx says which way to slide); pushing above
15 N aborts the episode. A press task is impossible without force (2 N vs 6 N under a
5×-uncertain stiffness); finding a hole that vision put 3 cm off needs a force-guided search.

## 5. Convergence: what to expect, what to watch

Measured numbers are in the report; the general lessons:

* **The training loss is not the progress signal.** It averages over all noise levels and is
  dominated by the noisy ones, which nothing can predict well; it flattens early while closed-loop
  success is still climbing. Track a held-out loss with *fixed* noise draws (smooth enough to
  compare) and, above all, closed-loop success on fixed evaluation scenes.
* **Budget.** Even the 2-D toy needs ~10⁴ steps at lr 1e-3 for sharp modes; at lr 3e-4 and
  5 k steps the samples are still blurred across the gap (43 % "collisions") while the loss looks
  converged.
* **EMA.** Evaluate the exponential moving average of the weights; it is standard for diffusion
  policies and removes most of the late-training jitter.
* **Sampling steps.** Flow matching works with few Euler steps on near-unimodal chunks and needs
  more where modes split; DDIM with very few steps degrades more.
* **Chunks.** H = 1 with a generative head can switch modes between consecutive steps (left,
  right, left... into the obstacle); chunks fix it; executing half a chunk keeps reactivity.
* **Normalisation.** Z-score every input including the torque history; mixing a 47 N grip force
  with 1 N joint torques (our first SmolVLA runs) makes the token hard to use.
* **Data.** Measure success against the number of demonstrations; the force token's benefit
  grows where the demonstrations contain force-dependent decisions the policy can't infer otherwise.
