"""A small, readable TA-VLA training library: diffusion / flow-matching action heads with a
torque token, from a 2-D toy up to a mini vision-language-action model.

Modules, in reading order:

* ``generative`` - the two ways to train a generative action head: flow matching (what pi0 /
  SmolVLA / TA-VLA use) and DDPM (Diffusion Policy). Training loss and sampling for both.
* ``nets`` - the denoisers: a plain MLP (T1, T2) and ``TAVLAMini``, a two-stream transformer with
  the pi0 layout (a "VLM" prefix over image / language / state, an action-expert suffix that
  attends to the prefix, the TA-VLA torque token, an optional auxiliary torque head).
* ``toy`` - T1 (two-mode toy) and T2 (point mass around an obstacle, action chunks).
* ``contact2d`` - a 2-D compliant contact world (impedance-controlled point peg, a surface with a
  chamfered hole, force sensing) used by T3 (peg search) and T4 (four language-conditioned
  tasks with rendered images).
* ``data`` - episodes -> (observation, action chunk, torque history, future torque) samples.
* ``train`` - the training loop, the chunked closed-loop policy and the evaluation.

Run the experiments with ``scripts/40_ta_tutorial.py``; ``docs/ta_tutorial.md`` explains them.
"""
