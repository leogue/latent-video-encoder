"""Latent world predictor — action-conditioned dynamics (V-JEPA 2-AC style).

Stage 2 of the latent world model. Takes the **frozen** encoder's latents plus
proprioceptive state and actions, and predicts the next-step latent. Trained by
L1 regression in latent space (teacher-forcing + multi-step rollout), with no
collapse machinery — the targets come from a fixed encoder, so this is plain
supervised regression of dynamics.

Planned modules:
- ``dataset.py``   — trajectory sampling (obs + state + action) from LeRobot.
- ``predictor.py`` — block-causal transformer with separate input heads for
  visual / state / action tokens.
- ``losses.py``    — teacher-forcing + rollout L1.
- ``trainer.py``   — training loop over a frozen encoder.
"""

from __future__ import annotations

__all__: list[str] = []
