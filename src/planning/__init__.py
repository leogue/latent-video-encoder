"""Planning over the latent world model — CEM / model-predictive control.

Stage 3. Given a goal observation encoded to a latent target, optimize an action
sequence so the predictor's rollout reaches it: sample action sequences (CEM),
score by latent distance to the goal, refine, execute the first action, replan.
All inference happens in the frozen encoder's latent space — no pixel decoding.

Planned modules:
- ``cem.py``      — Cross-Entropy Method action-sequence optimizer.
- ``planner.py``  — MPC loop wrapping the predictor + a goal latent.
"""

from __future__ import annotations

__all__: list[str] = []
