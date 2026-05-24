"""Latent Video Encoder — V-JEPA-style self-supervised video encoder pretraining.

This package trains the *encoder only*. The action-conditioned world-model
predictor (V-JEPA 2-AC style) is a separate later stage built on the frozen
encoder exported from here.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
