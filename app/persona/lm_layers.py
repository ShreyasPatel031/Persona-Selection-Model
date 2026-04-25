"""Resolve decoder layer stack without importing ``transformers.modeling_utils`` (keeps import graph light)."""

from __future__ import annotations

import torch.nn as nn


def language_model_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve decoder layers for Gemma-3 multimodal and common causal LMs."""
    m = model
    if hasattr(m, "model") and m.model is not None:
        inner = m.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    raise RuntimeError(
        "Could not find decoder layers on this model class; extend language_model_layers."
    )
