"""Run with: python3 tests/test_feature_extractors_clip_compat.py

Tests _unwrap_clip_pooled_features (feature_extractors.py), the defensive
fix for a transformers-version discrepancy where CLIPModel.get_text_features()
/get_image_features() have been observed to return a ModelOutput-like
object instead of a plain tensor. Uses real torch tensors and small fake
wrapper objects -- no real CLIP model needed (that requires GPU +
downloaded weights, exercised for real only on the server).
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from feature_extractors import _unwrap_clip_pooled_features


class _FakeOutputWithTextEmbeds:
    def __init__(self, text_embeds):
        self.text_embeds = text_embeds


class _FakeOutputWithImageEmbeds:
    def __init__(self, image_embeds):
        self.image_embeds = image_embeds


class _FakeOutputWithOnlyPooler:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _FakeOutputWithNothingUseful:
    def __init__(self):
        self.last_hidden_state = torch.randn(2, 10, 512)  # wrong shape for our purposes


def test_plain_tensor_passthrough():
    t = torch.randn(4, 512)
    result = _unwrap_clip_pooled_features(t)
    assert result is t
    print("test_plain_tensor_passthrough OK")


def test_unwraps_text_embeds():
    t = torch.randn(4, 512)
    wrapped = _FakeOutputWithTextEmbeds(t)
    result = _unwrap_clip_pooled_features(wrapped)
    assert result is t
    print("test_unwraps_text_embeds OK")


def test_unwraps_image_embeds():
    t = torch.randn(1, 512)
    wrapped = _FakeOutputWithImageEmbeds(t)
    result = _unwrap_clip_pooled_features(wrapped)
    assert result is t
    print("test_unwraps_image_embeds OK")


def test_falls_back_to_pooler_output_with_warning():
    t = torch.randn(4, 768)
    wrapped = _FakeOutputWithOnlyPooler(t)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = _unwrap_clip_pooled_features(wrapped)
    assert result is t
    assert "WARNING" in buf.getvalue()
    print("test_falls_back_to_pooler_output_with_warning OK")


def test_raises_clear_error_when_nothing_usable():
    wrapped = _FakeOutputWithNothingUseful()
    try:
        _unwrap_clip_pooled_features(wrapped)
        assert False, "expected TypeError"
    except TypeError as e:
        assert "text_embeds" in str(e) or "pooler_output" in str(e)
    print("test_raises_clear_error_when_nothing_usable OK")


def test_normalization_works_after_unwrap():
    """End-to-end check: after unwrapping, the L2-normalize step that
    ClipScorer applies (emb / emb.norm(...)) produces unit vectors,
    exactly as if a plain tensor had been returned directly."""
    t = torch.randn(3, 128) * 5.0
    wrapped = _FakeOutputWithTextEmbeds(t)
    emb = _unwrap_clip_pooled_features(wrapped)
    normalized = emb / emb.norm(p=2, dim=-1, keepdim=True)
    norms = normalized.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("test_normalization_works_after_unwrap OK")


if __name__ == "__main__":
    test_plain_tensor_passthrough()
    test_unwraps_text_embeds()
    test_unwraps_image_embeds()
    test_falls_back_to_pooler_output_with_warning()
    test_raises_clear_error_when_nothing_usable()
    test_normalization_works_after_unwrap()
    print("\nALL feature_extractors.py CLIP-compat TESTS PASSED")
