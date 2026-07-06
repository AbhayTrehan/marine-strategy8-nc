"""
gdino_scorer.py
================

GroundingDINO-based zero-shot detection scorer, providing the 4th feature
dimension s_gdino(w) for the Strategy 8-U-NC evidence vector.

IMPLEMENTATION NOTE 1: this uses transformers' built-in GroundingDINO
support (AutoModelForZeroShotObjectDetection, transformers>=4.40), NOT the
standalone `groundingdino` pip package from IDEA-Research/GroundingDINO --
that package needs a custom CUDA extension that frequently breaks on
newer PyTorch versions and loads checkpoints via a raw torch.load() call
incompatible with PyTorch 2.6's weights_only default change. transformers'
implementation is pure PyTorch and loads weights the same safetensors way
OWL-ViT/CLIP already do in this codebase.

IMPLEMENTATION NOTE 2: this does NOT call
`GroundingDinoProcessor.post_process_grounded_object_detection`. That
convenience method's keyword arguments (box_threshold / text_threshold /
threshold) have changed across transformers releases -- confirmed
empirically (a real run hit `unexpected keyword argument 'box_threshold'`
against an installed transformers version that had already renamed it).
Exactly like feature_extractors.py's OwlViTScorer avoids
`post_process_object_detection` for the same reason, we instead read
`outputs.logits` directly and apply the same sigmoid + max computation
the post-processing method performs internally -- this only depends on
the model's raw output tensor shape (batch, num_queries, max_text_len),
which is part of GroundingDINO's stable architecture, not a
convenience-wrapper's changing call signature.

Score computation, mirroring GroundingDINO's own post-processing math:
    probs = sigmoid(logits)                    # (num_queries, max_text_len)
    per_query_score = probs.max(dim=-1).values # max over text-token dim
    s_gdino = per_query_score.max()            # max over queries

We only need a scalar confidence (not box coordinates), so box
post-processing is skipped entirely -- simpler and removes another
version-sensitive surface (image-size/coordinate-convention handling).

NOTE: like OwlViTScorer, this class requires GPU + downloading model
weights and cannot run in a sandbox without that access. Tests exercise
`_query_text`, `gdino_postprocess`, and the batching/error-isolation logic
via synthetic tensors / a stub subclass.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def gdino_postprocess(logits) -> float:
    """Pure tensor math, factored out of GDINOScorer._score_one so it can
    be unit-tested with a synthetic tensor (no real model needed):
    given raw GroundingDINO logits for ONE (image, text query) pair,
        logits: (num_queries, max_text_len)
    returns a single scalar confidence: sigmoid, max over the text-token
    dimension (does this query match ANY token in the text), then max
    over queries (is ANY query a strong match)."""
    import torch

    if logits.numel() == 0 or logits.shape[0] == 0:
        return 0.0
    probs = torch.sigmoid(logits)
    per_query_score = probs.max(dim=-1).values  # (num_queries,)
    return float(per_query_score.max().item())


class GDINOScorer:
    """Zero-shot detector scorer using transformers' GroundingDINO: s_gdino."""

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _query_text(object_name: str) -> str:
        """GroundingDINO expects lowercase, period-terminated phrases
        (its text encoder segments captions on '.')."""
        cleaned = object_name.lower().strip().rstrip(".")
        return f"{cleaned}."

    def _score_one(self, image, object_name: str) -> float:
        """One GroundingDINO forward pass for a single text query against
        a single image. Computes the confidence directly from raw
        `outputs.logits` (see module docstring) -- no dependency on any
        post-processing convenience method's keyword-argument signature."""
        import torch

        text = self._query_text(object_name)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]  # (num_queries, max_text_len)
        return gdino_postprocess(logits)

    def score_batch(self, image, object_names: Sequence[str]) -> List[float]:
        """Score ALL `object_names` against a single `image`.

        Unlike OWL-ViT (which natively batches multiple text queries into
        one forward pass), GroundingDINO's caption-conditioned architecture
        processes one text query at a time here -- each call re-encodes
        the image, so cost scales with len(object_names). This is why
        probe_sampling's CLIP shortlisting step (Filter 1.5) matters: it
        keeps the number of words reaching this function bounded (e.g.
        ~200) regardless of vocabulary size.
        """
        if len(object_names) == 0:
            return []
        results = []
        for name in object_names:
            try:
                s = self._score_one(image, name)
            except Exception as e:
                print(f"[GDINOScorer] WARNING: query '{name}' failed ({e}), scoring 0.0")
                s = 0.0
            results.append(s)
        return results

    def score_map(self, image, object_names: Sequence[str]) -> Dict[str, float]:
        scores = self.score_batch(image, object_names)
        return dict(zip(object_names, scores))
