"""
gdino_scorer.py
================

GroundingDINO-based zero-shot detection scorer, providing the 4th feature
dimension s_gdino(w) for the Strategy 8-U-NC evidence vector.

IMPLEMENTATION NOTE: this uses transformers' built-in GroundingDINO support
(AutoModelForZeroShotObjectDetection, transformers>=4.40), NOT the
standalone `groundingdino` pip package from IDEA-Research/GroundingDINO.

Why: the standalone package (a) requires compiling a custom CUDA extension
(MultiScaleDeformableAttention) against a specific PyTorch/CUDA ABI, which
frequently breaks on newer PyTorch versions (2.1+, and especially 2.6,
where the extension's build flags/headers no longer match), and (b) loads
its checkpoint via a raw `torch.load(..., weights_only=False)` call
internally, which conflicts with PyTorch 2.6's new default of
`weights_only=True` and requires patching the package itself to fix.

transformers' GroundingDINO implementation avoids both problems: it's pure
PyTorch (no custom compiled kernels), loads weights through the standard
`from_pretrained` / safetensors path exactly like OWL-ViT and CLIP already
do in this codebase, and is maintained against current transformers/torch
releases. Since `transformers` is already a hard dependency here (see
feature_extractors.py), this adds no new package at all.

Interface mirrors OwlViTScorer's score_batch: given an image and a list of
object names, returns one confidence score per name — the maximum
detection confidence (post-sigmoid) GroundingDINO assigns to that text
query, using a near-zero box/text threshold so the score stays a genuinely
continuous evidence value (not pre-thresholded into a hard yes/no), the
same design principle feature_extractors.py's OWL-ViT scorer follows.

NOTE: like OwlViTScorer, this class requires GPU + downloading model
weights and cannot run in a sandbox without that access. Tests exercise
`_query_text` and the batching/aggregation logic via a stub subclass.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


class GDINOScorer:
    """Zero-shot detector scorer using transformers' GroundingDINO: s_gdino."""

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        score_threshold: float = 1e-6,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.score_threshold = score_threshold
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
        a single image. Returns the maximum detection confidence, or 0.0
        if literally no boxes survive even a near-zero threshold (i.e.
        the query produced no signal at all)."""
        import torch

        text = self._query_text(object_name)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        w, h = image.size  # PIL Image: (width, height)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.score_threshold,
            text_threshold=self.score_threshold,
            target_sizes=[(h, w)],
        )[0]

        scores = results["scores"]
        if scores.numel() == 0:
            return 0.0
        return float(scores.max().item())

    def score_batch(self, image, object_names: Sequence[str]) -> List[float]:
        """Score ALL `object_names` against a single `image`.

        Unlike OWL-ViT (which natively batches multiple text queries into
        one forward pass), GroundingDINO's caption-conditioned architecture
        processes one text query at a time here — each call re-encodes the
        image, so cost scales with len(object_names). This is why
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
