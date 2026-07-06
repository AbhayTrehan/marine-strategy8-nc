"""
gdino_scorer.py
================

GroundingDINO-based zero-shot detection scorer, providing the 4th feature
dimension s_gdino(w) for the Strategy 8-U-NC evidence vector. This is
architecturally independent from the existing OWL-ViT scorer (different
model family, different training data, different detection head), so
agreement between the two on the same word is a much stronger signal of
real presence than either one alone — the core motivation for adding this.

Interface mirrors OwlViTScorer's score_batch exactly: given an image and a
list of object names, returns one confidence score per name. The score is
the maximum bounding-box logit from GroundingDINO for that text query,
passed through a sigmoid — same convention as owlvit_postprocess.

GroundingDINO is loaded from the `groundingdino` package (DINO variant
from the IDEA-Research/GroundingDINO repo). If the package isn't installed,
this module raises a clear ImportError at construction time, not at import
time — so other modules can safely `import gdino_scorer` without the
package being present, and only fail if they actually try to instantiate
the scorer.

NOTE: Like OwlViTScorer, this class requires GPU + model weights and
cannot run in a sandbox without them. Tests use a stub scorer.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


class GDINOScorer:
    """Zero-shot detector scorer using GroundingDINO: s_gdino (4th feature)."""

    def __init__(
        self,
        config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weights_path: str = "groundingdino_swint_ogc.pth",
        device: str = "cuda",
        box_threshold: float = 0.01,
        text_threshold: float = 0.01,
    ):
        try:
            from groundingdino.util.inference import load_model
        except ImportError:
            raise ImportError(
                "GroundingDINO is not installed. Install it from "
                "https://github.com/IDEA-Research/GroundingDINO:\n"
                "  pip install groundingdino\n"
                "or clone + pip install -e . from the repo."
            )

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = load_model(config_path, weights_path, device=device)

    @staticmethod
    def _query_text(object_name: str) -> str:
        return object_name.lower().strip()

    def _score_single_query(self, image_source, image_tensor, query: str) -> float:
        """Score a single text query against the already-loaded image tensor.
        Returns the maximum box confidence (sigmoid of logit), or 0.0 if no
        boxes pass the threshold."""
        from groundingdino.util.inference import predict

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=query,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if len(logits) == 0:
            return 0.0
        return float(logits.max().item())

    def score_batch(self, image_path_or_pil, object_names: Sequence[str]) -> List[float]:
        """Score ALL `object_names` against a single image.

        Unlike OWL-ViT (which natively batches multiple text queries in one
        forward pass), GroundingDINO's standard inference API processes one
        text caption at a time. We call it per-query, but the image encoding
        is cached internally by the model, so the per-query overhead is
        primarily the text encoder + cross-attention, not a full re-encode
        of the image.

        Args:
            image_path_or_pil: either a file path (str) or a PIL.Image.
                GroundingDINO's load_image expects a path, so if given a
                PIL image, we handle the conversion.
            object_names: list of object words to score.

        Returns:
            list of s_gdino scores, one per object_name, same order.
        """
        if len(object_names) == 0:
            return []

        from groundingdino.util.inference import load_image
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(image_path_or_pil, str):
            image_source, image_tensor = load_image(image_path_or_pil)
        elif isinstance(image_path_or_pil, PILImage.Image):
            # GroundingDINO's load_image reads from a file path; convert
            # PIL -> the same (numpy_source, tensor) format it produces.
            import torch
            from torchvision import transforms

            img_rgb = image_path_or_pil.convert("RGB")
            image_source = np.array(img_rgb)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor = transform(img_rgb)
        else:
            raise TypeError(f"Expected str path or PIL Image, got {type(image_path_or_pil)}")

        results: List[float] = []
        for name in object_names:
            query = self._query_text(name)
            try:
                s = self._score_single_query(image_source, image_tensor, query)
            except Exception:
                s = 0.0
            results.append(s)

        return results

    def score_map(self, image, object_names: Sequence[str]) -> Dict[str, float]:
        scores = self.score_batch(image, object_names)
        return dict(zip(object_names, scores))
