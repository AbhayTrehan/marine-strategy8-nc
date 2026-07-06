"""
feature_extractors.py
======================

Real implementations of the zero-shot vision models used to build the
feature vector for every candidate/probe object.

CRITICAL DESIGN PRINCIPLE: every model output is consumed via raw tensor
math on the model's fundamental architectural outputs (logits, pred_boxes,
submodule outputs + projection layers), NEVER via a convenience
post-processing method or a high-level get_*_features() wrapper. This is
because:
  - post_process_object_detection keyword arguments have changed names
    across transformers versions (confirmed empirically);
  - get_text_features/get_image_features return type has changed from a
    plain tensor to a BaseModelOutputWithPooling across versions (confirmed
    empirically -- caused a crash in production);
  - the raw architectural outputs (logits, pred_boxes for OWL-ViT;
    text_model/vision_model submodule outputs + projection layers for CLIP)
    are the stable contract these models guarantee as part of their
    published architecture.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


# ============================================================================
# OWL-ViT: s_det and s_area
# ============================================================================

def owlvit_postprocess(logits: torch.Tensor, boxes: torch.Tensor) -> List[Tuple[float, float]]:
    """Pure tensor math: raw OWL-ViT outputs for ONE image ->
    [(s_det, s_area), ...] per query. sigmoid(logits), max over boxes."""
    num_queries = logits.shape[1]
    scores = torch.sigmoid(logits)
    results: List[Tuple[float, float]] = []
    for qi in range(num_queries):
        col = scores[:, qi]
        best_idx = int(torch.argmax(col).item())
        s_det = float(col[best_idx].item())
        w = float(max(boxes[best_idx, 2].item(), 0.0))
        h = float(max(boxes[best_idx, 3].item(), 0.0))
        results.append((s_det, float(w * h)))
    return results


class OwlViTScorer:
    """Zero-shot detector scorer: s_det and s_area."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = "cuda"):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _query_text(object_name: str) -> str:
        return f"a photo of a {object_name}"

    @torch.no_grad()
    def score_batch(self, image, object_names: Sequence[str]) -> List[Tuple[float, float]]:
        if len(object_names) == 0:
            return []
        queries = [self._query_text(o) for o in object_names]
        inputs = self.processor(text=[queries], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        boxes = outputs.pred_boxes[0]
        return owlvit_postprocess(logits, boxes)

    def score_map(self, image, object_names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
        scores = self.score_batch(image, object_names)
        return dict(zip(object_names, scores))


# ============================================================================
# CLIP: s_clip (cosine similarity)
# ============================================================================

def clip_cosine_similarities(image_embed: torch.Tensor, text_embeds: torch.Tensor) -> List[float]:
    """Pure tensor math: L2-normalized (1,D) image + (N,D) text -> N cosines."""
    sims = (image_embed @ text_embeds.T).squeeze(0)
    if sims.dim() == 0:
        return [float(sims.item())]
    return [float(s.item()) for s in sims]


class ClipScorer:
    """Image-text similarity scorer: s_clip.

    Computes CLIP embeddings by calling the base submodules directly:
        text_model -> text_projection   (text side)
        vision_model -> visual_projection  (image side)
    This bypasses get_text_features()/get_image_features(), whose return
    type has been observed to change across transformers versions (from a
    plain tensor to a BaseModelOutputWithPooling), causing silent
    correctness bugs (wrong embedding dimensionality) not just crashes.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _query_text(object_name: str) -> str:
        return f"a photo of a {object_name}"

    @torch.no_grad()
    def _image_embedding(self, image) -> torch.Tensor:
        inputs = self.processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        # Call the vision encoder submodule directly (NOT get_image_features)
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        # Index [1] = pooler_output: the stable BaseModelOutputWithPooling
        # contract (index 0 = last_hidden_state, index 1 = pooler_output)
        pooled = vision_out[1]
        # Project into the shared CLIP embedding space
        emb = self.model.visual_projection(pooled)  # (1, projection_dim)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def _text_embeddings(self, object_names: Sequence[str]) -> torch.Tensor:
        queries = [self._query_text(o) for o in object_names]
        inputs = self.processor(text=queries, return_tensors="pt", padding=True)
        # Filter to ONLY the keys text_model accepts (the processor may
        # return extra keys in some versions that text_model doesn't expect)
        text_keys = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
        text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k in text_keys}
        # Call the text encoder submodule directly (NOT get_text_features)
        text_out = self.model.text_model(**text_inputs)
        pooled = text_out[1]  # pooler_output
        emb = self.model.text_projection(pooled)  # (N, projection_dim)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def score_batch(self, image, object_names: Sequence[str]) -> List[float]:
        if len(object_names) == 0:
            return []
        image_emb = self._image_embedding(image)
        text_emb = self._text_embeddings(object_names)
        return clip_cosine_similarities(image_emb, text_emb)

    def score_map(self, image, object_names: Sequence[str]) -> Dict[str, float]:
        scores = self.score_batch(image, object_names)
        return dict(zip(object_names, scores))


# ============================================================================
# Combined 3D feature extractor (s_det, s_clip, s_area)
# ============================================================================

class FeatureExtractor:
    """Convenience wrapper combining OwlViTScorer + ClipScorer into the
    full 3D feature vector x_i = [s_det, s_clip, s_area]."""

    def __init__(
        self,
        owlvit_model: str = "google/owlvit-base-patch32",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        self.owlvit = OwlViTScorer(owlvit_model, device=device)
        self.clip = ClipScorer(clip_model, device=device)

    def extract(self, image, object_names: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
        if len(object_names) == 0:
            return {}
        det_area = self.owlvit.score_batch(image, object_names)
        clip_sims = self.clip.score_batch(image, object_names)
        out: Dict[str, Tuple[float, float, float]] = {}
        for name, (s_det, s_area), s_clip in zip(object_names, det_area, clip_sims):
            out[name] = (s_det, s_clip, s_area)
        return out
