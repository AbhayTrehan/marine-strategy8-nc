"""
report_nc.py
============

Strategy 8-U-NC's counterpart to report.py: builds an HTML visual report
for the null-calibrated conformal sorter, showing -- per image, with NO
ground-truth annotations (this is a sanity check of the METHOD's own
internal behavior, not an accuracy evaluation against labels) --

  (a) base unguided VLM response (Pass-1 caption)
  (b)-(d) raw object mentions by source (RAM++ / DETR / VLM)
  (e) the O_init canonical union (synonyms.py, unchanged)
  (f) each candidate's conformal p-value (Eq. 12) and signed distance
      (Eq. 11), plus its O_pos/O_neg status at EVERY epsilon in
      `epsilons` (so a single report can show, e.g., epsilon=0.05 vs
      epsilon=0.2 side by side for the same candidate, instead of
      duplicating the whole report per epsilon)
  (g) the probe pool summary for that image (K, tau_low, null model mean/
      shrinkage) -- there is no per-probe table by default (K is 50-100,
      too many rows to be useful inline), but the count and null-model
      diagnostics are shown so a reader can sanity-check e.g. "were there
      really 80 probes, did the shrinkage look reasonable"
  (h) the c_pos/c_neg guidance prompts actually used (built at the FIRST
      epsilon in `epsilons`, since prompts.py only needs one O_pos/O_neg
      split -- if the caller wants prompts at a different epsilon they can
      pass a different question_file built at that epsilon)
  (i) the final response after tri-state contrastive decoding, if available

Everything here is assembled from artifacts fit_null_calibration.py and
candidate_pool.py have ALREADY produced -- this module makes no model
calls of its own, mirroring report.py's own design.
"""

from __future__ import annotations

import html
import json
import os
from typing import Dict, List, Optional

from report import get_base64_image  # reuse verbatim, no need to duplicate


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _badge_list(items, css_class: str) -> str:
    if not items:
        return '<span class="empty-note">none</span>'
    return "".join(f'<span class="obj-badge {css_class}">{_esc(o)}</span>' for o in items)


def build_candidate_table_nc(
    candidates: List[dict],
    p_values: Dict[str, float],
    signed_distances: Dict[str, float],
    epsilons: List[float],
) -> str:
    """(e)-(f): one row per candidate, showing its raw evidence, conformal
    p-value, signed distance, and a POSITIVE/HALLUCINATED badge for every
    epsilon in `epsilons` (Eq. 13-14 applied at each threshold)."""
    if not candidates:
        return "<p class='no-objects'>No candidate objects in O_init for this image.</p>"

    def _p_of(c):
        p = p_values.get(c["canonical"])
        return p if p is not None else 1.0  # unseen candidate treated as never-verified

    ordered = sorted(candidates, key=lambda c: _p_of(c))

    epsilon_headers = "".join(f"<th>&epsilon;={e:g}</th>" for e in epsilons)

    rows = ""
    for c in ordered:
        canonical = c["canonical"]
        p = p_values.get(canonical)
        d = signed_distances.get(canonical)
        is_pos_at_smallest_eps = p is not None and epsilons and p <= epsilons[0]
        row_class = "pos-row" if is_pos_at_smallest_eps else "neg-row"

        sources_str = ", ".join(sorted(c.get("sources", [])))
        raw_str = ", ".join(c.get("raw_mentions", []))
        p_str = f"{p:.4f}" if p is not None else "N/A"
        d_str = ("-inf" if d == float("-inf") else f"{d:.3f}") if d is not None else "N/A"

        eps_cells = ""
        for eps in epsilons:
            if p is None:
                badge = '<span class="badge badge-na">N/A</span>'
            elif p <= eps:
                badge = '<span class="badge badge-pos">POSITIVE</span>'
            else:
                badge = '<span class="badge badge-neg">HALLUCINATED</span>'
            eps_cells += f"<td>{badge}</td>"

        rows += f"""
        <tr class="{row_class}">
            <td><strong>{_esc(canonical)}</strong></td>
            <td>{_esc(sources_str)}</td>
            <td class="raw-col">{_esc(raw_str)}</td>
            <td>{c.get('s_det', 0.0):.3f}</td>
            <td>{c.get('s_clip', 0.0):.3f}</td>
            <td>{c.get('s_area', 0.0):.3f}</td>
            <td class="dist-col">{d_str}</td>
            <td class="pval-col">{p_str}</td>
            {eps_cells}
        </tr>
        """
    return f"""
    <table class="score-table">
        <thead>
        <tr>
            <th>Canonical Object</th>
            <th>Sources</th>
            <th>Raw Mentions</th>
            <th>s_det (OWL-ViT)</th>
            <th>s_clip (CLIP)</th>
            <th>s_area</th>
            <th>D(w) signed dist.</th>
            <th>p-value</th>
            {epsilon_headers}
        </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def build_probe_summary_html(probe_rec: Optional[dict], sort_result_dict: Optional[dict]) -> str:
    if probe_rec is None:
        return "<p class='no-objects'>No probe pool cached for this image.</p>"

    K = probe_rec.get("K", len(probe_rec.get("probes", [])))
    tau_low = probe_rec.get("tau_low")
    lines = [f"<b>K (probe count)</b>: {K}"]
    if tau_low is not None:
        lines.append(f"<b>&tau;<sub>low</sub></b>: {tau_low}")

    if sort_result_dict is not None:
        null_model = sort_result_dict.get("null_model", {})
        mean = null_model.get("mean")
        shrinkage = null_model.get("shrinkage")
        if mean is not None:
            mean_str = ", ".join(f"{m:.3f}" for m in mean)
            lines.append(f"<b>&mu;<sub>0</sub></b> (post-normalization, ~0 expected): [{mean_str}]")
        if shrinkage is not None:
            lines.append(f"<b>Ledoit-Wolf &lambda;</b>: {shrinkage:.3f}")

    return "<div class='probe-summary'>" + " &nbsp;|&nbsp; ".join(lines) + "</div>"


def build_image_card_nc(
    idx: int,
    img: str,
    pool: dict,
    probe_rec: Optional[dict],
    sort_result_dict: Optional[dict],
    question_entry: Optional[dict],
    final_response: str,
    epsilons: List[float],
    image_dir: str,
) -> str:
    convs = {c["from"]: c["value"] for c in question_entry["conversations"]} if question_entry else {}
    c_pos_text = convs.get("guidance_pos", "")
    c_neg_text = convs.get("guidance_neg", "")

    p_values = dict(zip(
        sort_result_dict.get("candidate_names", []) if sort_result_dict else [],
        sort_result_dict.get("candidate_p_values", []) if sort_result_dict else [],
    ))
    signed_distances = dict(zip(
        sort_result_dict.get("candidate_names", []) if sort_result_dict else [],
        sort_result_dict.get("candidate_signed_distances", []) if sort_result_dict else [],
    ))

    candidates = pool.get("candidates", [])
    raw = pool.get("raw", {"ram": [], "detr": [], "vlm": []})

    n_pos_smallest_eps = sum(1 for c in candidates if p_values.get(c["canonical"], 1.0) <= (epsilons[0] if epsilons else 0.1))
    n_neg_smallest_eps = len(candidates) - n_pos_smallest_eps

    candidate_table_html = build_candidate_table_nc(candidates, p_values, signed_distances, epsilons)
    probe_summary_html = build_probe_summary_html(probe_rec, sort_result_dict)
    img_b64 = get_base64_image(os.path.join(image_dir, img))

    return f"""  <div class="card">
    <div class="card-header">
      <span>Image {idx} &nbsp;|&nbsp; {_esc(img)}</span>
      <span>O_init: {len(candidates)} &nbsp;|&nbsp;
            <span style="color:#166534;font-weight:700;">O_pos (&epsilon;={epsilons[0] if epsilons else '?':g}): {n_pos_smallest_eps}</span> &nbsp;|&nbsp;
            <span style="color:#991b1b;font-weight:700;">O_neg: {n_neg_smallest_eps}</span></span>
    </div>
    <div class="card-body">
      <div class="image-container">
        <img src="{img_b64}" alt="{_esc(img)}">
      </div>
      <div class="content">

        <div class="section-title">(a) Base LVLM response &mdash; unguided Pass-1 caption (y&#8201;<sup>(1)</sup>)</div>
        <div class="text-box original">{_esc(pool.get('pass1_caption', ''))}</div>

        <div class="section-title">(b)&ndash;(d) Raw object mentions by source</div>
        <div class="source-row"><span class="source-label src-ram">RAM++</span> {_badge_list(raw.get('ram', []), 'badge-ram')}</div>
        <div class="source-row"><span class="source-label src-detr">DETR</span> {_badge_list(raw.get('detr', []), 'badge-detr')}</div>
        <div class="source-row"><span class="source-label src-vlm">VLM (Pass-1)</span> {_badge_list(raw.get('vlm', []), 'badge-vlm')}</div>

        <div class="section-title">(e) Guaranteed-absent probe pool (Section 3.1)</div>
        {probe_summary_html}

        <div class="section-title">(f) O_init: conformal p-values, signed distances, and O_pos/O_neg at each &epsilon;</div>
        {candidate_table_html}

        <div class="section-title">(g) Guidance prompts used at &epsilon;={epsilons[0] if epsilons else '?':g}: c_pos / c_neg</div>
        <div class="text-box prompt-pos"><span class="prompt-label">c_pos</span>{_esc(c_pos_text)}</div>
        <div class="text-box prompt-neg"><span class="prompt-label">c_neg</span>{_esc(c_neg_text)}</div>

        <div class="section-title">(h) Final response after tri-state contrastive decoding</div>
        <div class="text-box corrected">{_esc(final_response)}</div>

      </div>
    </div>
  </div>
"""


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f0f2f5;
    color: #2d2d2d;
    margin: 0;
    padding: 20px;
  }}
  h1 {{ text-align: center; color: #1a1a2e; margin-bottom: 6px; font-size: 24px; }}
  .subtitle {{ text-align: center; color: #555; margin-bottom: 24px; font-size: 14px; }}
  .container {{ max-width: 1500px; margin: 0 auto; }}

  .banner {{
    background: #1a1a2e; color: white; padding: 20px 28px; border-radius: 10px;
    margin-bottom: 28px; box-shadow: 0 4px 14px rgba(0,0,0,.18);
  }}
  .banner h2 {{ margin: 0 0 8px 0; font-size: 16px; color: #b0b8d8; letter-spacing: .3px; }}
  .banner .meta {{ font-size: 13px; color: #9ea8cc; }}
  .banner .meta b {{ color: #e8ecff; }}
  .banner .note {{ font-size: 12px; color: #f0c419; margin-top: 8px; }}

  .card {{
    background: #fff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.09);
    margin-bottom: 28px; overflow: hidden;
  }}
  .card-header {{
    background: #f7f8fa; padding: 12px 20px; border-bottom: 1px solid #e5e7eb;
    font-size: 13px; color: #666; display: flex; justify-content: space-between; align-items: center;
  }}
  .card-body {{ display: flex; flex-wrap: wrap; }}
  .image-container {{
    flex: 0 0 320px; background: #111; display: flex; align-items: center;
    justify-content: center; padding: 12px; min-height: 240px;
  }}
  .image-container img {{ max-width: 100%; max-height: 300px; object-fit: contain; border-radius: 4px; }}
  .content {{ flex: 1; padding: 20px 24px; min-width: 0; }}

  .section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px;
    color: #888; margin: 18px 0 6px; padding-bottom: 4px; border-bottom: 1px solid #eee;
  }}
  .section-title:first-child {{ margin-top: 0; }}

  .text-box {{
    padding: 12px 14px; border-left: 4px solid #3b82f6; border-radius: 0 6px 6px 0;
    background: #fafbff; font-size: 14.5px; line-height: 1.55;
  }}
  .text-box.original {{ border-left-color: #6b7280; }}
  .text-box.corrected {{ border-left-color: #22c55e; background: #f0fdf4; font-weight: 500; }}
  .text-box.prompt-pos {{ border-left-color: #16a34a; background: #f0fdf4; margin-bottom: 8px; }}
  .text-box.prompt-neg {{ border-left-color: #dc2626; background: #fef2f2; }}
  .prompt-label {{
    display: inline-block; font-family: monospace; font-weight: 700; font-size: 12px;
    color: #555; margin-right: 8px; background: #eee; padding: 1px 6px; border-radius: 4px;
  }}

  .probe-summary {{
    padding: 10px 14px; border-left: 4px solid #a855f7; border-radius: 0 6px 6px 0;
    background: #faf5ff; font-size: 13px; line-height: 1.7;
  }}

  .source-row {{ margin: 4px 0; font-size: 13.5px; line-height: 1.9; }}
  .source-label {{
    display: inline-block; min-width: 84px; font-weight: 700; font-size: 11px;
    text-transform: uppercase; letter-spacing: .3px; color: #555; margin-right: 6px;
  }}
  .obj-badge {{
    display: inline-block; padding: 2px 9px; margin: 2px 3px 2px 0; border-radius: 12px;
    font-size: 12px; font-family: monospace; border: 1px solid transparent;
  }}
  .badge-ram {{ background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }}
  .badge-detr {{ background: #fdf4ff; color: #a21caf; border-color: #f5d0fe; }}
  .badge-vlm {{ background: #fff7ed; color: #c2410c; border-color: #fed7aa; }}
  .empty-note {{ color: #9ca3af; font-style: italic; font-size: 12.5px; }}

  .score-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; font-family: monospace; }}
  .score-table thead th {{
    background: #f1f5f9; color: #374151; font-weight: 700; padding: 8px 6px;
    text-align: center; border: 1px solid #e2e8f0; white-space: nowrap;
  }}
  .score-table td {{ border: 1px solid #e2e8f0; padding: 6px 6px; text-align: center; vertical-align: middle; }}
  .score-table .raw-col {{ text-align: left; max-width: 200px; }}
  .score-table .dist-col, .score-table .pval-col {{ font-weight: 700; color: #7e22ce; }}
  .pos-row {{ background: #f7fef7; }}
  .neg-row {{ background: #fff5f5; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 10.5px;
    font-weight: 700; letter-spacing: .2px; white-space: nowrap;
  }}
  .badge-pos {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
  .badge-neg {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
  .badge-na {{ background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; }}

  .no-objects {{ color: #9ca3af; font-style: italic; font-size: 13px; margin-top: 10px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Strategy 8-U-NC: Null-Calibrated Conformal Sorter</h1>
  <p class="subtitle">Visual sanity-check report (no ground-truth annotations) &middot; showing {n_shown} of {n_requested} requested images</p>

  <div class="banner">
    <h2>Pipeline configuration used for this report</h2>
    <div class="meta">{config_html}</div>
    <div class="note">This report shows only what the method itself produced (candidate pool, probe pool,
    conformal p-values, O_pos/O_neg at each &epsilon;) -- it deliberately does NOT use CHAIR/COCO ground-truth
    labels. For a real/hallucinated accuracy check against ground truth, see the companion histogram report.</div>
  </div>

{cards}
</div>
</body>
</html>
"""


def generate_report_nc(
    report_images: List[str],
    candidate_pool_cache: Dict[str, dict],
    probe_pool_cache: Dict[str, dict],
    sort_results_file: str,
    epsilons: List[float],
    image_dir: str,
    output_path: str,
    question_file: Optional[str] = None,
    answers_file: Optional[str] = None,
    config_info: dict = None,
    title: str = "Strategy 8-U-NC Visual Report",
) -> str:
    """Assembles the HTML report from already-produced pipeline artifacts.
    Returns output_path for convenience.

    `epsilons` should be given smallest-first (e.g. [0.05, 0.2]) -- the
    card header's O_pos/O_neg summary counts and the c_pos/c_neg prompts
    shown are computed at epsilons[0].
    """
    if not epsilons:
        raise ValueError("epsilons must be a non-empty list")
    epsilons = sorted(epsilons)

    sort_results = _load_json(sort_results_file)

    questions = _load_json(question_file) if question_file else []
    answers = _load_jsonl(answers_file) if answers_file else []
    question_by_image = {q["image"]: q for q in questions}
    answer_by_qid = {a["question_id"]: a for a in answers}

    cards_html = []
    n_shown = 0
    for img in report_images:
        pool = candidate_pool_cache.get(img)
        if pool is None:
            continue
        probe_rec = probe_pool_cache.get(img)
        sort_result_dict = sort_results.get(img)
        q = question_by_image.get(img)
        ans = answer_by_qid.get(q["id"]) if q else None
        final_response = ans["text"] if ans else "N/A (no generation found for this image/question)"

        n_shown += 1
        cards_html.append(build_image_card_nc(
            n_shown, img, pool, probe_rec, sort_result_dict, q, final_response, epsilons, image_dir,
        ))

    config_info = config_info or {}
    config_bits = " &nbsp;|&nbsp; ".join(f"<b>{_esc(k)}</b>: {_esc(v)}" for k, v in config_info.items())
    if not config_bits:
        config_bits = "(no config metadata supplied)"

    html_out = _PAGE_TEMPLATE.format(
        title=title,
        n_shown=n_shown,
        n_requested=len(report_images),
        config_html=config_bits,
        cards="\n".join(cards_html),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    return output_path
