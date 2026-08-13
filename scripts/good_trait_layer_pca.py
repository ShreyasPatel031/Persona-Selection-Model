#!/usr/bin/env python3
"""RepE-style PCA on good-trait pos/neg activations for selected layers.

Writes app/static/good_layer_pca.json + app/static/good_layer_pca.html
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.persona.activations import (  # noqa: E402
    iter_kept_rollouts,
    load_model_and_tokenizer,
    mean_residuals_over_assistant,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Main steer layers for good + a few context layers
DEFAULT_LAYERS = [8, 12, 15, 16, 20, 28]


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (n, d) -> projections (n, 2), explained var ratios (2,), components (2, d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    total = float(var.sum()) or 1.0
    ratios = (var[:2] / total).astype(np.float64)
    comps = Vt[:2]
    proj = Xc @ comps.T
    return proj.astype(np.float64), ratios, comps.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="dnd_good")
    ap.add_argument("--max-per-arm", type=int, default=80)
    ap.add_argument("--layers", default=",".join(str(l) for l in DEFAULT_LAYERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=REPO / "app/static/good_layer_pca.json")
    ap.add_argument("--out-html", type=Path, default=REPO / "app/static/good_layer_pca.html")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    run_dir = REPO / "persona_runs" / args.run_id
    rollouts = run_dir / "rollouts" / "rollouts.jsonl"
    if not rollouts.is_file():
        raise SystemExit(f"missing {rollouts}")

    pos_rows, neg_rows = [], []
    for o in iter_kept_rollouts(rollouts):
        if o.get("arm") == "pos":
            pos_rows.append(o)
        elif o.get("arm") == "neg":
            neg_rows.append(o)

    rng = random.Random(args.seed)
    if len(pos_rows) > args.max_per_arm:
        pos_rows = rng.sample(pos_rows, args.max_per_arm)
    if len(neg_rows) > args.max_per_arm:
        neg_rows = rng.sample(neg_rows, args.max_per_arm)
    logger.info("Using %d pos + %d neg", len(pos_rows), len(neg_rows))

    model, tok, dev = load_model_and_tokenizer()
    logger.info("device=%s", dev)

    def collect(rows: list[dict]) -> dict[int, list[np.ndarray]]:
        by_layer: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        for i, o in enumerate(rows):
            logger.info("Forward %d/%d (%s)", i + 1, len(rows), o.get("arm"))
            m = mean_residuals_over_assistant(
                model, tok, dev, o["system"], o["question"], o["assistant_a"]
            )  # (n_layers, d)
            for l in layers:
                by_layer[l].append(m[l].float().cpu().numpy())
            if dev.type == "cuda" and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        return by_layer

    pos_by = collect(pos_rows)
    neg_by = collect(neg_rows)

    # Fit PCA on difference vectors (RepE LAT), project all points
    layer_docs = []
    for l in layers:
        pos = np.stack(pos_by[l], axis=0)
        neg = np.stack(neg_by[l], axis=0)
        n = min(len(pos), len(neg))
        diffs = pos[:n] - neg[:n]
        # also include unpaired if counts differ
        _, ratios, comps = _pca_2d(diffs)

        # project all activations into PC space of diffs
        all_X = np.concatenate([pos, neg], axis=0)
        mean = diffs.mean(axis=0)
        proj = (all_X - mean) @ comps.T

        # separation: mean PC1 pos - mean PC1 neg
        pc1_pos = float(proj[: len(pos), 0].mean())
        pc1_neg = float(proj[len(pos) :, 0].mean())
        sep = abs(pc1_pos - pc1_neg)

        points = []
        for i in range(len(pos)):
            points.append({"arm": "pos", "pc1": round(float(proj[i, 0]), 4), "pc2": round(float(proj[i, 1]), 4)})
        for i in range(len(neg)):
            j = len(pos) + i
            points.append({"arm": "neg", "pc1": round(float(proj[j, 0]), 4), "pc2": round(float(proj[j, 1]), 4)})

        layer_docs.append({
            "layer": l,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "pc1_var": round(float(ratios[0]), 4),
            "pc2_var": round(float(ratios[1]), 4),
            "pc1_mean_pos": round(pc1_pos, 4),
            "pc1_mean_neg": round(pc1_neg, 4),
            "separation_pc1": round(sep, 4),
            "points": points,
        })
        logger.info(
            "L%02d  PC1=%.1f%% PC2=%.1f%% sep=%.2f",
            l, 100 * ratios[0], 100 * ratios[1], sep,
        )

    doc = {
        "trait": "good",
        "run_id": args.run_id,
        "method": "PCA on pos-neg difference vectors (RepE LAT), project all activations",
        "main_layers": [15, 16],
        "layers": layer_docs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc), encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(doc))
    args.out_html.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", args.out_html)
    return 0


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Good trait — layer PCA (RepE LAT)</title>
<style>
  :root { font-family: system-ui, sans-serif; color: #111; background: #fafafa; }
  body { margin: 0; padding: 1rem 1.25rem 2rem; }
  h1 { font-size: 1.15rem; margin: 0 0 .25rem; }
  .sub { color: #555; font-size: .85rem; margin-bottom: 1rem; max-width: 52rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
  .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: .75rem; }
  .card.main { border-color: #7c3aed; box-shadow: 0 0 0 2px rgba(124,58,237,.15); }
  .card h2 { font-size: .95rem; margin: 0 0 .35rem; display: flex; justify-content: space-between; }
  .badge { font-size: .7rem; background: #7c3aed; color: #fff; padding: .1rem .4rem; border-radius: 4px; }
  .meta { font-size: .75rem; color: #555; margin-bottom: .4rem; }
  svg { width: 100%; height: auto; display: block; background: #fcfcfd; border-radius: 6px; }
  .leg { display: flex; gap: 1rem; font-size: .8rem; margin: .5rem 0 1rem; }
  .leg span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .leg .pos::before { background: #16a34a; }
  .leg .neg::before { background: #dc2626; }
</style>
</head>
<body>
<h1>Good trait — PCA on activations (RepE LAT)</h1>
<p class="sub">
  Per layer: PCA fit on <code>h<sub>pos</sub> − h<sub>neg</sub></code> difference vectors,
  then all rollout activations projected onto PC1–PC2.
  <strong>PC1</strong> = main good↔not-good axis. Highlighted cards = main steer layers (L15, L16).
</p>
<div class="leg"><span class="pos">pos (good)</span><span class="neg">neg (baseline)</span></div>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA_JSON__;
const main = new Set(DATA.main_layers || [15, 16]);

function extent(vals, pad = 0.08) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const m = (hi - lo) * pad;
  return [lo - m, hi + m];
}

function drawCard(layerDoc) {
  const card = document.createElement("div");
  card.className = "card" + (main.has(layerDoc.layer) ? " main" : "");
  const title = document.createElement("h2");
  title.innerHTML = `Layer ${layerDoc.layer}` + (main.has(layerDoc.layer) ? ' <span class="badge">main</span>' : "");
  card.appendChild(title);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent =
    `PC1 ${ (layerDoc.pc1_var * 100).toFixed(1) }% · PC2 ${ (layerDoc.pc2_var * 100).toFixed(1) }% · ` +
    `sep |ΔPC1| = ${layerDoc.separation_pc1.toFixed(2)} · n=${layerDoc.n_pos}+${layerDoc.n_neg}`;
  card.appendChild(meta);

  const W = 320, H = 260, m = { t: 12, r: 12, b: 28, l: 28 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = layerDoc.points.map(p => p.pc1);
  const ys = layerDoc.points.map(p => p.pc2);
  const [x0, x1] = extent(xs);
  const [y0, y1] = extent(ys);
  const sx = v => m.l + ((v - x0) / (x1 - x0)) * iw;
  const sy = v => m.t + (1 - (v - y0) / (y1 - y0)) * ih;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // axes
  const axis = (x1a, y1a, x2a, y2a) => {
    const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
    l.setAttribute("x1", x1a); l.setAttribute("y1", y1a);
    l.setAttribute("x2", x2a); l.setAttribute("y2", y2a);
    l.setAttribute("stroke", "#e5e7eb");
    svg.appendChild(l);
  };
  axis(m.l, m.t + ih, m.l + iw, m.t + ih);
  axis(m.l, m.t, m.l, m.t + ih);
  // zero lines if in range
  if (x0 < 0 && x1 > 0) axis(sx(0), m.t, sx(0), m.t + ih);
  if (y0 < 0 && y1 > 0) axis(m.l, sy(0), m.l + iw, sy(0));

  for (const p of layerDoc.points) {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", sx(p.pc1));
    c.setAttribute("cy", sy(p.pc2));
    c.setAttribute("r", 3.2);
    c.setAttribute("fill", p.arm === "pos" ? "#16a34a" : "#dc2626");
    c.setAttribute("fill-opacity", "0.75");
    svg.appendChild(c);
  }

  const label = (text, x, y) => {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#6b7280");
    t.textContent = text;
    svg.appendChild(t);
  };
  label("PC1 →", m.l + iw - 36, m.t + ih + 18);
  label("PC2", 4, m.t + 10);

  card.appendChild(svg);
  return card;
}

const grid = document.getElementById("grid");
// main layers first
const ordered = [...DATA.layers].sort((a, b) => {
  const am = main.has(a.layer) ? 0 : 1;
  const bm = main.has(b.layer) ? 0 : 1;
  if (am !== bm) return am - bm;
  return a.layer - b.layer;
});
for (const L of ordered) grid.appendChild(drawCard(L));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
