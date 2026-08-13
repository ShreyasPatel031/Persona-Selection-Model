#!/usr/bin/env python3
"""Does activation separation along the good direction grow with steer alpha?

Batched generation (same path as Step C / OMP K-sweep): for each alpha, generate
all questions in one left-padded batch under neg system + alpha*v, then score
mean-assistant activations as h · v_hat.

Outputs:
  app/static/good_alpha_separation.json
  app/static/good_alpha_separation.html
  app/static/good_alpha_pca.json  (PCA coords per α, PC1–PC3 for slider viz)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.persona.activations import (  # noqa: E402
    load_model_and_tokenizer,
    mean_residuals_over_assistant,
)
from app.persona.response_style import with_paragraph_cap  # noqa: E402
from app.persona.schemas import PersonaTraitArtifact  # noqa: E402
from app.persona.steering_demo import (  # noqa: E402
    _language_model_layers,
    _steering_hook_fn,
)
from scripts.ssv_omp_k_sweep import generate_batched  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_PCA_DIMS = 3


def _pca_nd(X: np.ndarray, n: int = _PCA_DIMS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (m, d) -> explained-var ratios (n,), components (n, d), mean (d,)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    total = float(var.sum()) or 1.0
    k = min(n, len(S))
    ratios = (var[:k] / total).astype(np.float64)
    if k < n:
        ratios = np.pad(ratios, (0, n - k))
    comps = Vt[:k]
    if k < n:
        comps = np.pad(comps, ((0, n - k), (0, 0)))
    return ratios, comps.astype(np.float64), X.mean(axis=0).astype(np.float64)


def _project(h: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> tuple[float, ...]:
    proj = (h - mean) @ comps.T
    return tuple(float(x) for x in proj[: comps.shape[0]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="dnd_good_scale")
    ap.add_argument("--layer", type=int, default=15, help="Steer layer")
    ap.add_argument("--alphas", default="0,0.5,1.0,1.5,2.0,2.5,3.0,4.0")
    ap.add_argument("--n-questions", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--gen-batch-size", type=int, default=15,
                    help="Questions per generate() call (default: all)")
    ap.add_argument("--out-json", type=Path, default=REPO / "app/static/good_alpha_separation.json")
    ap.add_argument("--out-html", type=Path, default=REPO / "app/static/good_alpha_separation.html")
    ap.add_argument("--out-pca-json", type=Path, default=REPO / "app/static/good_alpha_pca.json")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    run_dir = REPO / "persona_runs" / args.run_id
    bundle = PersonaTraitArtifact.model_validate_json(
        (run_dir / "artifacts" / "trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = with_paragraph_cap(bundle.neg_system_prompt)
    questions = list(bundle.eval_questions[: args.n_questions])

    ck = torch.load(run_dir / "vectors" / "persona_vectors.pt", map_location="cpu", weights_only=False)
    v = ck["v"].float()[args.layer]
    v_hat = v / v.norm().clamp(min=1e-8)
    v_norm = float(v.norm())

    model, tok, dev = load_model_and_tokenizer()
    dtype = next(model.parameters()).dtype
    layers = _language_model_layers(model)
    layer_mod = layers[args.layer]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    direction = v.to(device=dev, dtype=dtype).view(1, 1, -1)

    logger.info(
        "layer=%d ||v||=%.2f n_q=%d alphas=%s batch=%d",
        args.layer, v_norm, len(questions), alphas, args.gen_batch_size,
    )

    behavior_by_alpha: dict[float, dict] = {}
    sweep_path = REPO / "app/static/layer3d_alpha_sweep.json"
    if sweep_path.is_file():
        sweep = json.loads(sweep_path.read_text())
        for row in (sweep.get("traits") or {}).get("good", {}).get("rows") or []:
            behavior_by_alpha[float(row["alpha"])] = {
                "mean_trait": row.get("mean_trait"),
                "mean_coherence": row.get("mean_coherence"),
            }

    rows = []
    baseline_mean: float | None = None
    all_h: dict[float, list[np.ndarray]] = {}

    for alpha in alphas:
        logger.info("α=%.1f — batched generate (%d questions)", alpha, len(questions))
        hook_calls = [0]
        hook = _steering_hook_fn(
            float(alpha), direction,
            steer_last_token_only=False,
            hook_calls=hook_calls,
        )
        replies = generate_batched(
            model, tok, neg_sys, questions, dev, pad_id,
            hook, layer_mod,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
        )
        if hook_calls[0] == 0 and abs(alpha) > 1e-12:
            raise RuntimeError("Steering hook never ran during batched generation.")

        scores: list[float] = []
        points: list[dict] = []
        hs: list[np.ndarray] = []
        for qi, (q, reply) in enumerate(zip(questions, replies)):
            h = mean_residuals_over_assistant(model, tok, dev, neg_sys, q, reply)
            h_l = h[args.layer].float().cpu().numpy()
            hs.append(h_l)
            s = float(h_l @ v_hat.cpu().numpy())
            scores.append(s)
            points.append({"q": qi, "score": round(s, 4), "reply_len": len(reply)})
        all_h[float(alpha)] = hs

        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        if baseline_mean is None:
            baseline_mean = mean_s
            sep = 0.0
        else:
            sep = mean_s - baseline_mean

        beh = behavior_by_alpha.get(alpha, {})
        row = {
            "alpha": alpha,
            "mean_score": round(mean_s, 4),
            "std_score": round(std_s, 4),
            "separation_from_alpha0": round(sep, 4),
            "n": len(scores),
            "points": points,
            "mean_trait_judge": beh.get("mean_trait"),
            "mean_coherence": beh.get("mean_coherence"),
        }
        rows.append(row)
        logger.info(
            "α=%.1f  mean(h·v̂)=%.1f  sep=%.1f  trait=%s  coh=%s",
            alpha, mean_s, sep, beh.get("mean_trait"), beh.get("mean_coherence"),
        )
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    doc = {
        "trait": "good",
        "run_id": args.run_id,
        "steer_layer": args.layer,
        "v_norm": round(v_norm, 4),
        "method": (
            "Batched generate under neg system + alpha*v at steer layer; "
            "score = mean_assistant_h[layer] · v_hat; "
            "separation = mean(score_alpha) - mean(score_0)"
        ),
        "n_questions": len(questions),
        "gen_batch_size": args.gen_batch_size,
        "alphas": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.out_json)

    html = _HTML.replace("__DATA__", json.dumps(doc))
    args.out_html.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", args.out_html)

    pca_doc = _build_pca_doc(
        alphas=alphas,
        rows=rows,
        all_h=all_h,
        steer_layer=args.layer,
        run_id=args.run_id,
        n_questions=len(questions),
        v_norm=v_norm,
    )
    args.out_pca_json.write_text(json.dumps(pca_doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.out_pca_json)
    return 0


def _build_pca_doc(
    *,
    alphas: list[float],
    rows: list[dict],
    all_h: dict[float, list[np.ndarray]],
    steer_layer: int,
    run_id: str,
    n_questions: int,
    v_norm: float,
) -> dict:
    """Fixed PCA basis from pooled (h_alpha - h_0) diffs; project baseline + each α."""
    alpha0 = float(alphas[0])
    h0 = all_h[alpha0]
    diffs = []
    for alpha in alphas[1:]:
        for qi in range(n_questions):
            diffs.append(all_h[float(alpha)][qi] - h0[qi])
    diffs_arr = np.stack(diffs, axis=0)
    ratios, comps, mean = _pca_nd(diffs_arr, _PCA_DIMS)

    def _pt_dict(qi: int, coords: tuple[float, ...]) -> dict:
        return {
            "q": qi,
            "pc1": round(coords[0], 4),
            "pc2": round(coords[1], 4),
            "pc3": round(coords[2], 4),
        }

    baseline_pts = []
    for qi, h in enumerate(h0):
        baseline_pts.append(_pt_dict(qi, _project(h, mean, comps)))

    alpha_docs = []
    all_pc = {f"pc{i}": [] for i in range(1, _PCA_DIMS + 1)}
    for pt in baseline_pts:
        for i in range(1, _PCA_DIMS + 1):
            all_pc[f"pc{i}"].append(pt[f"pc{i}"])

    for row in rows:
        alpha = float(row["alpha"])
        steered_pts = []
        for qi, h in enumerate(all_h[alpha]):
            coords = _project(h, mean, comps)
            steered_pts.append(_pt_dict(qi, coords))
            for i in range(1, _PCA_DIMS + 1):
                all_pc[f"pc{i}"].append(coords[i - 1])

        base_mean = np.array([p["pc1"] for p in baseline_pts], dtype=np.float64)
        steer_mean = np.array([p["pc1"] for p in steered_pts], dtype=np.float64)
        base_cent = np.array([
            [p["pc1"], p["pc2"], p["pc3"]] for p in baseline_pts
        ], dtype=np.float64).mean(axis=0)
        steer_cent = np.array([
            [p["pc1"], p["pc2"], p["pc3"]] for p in steered_pts
        ], dtype=np.float64).mean(axis=0)
        alpha_docs.append({
            "alpha": alpha,
            "separation_pc1": round(abs(float(steer_mean.mean()) - float(base_mean.mean())), 4),
            "separation_pc3d": round(float(np.linalg.norm(steer_cent - base_cent)), 4),
            "separation_h_vhat": row["separation_from_alpha0"],
            "mean_trait_judge": row.get("mean_trait_judge"),
            "mean_coherence": row.get("mean_coherence"),
            "steered": steered_pts,
        })

    pad = 0.1
    extent = {}
    for i in range(1, _PCA_DIMS + 1):
        key = f"pc{i}"
        lo, hi = min(all_pc[key]), max(all_pc[key])
        margin = (hi - lo) * pad or 1.0
        extent[key] = [round(lo - margin, 4), round(hi + margin, 4)]

    # Diagnostics: explain low PC2/PC3 vs layer PCA
    proj_diffs = (diffs_arr - mean) @ comps.T
    top3_var = (np.var(proj_diffs, axis=0, ddof=1)[:3])
    top3_norm = top3_var / (top3_var.sum() or 1.0)
    sep_rows = {float(r["alpha"]): r for r in rows}
    scores0 = {p["q"]: p["score"] for p in sep_rows[alpha0]["points"]}
    alphas_list, deltas = [], []
    for alpha in alphas[1:]:
        for p in sep_rows[float(alpha)]["points"]:
            alphas_list.append(alpha)
            deltas.append(p["score"] - scores0[p["q"]])
    corr_av = float(np.corrcoef(alphas_list, deltas)[0, 1]) if len(deltas) > 1 else 0.0

    return {
        "trait": "good",
        "run_id": run_id,
        "steer_layer": steer_layer,
        "v_norm": round(v_norm, 4),
        "n_questions": n_questions,
        "dims": _PCA_DIMS,
        "method": (
            "Fixed PCA (PC1–PC3) on pooled (h_alpha - h_0) diffs; "
            "project baseline (α=0) + steered activations"
        ),
        "pc1_var": round(float(ratios[0]), 4),
        "pc2_var": round(float(ratios[1]), 4),
        "pc3_var": round(float(ratios[2]), 4),
        "extent": extent,
        "baseline": baseline_pts,
        "alphas": alpha_docs,
        "pca_debug": {
            "fit_on": (
                f"{len(diffs_arr)} vectors of (h_alpha - h_0) in R^{diffs_arr.shape[1]}, "
                "one per (question, alpha>0)"
            ),
            "variance_full_dims": {
                "pc1": round(float(ratios[0]), 4),
                "pc2": round(float(ratios[1]), 4),
                "pc3": round(float(ratios[2]), 4),
                "tail_pc4_plus": round(float(1 - ratios[:3].sum()), 4),
                "note": "each PCi / sum(ALL singular values in full diff space)",
            },
            "variance_within_top3_subspace": {
                "pc1": round(float(top3_norm[0]), 4),
                "pc2": round(float(top3_norm[1]), 4),
                "pc3": round(float(top3_norm[2]), 4),
            },
            "pc2_over_pc1_std_ratio": round(
                float(proj_diffs[:, 1].std() / max(proj_diffs[:, 0].std(), 1e-9)), 4
            ),
            "corr_alpha_vs_delta_h_dot_vhat": round(corr_av, 4),
        },
    }


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Good — activation separation vs α</title>
<style>
  :root { font-family: system-ui, sans-serif; color: #111; background: #fafafa; }
  body { margin: 0; padding: 1.25rem; max-width: 920px; }
  h1 { font-size: 1.15rem; margin: 0 0 .35rem; }
  .sub { color: #555; font-size: .88rem; margin-bottom: 1rem; line-height: 1.45; }
  .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
  svg { width: 100%; height: auto; display: block; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td { border-bottom: 1px solid #eee; padding: .4rem .5rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #555; font-weight: 600; }
  .yes { color: #16a34a; font-weight: 600; }
  .no { color: #dc2626; }
</style>
</head>
<body>
<h1>Good trait — does activation separation grow with α?</h1>
<p class="sub" id="sub"></p>
<div class="card"><svg id="chart" viewBox="0 0 840 340"></svg></div>
<div class="card"><table id="tbl"></table></div>
<script>
const D = __DATA__;
document.getElementById("sub").innerHTML =
  `Steer layer <b>L${D.steer_layer}</b> · score = mean assistant activation · unit persona vector ` +
  `(‖v‖=${D.v_norm}). Separation = mean(score<sub>α</sub>) − mean(score<sub>0</sub>). ` +
  `Judge trait/coherence from existing alpha sweep (overlay). n=${D.n_questions} questions, batched.`;

const rows = D.alphas;
const W = 840, H = 340, m = { t: 24, r: 90, b: 40, l: 56 };
const iw = W - m.l - m.r, ih = H - m.t - m.b;
const xs = rows.map(r => r.alpha);
const seps = rows.map(r => r.separation_from_alpha0);
const x0 = Math.min(...xs), x1 = Math.max(...xs);
const y0 = Math.min(0, ...seps), y1 = Math.max(...seps, 1);
const yPad = (y1 - y0) * 0.08 || 1;
const sx = v => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
const sy = v => m.t + ih - ((v - (y0 - yPad)) / ((y1 + yPad) - (y0 - yPad))) * ih;
const sty = v => m.t + ih - (v / 100) * ih;

const svg = document.getElementById("chart");
function line(x1a,y1a,x2a,y2a, stroke, w=1, dash=null) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
  el.setAttribute("x1", x1a); el.setAttribute("y1", y1a);
  el.setAttribute("x2", x2a); el.setAttribute("y2", y2a);
  el.setAttribute("stroke", stroke); el.setAttribute("stroke-width", w);
  if (dash) el.setAttribute("stroke-dasharray", dash);
  svg.appendChild(el);
}
function text(str, x, y, opts={}) {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", x); t.setAttribute("y", y);
  t.setAttribute("font-size", opts.size || 11);
  t.setAttribute("fill", opts.fill || "#6b7280");
  if (opts.anchor) t.setAttribute("text-anchor", opts.anchor);
  t.textContent = str;
  svg.appendChild(t);
}
line(m.l, m.t, m.l, m.t + ih, "#e5e7eb");
line(m.l, m.t + ih, m.l + iw, m.t + ih, "#e5e7eb");
line(m.l + iw, m.t, m.l + iw, m.t + ih, "#e5e7eb");
if (y0 - yPad < 0 && y1 + yPad > 0) line(m.l, sy(0), m.l + iw, sy(0), "#d1d5db", 1, "4 3");

const traitPts = rows.filter(r => r.mean_trait_judge != null);
if (traitPts.length) {
  let d = "";
  traitPts.forEach((r, i) => {
    const x = sx(r.alpha), y = sty(r.mean_trait_judge);
    d += (i ? "L" : "M") + x + " " + y + " ";
  });
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", d); p.setAttribute("fill", "none");
  p.setAttribute("stroke", "#a78bfa"); p.setAttribute("stroke-width", 2);
  p.setAttribute("stroke-dasharray", "5 4");
  svg.appendChild(p);
  traitPts.forEach(r => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", sx(r.alpha)); c.setAttribute("cy", sty(r.mean_trait_judge));
    c.setAttribute("r", 3.5); c.setAttribute("fill", "#a78bfa");
    svg.appendChild(c);
  });
}

let d2 = "";
rows.forEach((r, i) => {
  d2 += (i ? "L" : "M") + sx(r.alpha) + " " + sy(r.separation_from_alpha0) + " ";
});
const p2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
p2.setAttribute("d", d2); p2.setAttribute("fill", "none");
p2.setAttribute("stroke", "#2563eb"); p2.setAttribute("stroke-width", 2.5);
svg.appendChild(p2);
rows.forEach(r => {
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", sx(r.alpha)); c.setAttribute("cy", sy(r.separation_from_alpha0));
  c.setAttribute("r", 5); c.setAttribute("fill", "#2563eb");
  svg.appendChild(c);
  line(sx(r.alpha), sy(r.separation_from_alpha0 - r.std_score),
       sx(r.alpha), sy(r.separation_from_alpha0 + r.std_score), "#93c5fd", 1.5);
});

text("α", m.l + iw / 2, H - 8, { anchor: "middle" });
text("activation sep", m.l + 8, m.t + 14, { fill: "#2563eb" });
text("judge trait (0–100)", m.l + iw - 8, m.t + 14, { anchor: "end", fill: "#a78bfa" });
xs.forEach(a => text(String(a), sx(a), m.t + ih + 16, { anchor: "middle", size: 10 }));

const tbl = document.getElementById("tbl");
tbl.innerHTML = `<thead><tr>
  <th>α</th><th>mean h·v̂</th><th>sep from α=0</th><th>std</th>
  <th>judge trait</th><th>coherence</th><th>sep ↑?</th>
</tr></thead><tbody></tbody>`;
const tb = tbl.querySelector("tbody");
rows.forEach((r, i) => {
  const up = i === 0 ? "—" : (r.separation_from_alpha0 > rows[i-1].separation_from_alpha0 ? "yes" : "no");
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${r.alpha}</td>
    <td>${r.mean_score.toFixed(1)}</td>
    <td><b>${r.separation_from_alpha0.toFixed(1)}</b></td>
    <td>${r.std_score.toFixed(1)}</td>
    <td>${r.mean_trait_judge ?? "—"}</td>
    <td>${r.mean_coherence ?? "—"}</td>
    <td class="${up === "yes" ? "yes" : (up === "no" ? "no" : "")}">${up}</td>`;
  tb.appendChild(tr);
});
const increasing = rows.every((r, i) => i === 0 || r.separation_from_alpha0 >= rows[i-1].separation_from_alpha0 - 1e-6);
const foot = document.createElement("p");
foot.className = "sub";
foot.innerHTML = increasing
  ? `<span class="yes">Monotonic:</span> separation increases with α over the full range tested.`
  : `<span class="no">Not fully monotonic:</span> separation does not strictly increase at every step (see table).`;
document.body.appendChild(foot);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
