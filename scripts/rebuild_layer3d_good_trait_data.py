#!/usr/bin/env python3
"""
Build layer3d trait-test activation JSON from real Good steering eval data.

Sources (in priority order):
  1. export_good_trait_layer_activations.py output (full 34×2560 hidden deltas)
  2. app/static/sae_alpha_viz_data.json — SAE feature deltas @ α=1.5 vs α=0,
     projected onto L16 residual dims via SAE decoder W_dec

Usage:
  python scripts/rebuild_layer3d_good_trait_data.py
  python scripts/rebuild_layer3d_good_trait_data.py --from-export path/to/export.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VIZ = REPO / "app/static/sae_alpha_viz_data.json"
DEFAULT_EXPORT = REPO / "app/static/layer3d_good_trait_activation_export.json"
DEFAULT_TRAIT_SCORES = REPO / "app/static/good_layer_trait_scores.json"
DEFAULT_VECTORS = REPO / "persona_runs/dnd_good/vectors/persona_vectors.pt"
HIDDEN_DIM = 2560
NUM_LAYERS = 34

TRAIT_BUILDS: dict[str, dict] = {
    "good": {
        "vectors": REPO / "persona_runs/dnd_good/vectors/persona_vectors.pt",
        "trait_scores": REPO / "app/static/good_layer_trait_scores.json",
        "out": REPO / "app/static/layer3d_good_trait_activation.json",
    },
    "evil": {
        "vectors": REPO / "persona_runs/dnd_evil/vectors/persona_vectors.pt",
        "trait_scores": REPO / "app/static/evil_layer_trait_scores.json",
        "out": REPO / "app/static/layer3d_evil_trait_activation.json",
    },
    "lawful": {
        "vectors": REPO / "persona_runs/dnd_lawful/vectors/persona_vectors.pt",
        "trait_scores": REPO / "app/static/lawful_layer_trait_scores.json",
        "out": REPO / "app/static/layer3d_lawful_trait_activation.json",
    },
    "chaotic": {
        "vectors": REPO / "persona_runs/dnd_chaotic/vectors/persona_vectors.pt",
        "trait_scores": REPO / "app/static/chaotic_layer_trait_scores.json",
        "out": REPO / "app/static/layer3d_chaotic_trait_activation.json",
    },
}


def _normalize_layer(vec: torch.Tensor, top_frac: float = 0.35) -> list[float]:
    """Top-|Δ| dims ranked for visibility (rest = 0 in viz JSON)."""
    v = vec.float().abs()
    k = max(1, int((1.0 - top_frac) * v.numel()))
    thresh = float(torch.kthvalue(v, k).values)
    out = torch.zeros_like(v)
    mask = v >= thresh
    if not bool(mask.any()):
        return [0.0] * int(v.numel())
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    vals = v[idx]
    order = torch.argsort(vals)
    n = len(order)
    ranks = torch.linspace(0.35, 1.0, n)
    out[idx[order]] = ranks
    return [round(x, 5) for x in out.tolist()]


def from_hidden_export(export_path: Path) -> dict:
    doc = json.loads(export_path.read_text(encoding="utf-8"))
    layers_out = []
    for row in doc["layers"]:
        l = int(row["layer"])
        if row.get("activation"):
            dims = row["activation"]
        else:
            dims = _normalize_layer(torch.tensor(row["delta_abs"]))
        layers_out.append({"layer": l, "activation": dims})
    return {
        "trait": doc.get("trait", "good"),
        "run_id": doc.get("run_id", ""),
        "source": doc.get("source", "trait_test_hidden_export"),
        "steer_alpha": doc.get("steer_alpha", 1.5),
        "steer_layer": doc.get("steer_layer", 16),
        "n_questions": doc.get("n_questions"),
        "layers": layers_out,
    }


def _feature_deltas_from_viz(viz: dict, steer_alpha: float) -> dict[int, float]:
    """SAE feature Δz = mean(z@α) − mean(z@0) from trait steering eval."""
    by_top = viz.get("by_alpha_top") or {}
    key0 = "0"
    key_a = f"{steer_alpha:g}"
    if key_a not in by_top:
        raise KeyError(f"Alpha {key_a} not in by_alpha_top")

    by0 = {int(r["feature_id"]): float(r["mean_activation"]) for r in by_top[key0]}
    by_a = {int(r["feature_id"]): float(r["mean_activation"]) for r in by_top[key_a]}
    all_fids = set(by0) | set(by_a)

    # Trajectories include more features (when both alphas were measured).
    for traj in (viz.get("trajectories") or {}).values():
        fid = int(traj["feature_id"])
        series = {float(pt["alpha"]): pt for pt in traj["series"]}
        pt0 = series.get(0.0)
        pt_a = series.get(steer_alpha)
        if pt0 and pt_a and pt0.get("mean_activation") is not None and pt_a.get("mean_activation") is not None:
            all_fids.add(fid)
            by0[fid] = float(pt0["mean_activation"])
            by_a[fid] = float(pt_a["mean_activation"])

    deltas: dict[int, float] = {}
    for fid in all_fids:
        deltas[fid] = by_a.get(fid, 0.0) - by0.get(fid, 0.0)
    return deltas


def from_sae_alpha_viz(viz_path: Path, steer_alpha: float = 1.5) -> dict:
    from app.phase2 import load_sae_for_layer
    from app.persona.sae_common import _get_decoder_columns

    viz = json.loads(viz_path.read_text(encoding="utf-8"))
    meta = viz.get("meta") or {}
    layer = int(meta.get("layer", 16))
    sae_id = meta.get("sae_id", "layer_16_width_16k_l0_small")
    run_id = meta.get("run_id", "dnd_good_scale")

    deltas = _feature_deltas_from_viz(viz, steer_alpha)
    logger.info(
        "Trait-test SAE deltas: %d features with α=%s vs baseline (from %s)",
        len(deltas),
        steer_alpha,
        viz_path.name,
    )

    sae, _ = load_sae_for_layer(
        torch.device("cpu"),
        release="gemma-scope-2-4b-it-res-all",
        sae_id=sae_id,
        hidden_state_index=layer + 1,
    )
    w_dec = _get_decoder_columns(sae)  # (d_sae, d_in)
    delta_z = torch.zeros(w_dec.shape[0], dtype=torch.float32)
    for fid, d in deltas.items():
        if 0 <= fid < delta_z.shape[0]:
            delta_z[fid] = d

    # Residual shift ≈ Σ_i Δz_i · W_dec[i]
    h_delta = delta_z @ w_dec
    if h_delta.shape[0] != HIDDEN_DIM:
        raise ValueError(f"Expected d_in={HIDDEN_DIM}, got {h_delta.shape[0]}")

    act_l16 = _normalize_layer(h_delta)
    layers_out = []
    for l in range(NUM_LAYERS):
        if l == layer:
            layers_out.append({"layer": l, "activation": act_l16})
        else:
            layers_out.append({"layer": l, "activation": [0.0] * HIDDEN_DIM})

    return {
        "trait": "good",
        "run_id": run_id,
        "source": "trait_test_sae_alpha_viz_decoder_projection",
        "steer_alpha": steer_alpha,
        "steer_layer": layer,
        "sae_id": sae_id,
        "n_features_in_delta": len(deltas),
        "n_questions": meta.get("n_questions"),
        "viz_source": str(viz_path.resolve()),
        "note": (
            f"L{layer} only: |Σ Δz_i W_dec[i]| from Good trait eval "
            f"(neg system prompt, steer α={steer_alpha}). "
            "Other layers need export_good_trait_layer_activations.py."
        ),
        "layers": layers_out,
    }


def _load_layer_trait_scores(path: Path) -> dict[int, float]:
    """Causal trait judge scores when steering at layer L (0–100)."""
    scores: dict[int, float] = {}
    sweep = REPO / "persona_runs/dnd_layer_sweep.json"
    if sweep.is_file():
        doc = json.loads(sweep.read_text(encoding="utf-8"))
        good = (doc.get("traits") or {}).get("good") or {}
        for k, v in (good.get("mean_trait_score_per_layer") or {}).items():
            scores[int(k)] = float(v)
        if scores:
            logger.info("Layer trait scores from %s", sweep.name)
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        default = float(doc.get("default_score", 5.0))
        for k, v in (doc.get("scores") or {}).items():
            scores[int(k)] = float(v)
        for l in range(NUM_LAYERS):
            scores.setdefault(l, default)
        logger.info("Layer trait scores from %s (default=%.1f)", path.name, default)
    if not scores:
        scores = {l: 5.0 for l in range(NUM_LAYERS)}
        scores[16] = 80.3
        scores[22] = 0.75
    return scores


SHARED_OUTLIER_DIMS = {443}


def from_persona_vectors(
    vectors_path: Path,
    trait_scores_path: Path,
    trait_name: str = "good",
) -> dict:
    ck = torch.load(vectors_path, map_location="cpu", weights_only=False)
    v = ck["v"].float()
    trait_scores = _load_layer_trait_scores(trait_scores_path)

    v_viz = v.clone()
    for d in SHARED_OUTLIER_DIMS:
        v_viz[:, d] = 0.0

    global_max = float(v_viz.abs().max())
    if global_max <= 0:
        raise ValueError("Zero persona vector (after removing shared outlier dims)")

    layers_out = []
    for l in range(int(v_viz.shape[0])):
        acts = [
            round(float(v_viz[l, d].abs() / global_max), 5)
            for d in range(int(v_viz.shape[1]))
        ]
        layers_out.append({
            "layer": l,
            "activation": acts,
            "trait_score": round(trait_scores.get(l, 5.0), 2),
        })

    best_l = max(trait_scores, key=lambda k: trait_scores[k])
    return {
        "trait": trait_name,
        "run_id": ck.get("meta", {}).get("rollouts_jsonl", f"dnd_{trait_name}").split("/")[-2],
        "source": f"persona_vectors |v| / global_max, dim 443 zeroed (shared outlier)",
        "steer_alpha": 1.5,
        "steer_layer": int(best_l),
        "layer_trait_scores": {str(k): trait_scores[k] for k in sorted(trait_scores)},
        "note": (
            "Per-dim color/size/opacity = |v_l,d| / global_max (dim 443 zeroed — "
            "shared persona-instruction outlier across all traits). "
            "Layer trait judge score scales whole-layer brightness and size."
        ),
        "excluded_dims": sorted(SHARED_OUTLIER_DIMS),
        "layers": layers_out,
    }


def _write_doc(out: Path, doc: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d layers)", out, len(doc["layers"]))
    steer_l = doc.get("steer_layer", 16)
    lrow = next(r for r in doc["layers"] if r["layer"] == steer_l)
    active = sum(1 for x in lrow["activation"] if x > 0.15)
    logger.info(
        "L%d trait_score=%.1f — %d/%d dims activation > 0.15",
        steer_l,
        lrow.get("trait_score", doc.get("layer_trait_scores", {}).get(str(steer_l), 0)),
        active,
        HIDDEN_DIM,
    )


def build_trait_doc(
    trait_name: str,
    *,
    vectors: Path,
    trait_scores: Path,
    viz: Path,
    export_path: Path | None,
    steer_alpha: float,
) -> dict | None:
    if export_path and export_path.is_file():
        return from_hidden_export(export_path)
    if vectors.is_file():
        return from_persona_vectors(vectors, trait_scores, trait_name=trait_name)
    if trait_name == "good" and viz.is_file():
        return from_sae_alpha_viz(viz, steer_alpha=steer_alpha)
    logger.error("No input for trait %s (vectors=%s)", trait_name, vectors)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild layer3d trait activation JSON")
    ap.add_argument(
        "--trait",
        choices=list(TRAIT_BUILDS) + ["all"],
        default="good",
        help="Trait to build (all = good, evil, lawful, chaotic)",
    )
    ap.add_argument("--viz", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--vectors", type=Path, default=None)
    ap.add_argument("--trait-scores", type=Path, default=None)
    ap.add_argument("--from-export", type=Path, default=None, help="Full hidden-state export JSON")
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT, help="Auto-use export if present")
    ap.add_argument("--steer-alpha", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    traits = list(TRAIT_BUILDS) if args.trait == "all" else [args.trait]
    export_path = args.from_export
    if export_path is None and args.export.is_file():
        export_path = args.export

    ok = 0
    for trait_name in traits:
        cfg = TRAIT_BUILDS[trait_name]
        vectors = args.vectors or cfg["vectors"]
        trait_scores = args.trait_scores or cfg["trait_scores"]
        out = args.out or cfg["out"]
        if args.trait == "all":
            out = cfg["out"]
            export_for_trait = export_path if trait_name == "good" else None
        else:
            export_for_trait = export_path

        doc = build_trait_doc(
            trait_name,
            vectors=vectors,
            trait_scores=trait_scores,
            viz=args.viz,
            export_path=export_for_trait,
            steer_alpha=args.steer_alpha,
        )
        if doc is None:
            continue
        _write_doc(out, doc)
        ok += 1

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
