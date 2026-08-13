#!/usr/bin/env python3
"""Rebuild layer3d static JSON from validate + layer-sweep artifacts (post-consolidation)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.rebuild_layer3d_good_trait_data import (
    NUM_LAYERS,
    TRAIT_BUILDS,
    _write_doc,
    from_persona_vectors,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIT_RUNS = {
    "good": "dnd_good_scale",
    "evil": "dnd_evil",
    "lawful": "dnd_lawful",
    "chaotic": "dnd_chaotic",
}

STATIC = REPO / "app/static"
DEFAULT_SCORE = 5.0


def _gate(report: dict, name: str) -> dict | None:
    for g in report.get("gates") or []:
        if g.get("gate") == name:
            return g
    return None


def layer_scores_from_sweep(sweep: dict, trait: str) -> dict[int, float]:
    block = (sweep.get("traits") or {}).get(trait) or {}
    raw = block.get("mean_trait_score_per_layer") or {}
    return {int(k): float(v) for k, v in raw.items()}


def layer_scores_from_validate(report: dict) -> dict[int, float]:
    g2 = _gate(report, "layer_selection")
    if not g2:
        return {}
    raw = (g2.get("details") or {}).get("mean_trait_score_per_layer") or {}
    return {int(k): float(v) for k, v in raw.items()}


def merged_layer_scores(sweep: dict, report: dict, trait: str) -> dict[int, float]:
    """Prefer layer-sweep @ α=1.5 (L12–16); fill gaps from validate gate2 @ α=1.0."""
    scores = {l: DEFAULT_SCORE for l in range(NUM_LAYERS)}
    scores.update(layer_scores_from_validate(report))
    scores.update(layer_scores_from_sweep(sweep, trait))
    return scores


def alpha_rows_from_validate(report: dict) -> list[dict]:
    g3 = _gate(report, "steering_effectiveness")
    if not g3:
        return []
    per = (g3.get("details") or {}).get("per_alpha") or {}
    rows = []
    for akey in sorted(per.keys(), key=float):
        row = per[akey]
        rows.append(
            {
                "alpha": float(akey),
                "mean_trait": float(row.get("mean_trait", 0)),
                "mean_coherence": float(row.get("mean_coherence", 0)),
            }
        )
    return rows


def write_trait_scores_json(trait: str, scores: dict[int, float], meta: dict) -> Path:
    out = STATIC / f"{trait}_layer_trait_scores.json"
    doc = {
        "trait": trait,
        "steer_alpha": meta["recommended_alpha"],
        "steer_layer": meta["recommended_layer"],
        "source": meta["source"],
        "default_score": DEFAULT_SCORE,
        "scores": {str(k): round(scores[k], 2) for k in sorted(scores) if scores[k] != DEFAULT_SCORE},
    }
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d non-default layers)", out.name, len(doc["scores"]))
    return out


def main() -> int:
    sweep_path = REPO / "persona_runs/dnd_layer_sweep.json"
    if not sweep_path.is_file():
        logger.error("Missing %s — fetch from VM first", sweep_path)
        return 1
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))

    alpha_bundle: dict = {
        "source": "validate steering_effectiveness (Gate 3), Jun 2026 consolidation",
        "coherence_floor": 80,
        "traits": {},
    }

    for trait, run_id in TRAIT_RUNS.items():
        report_path = REPO / "persona_runs" / run_id / "eval" / "validation_report.json"
        if not report_path.is_file():
            logger.error("Missing %s", report_path)
            return 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        layer = int(report["recommended_layer"])
        alpha = float(report["recommended_alpha"])
        scores = merged_layer_scores(sweep, report, trait)
        write_trait_scores_json(
            trait,
            scores,
            {
                "recommended_layer": layer,
                "recommended_alpha": alpha,
                "source": (
                    f"layer scores: dnd_layer_sweep.json @ α=1.5 (L12–16) + validate gate2; "
                    f"steer L{layer} α={alpha:g} from validation_report.json"
                ),
            },
        )

        rows = alpha_rows_from_validate(report)
        max_coh80 = None
        for r in rows:
            if r["mean_coherence"] >= 80:
                max_coh80 = r["alpha"]
        alpha_bundle["traits"][trait] = {
            "run_id": run_id,
            "layer": layer,
            "recommended_alpha": alpha,
            "max_alpha_coherence_ge_80": max_coh80,
            "rows": rows,
        }

    alpha_out = STATIC / "layer3d_alpha_sweep.json"
    alpha_out.write_text(json.dumps(alpha_bundle, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", alpha_out)

    cfg = TRAIT_BUILDS.copy()
    cfg["good"] = {
        **cfg["good"],
        "vectors": REPO / "persona_runs/dnd_good_scale/vectors/persona_vectors.pt",
    }
    if not cfg["good"]["vectors"].is_file():
        cfg["good"]["vectors"] = REPO / "persona_runs/dnd_good/vectors/persona_vectors.pt"
        logger.warning("dnd_good_scale vectors missing — using dnd_good persona_vectors.pt")

    for trait in TRAIT_RUNS:
        trait_scores_path = STATIC / f"{trait}_layer_trait_scores.json"
        vectors = cfg[trait]["vectors"]
        if not vectors.is_file():
            logger.warning("Skipping layer3d_%s — no vectors at %s", trait, vectors)
            continue
        doc = from_persona_vectors(vectors, trait_scores_path, trait_name=trait)
        report = json.loads(
            (REPO / "persona_runs" / TRAIT_RUNS[trait] / "eval" / "validation_report.json").read_text()
        )
        doc["steer_layer"] = int(report["recommended_layer"])
        doc["steer_alpha"] = float(report["recommended_alpha"])
        doc["run_id"] = TRAIT_RUNS[trait]
        doc["validation_source"] = str(
            (REPO / "persona_runs" / TRAIT_RUNS[trait] / "eval" / "validation_report.json").resolve()
        )
        doc["alpha_sweep"] = alpha_bundle["traits"][trait]["rows"]
        _write_doc(cfg[trait]["out"], doc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
