#!/usr/bin/env python3
"""E0 — injection-scope ablation: full sequence vs assistant-span (Blas inventory mode).

Hypothesis: our inventory dose-response survives full-sequence injection (~65
positions) but collapses when injection is restricted to the generation-prompt
span (~3–4 positions), matching Blas et al. inventory protocol.

Run on GPU with Gemma vectors (Colab L4). CPU smoke mode verifies plumbing only.

    # Full E0 on Colab (see docs/E0_COLAB.md):
    python3 scripts/ablate_injection_scope.py \\
        --vectors-dir /content/ladder \\
        --out-dir results/injection_scope_ablation \\
        --model-id unsloth/gemma-3-4b-it

    # CPU plumbing check (not scientifically conclusive):
    python3 scripts/ablate_injection_scope.py --smoke

    # Re-score committed JSONs:
    python3 scripts/ablate_injection_scope.py --evaluate-only \\
        --out-dir results/injection_scope_ablation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("e0_scope")

# Magnitude grids copied from committed gemma_final / e1_inspan sweeps (raw units).
POLES = [
    {
        "trait": "conscientiousness",
        "pole": "high",
        "layer": 15,
        "grid_source": "results/gemma_final/validated_sweep_conscientiousness_pc1_high.json",
    },
    {
        "trait": "conscientiousness",
        "pole": "low",
        "layer": 15,
        "grid_source": "results/gemma_final/validated_sweep_conscientiousness_pc1_low.json",
    },
    {
        "trait": "extraversion",
        "pole": "high",
        "layer": 15,
        "grid_source": "results/e1_inspan/validated_sweep_extraversion_pc1_high.json",
    },
]

SCOPES = ("full", "assistant_span")
MIN_RUNGS = 3
CLEAR_MARGIN = 2.0


def _load_magnitude_grid(path: Path) -> list[float]:
    rep = json.loads(path.read_text())
    rows = rep.get("trait_curve", {}).get("rows") or []
    mags = [
        float(r["magnitude"])
        for r in rows
        if r.get("magnitude") is not None and abs(float(r.get("alpha") or 0)) > 1e-12
    ]
    if mags:
        return mags
    grid = rep.get("magnitude_grid") or []
    scale = float(rep.get("alpha_scale") or 1.0)
    if rep.get("alpha_units") == "relative":
        return [float(a) * scale for a in grid if abs(float(a)) > 1e-12]
    return [float(x) for x in grid if abs(float(x)) > 1e-12]


def shipped_screen(lock: dict) -> bool:
    return lock["top_option_fraction"] < 0.90 and lock["option_entropy"] >= 0.30


def evaluate_argmax(sw: dict, pole: str) -> dict:
    """Argmax dose-response stats for one sweep JSON."""
    from app.persona.intensity_ladder import spearman_rho

    sign = 1 if pole == "high" else -1
    rows = [
        r
        for r in sw["trait_curve"]["rows"]
        if r.get("target_argmax") is not None and shipped_screen(r["lock"])
    ]
    if len(rows) < MIN_RUNGS:
        return {
            "n_rungs": len(rows),
            "rho": None,
            "delta": None,
            "span": None,
            "sign_correct": None,
            "control_span": None,
            "trait_over_control": None,
            "supported": False,
        }
    xs = [abs(float(r["alpha"])) for r in rows]
    ys = [float(r["target_argmax"]) for r in rows]
    base = next(
        (float(r["target_argmax"]) for r in rows if abs(float(r.get("magnitude") or 0)) < 1e-9),
        ys[0],
    )
    extreme = max(ys) if sign > 0 else min(ys)
    rho = spearman_rho(xs, ys)
    span = max(ys) - min(ys)
    sign_ok = rho is not None and ((rho > 0) if sign > 0 else (rho < 0))

    ctrl_spans: list[float] = []
    for c in sw.get("control_curves") or []:
        crows = [
            r
            for r in c["rows"]
            if r.get("target_argmax") is not None and shipped_screen(r["lock"])
        ]
        if len(crows) >= 2:
            vals = [float(r["target_argmax"]) for r in crows]
            ctrl_spans.append(max(vals) - min(vals))
    ctrl = max(ctrl_spans) if ctrl_spans else None
    ratio = (span / ctrl) if ctrl not in (None, 0) else None

    return {
        "n_rungs": len(rows),
        "rho": round(rho, 4) if rho is not None else None,
        "delta": round(extreme - base, 4),
        "span": round(span, 4),
        "sign_correct": sign_ok,
        "control_span": round(ctrl, 4) if ctrl is not None else None,
        "trait_over_control": round(ratio, 2) if ratio is not None else None,
        "supported": bool(sign_ok and ratio is not None and ratio >= CLEAR_MARGIN),
    }


def run_one(
    spec: dict,
    *,
    scope: str,
    vectors_dir: Path,
    out_dir: Path,
    model_id: str,
    items_csv: Path,
    n_controls: int,
    n_probes: int,
    max_new_tokens: int,
) -> dict:
    from app.persona.intensity_ladder import run_validated_sweep

    trait, pole, layer = spec["trait"], spec["pole"], spec["layer"]
    vec_pt = vectors_dir / f"ladder_vectors_{trait}.pt"
    if not vec_pt.is_file():
        raise FileNotFoundError(vec_pt)

    grid_path = REPO_ROOT / spec["grid_source"]
    mags = _load_magnitude_grid(grid_path)
    logger.info(
        "%s-%s scope=%s L%s grid(abs)=%s (from %s)",
        trait,
        pole,
        scope,
        layer,
        [round(m, 1) for m in mags],
        grid_path.name,
    )

    out_json = out_dir / f"validated_sweep_{trait}_pc1_{pole}_{scope}.json"
    run_validated_sweep(
        vec_pt,
        out_json,
        trait=trait,
        which="pc1",
        layer_idx=layer,
        magnitudes=mags,
        auto_calibrate=False,
        steer_toward=pole,
        n_random_controls=n_controls,
        alpha_units="raw",
        model_id=model_id,
        items_csv=items_csv,
        probe_questions=[],
        max_new_tokens=max_new_tokens,
        baseline="persona_free",
        injection_scope=scope,
    )
    sw = json.loads(out_json.read_text())
    stats = evaluate_argmax(sw, pole)
    return {
        "trait": trait,
        "pole": pole,
        "layer": layer,
        "scope": scope,
        "magnitude_grid_abs": [round(m, 2) for m in mags],
        "report": str(out_json),
        **stats,
    }


def run_smoke(out_dir: Path, model_id: str) -> list[dict]:
    """Minimal CPU run: one trait, six items, two scopes, tiny grid."""
    from app.persona.intensity_ladder import (
        build_ladder_vectors,
        run_prompt_ladder,
        run_validated_sweep,
    )

    os.environ.setdefault("PERSONA_FORCE_CPU", "1")
    trait = "conscientiousness"
    work = out_dir / "smoke_vectors"
    work.mkdir(parents=True, exist_ok=True)
    items_csv = REPO_ROOT / "data" / "ipip_neo_120.csv"

    ladder_json = work / "prompt_ladder.json"
    centroids = work / "centroids.pt"
    run_prompt_ladder(
        ladder_json,
        centroids,
        trait=trait,
        model_id=model_id,
        variants=1,
        levels=(1, 5, 9),
        all_traits=False,
        items_csv=items_csv,
    )
    vec_pt = work / f"ladder_vectors_{trait}.pt"
    build_ladder_vectors(centroids, vec_pt, work / "vectors.json")

    rows: list[dict] = []
    for scope in SCOPES:
        out_json = out_dir / f"smoke_{trait}_high_{scope}.json"
        run_validated_sweep(
            vec_pt,
            out_json,
            trait=trait,
            which="pc1",
            layer_idx=1,
            magnitudes=(50.0, 150.0),
            auto_calibrate=False,
            steer_toward="high",
            n_random_controls=1,
            alpha_units="raw",
            model_id=model_id,
            items_csv=items_csv,
            probe_questions=[],
            max_new_tokens=32,
            baseline="persona_free",
            injection_scope=scope,
        )
        sw = json.loads(out_json.read_text())
        stats = evaluate_argmax(sw, "high")
        rows.append({"trait": trait, "pole": "high", "scope": scope, "smoke": True, **stats})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectors-dir", default="/content/ladder")
    p.add_argument("--out-dir", default="results/injection_scope_ablation")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--probes", type=int, default=0, help="Free-text probes per rung (0 = inventory only)")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--only",
        default="",
        help="Comma list trait:pole:scope e.g. conscientiousness:high:full",
    )
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="CPU plumbing check with Qwen-0.5B (not Gemma E0)",
    )
    p.add_argument("--smoke-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("E0 smoke: verifying injection_scope plumbing on CPU …", flush=True)
        rows = run_smoke(out_dir, args.smoke_model_id)
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "e0_injection_scope_smoke",
            "note": "Plumbing only — not the Gemma E0 result.",
            "table": rows,
        }
        (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for r in rows:
            print(
                f"  {r['scope']:<16} rungs={r['n_rungs']} span={r['span']} "
                f"rho={r['rho']} supported={r['supported']}"
            )
        return 0

    vectors_dir = Path(args.vectors_dir)
    items_csv = Path(args.items_csv)

    wanted: set[tuple[str, str, str]] | None = None
    if args.only.strip():
        wanted = set()
        for part in args.only.split(","):
            part = part.strip()
            if not part:
                continue
            bits = part.split(":")
            if len(bits) == 2:
                wanted.add((bits[0], bits[1], "full"))
                wanted.add((bits[0], bits[1], "assistant_span"))
            elif len(bits) == 3:
                wanted.add((bits[0], bits[1], bits[2]))

    rows: list[dict] = []
    for spec in POLES:
        for scope in SCOPES:
            key = (spec["trait"], spec["pole"], scope)
            if wanted is not None and key not in wanted:
                continue
            out_json = out_dir / f"validated_sweep_{spec['trait']}_pc1_{spec['pole']}_{scope}.json"
            if args.evaluate_only:
                if not out_json.is_file():
                    rows.append({**spec, "scope": scope, "error": f"missing {out_json}"})
                    continue
                sw = json.loads(out_json.read_text())
                stats = evaluate_argmax(sw, spec["pole"])
                rows.append({**spec, "scope": scope, "report": str(out_json), **stats})
                continue
            try:
                rows.append(
                    run_one(
                        spec,
                        scope=scope,
                        vectors_dir=vectors_dir,
                        out_dir=out_dir,
                        model_id=args.model_id,
                        items_csv=items_csv,
                        n_controls=args.random_controls,
                        n_probes=args.probes,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            except Exception as exc:
                logger.exception("%s-%s %s failed: %s", spec["trait"], spec["pole"], scope, exc)
                rows.append({**spec, "scope": scope, "error": str(exc), "supported": False})

    # Pairwise comparison per pole
    comparisons: list[dict] = []
    for spec in POLES:
        pole_key = (spec["trait"], spec["pole"])
        full = next((r for r in rows if r.get("trait") == pole_key[0] and r.get("pole") == pole_key[1] and r.get("scope") == "full"), None)
        span = next(
            (r for r in rows if r.get("trait") == pole_key[0] and r.get("pole") == pole_key[1] and r.get("scope") == "assistant_span"),
            None,
        )
        if full and span and "error" not in full and "error" not in span:
            collapse = (
                full.get("supported")
                and not span.get("supported")
                and (span.get("span") or 0) < (full.get("span") or 0) * 0.5
            )
            comparisons.append(
                {
                    "pole": f"{spec['trait'][0].upper()}-{'up' if spec['pole']=='high' else 'down'}",
                    "full_supported": full.get("supported"),
                    "span_supported": span.get("supported"),
                    "full_span": full.get("span"),
                    "assistant_span": span.get("span"),
                    "hypothesis_confirmed": collapse,
                }
            )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "e0_injection_scope_ablation",
        "pass_gate": {"min_rungs": MIN_RUNGS, "margin": CLEAR_MARGIN, "readout": "argmax"},
        "n_supported_full": sum(1 for r in rows if r.get("scope") == "full" and r.get("supported")),
        "n_supported_assistant_span": sum(
            1 for r in rows if r.get("scope") == "assistant_span" and r.get("supported")
        ),
        "comparisons": comparisons,
        "table": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 96)
    print("E0 INJECTION SCOPE ABLATION  (argmax readout; full vs assistant_span)")
    print("=" * 96)
    hdr = f"{'pole':<12}{'scope':<16}{'rho':>7}{'delta':>8}{'span':>7}{'ctrl':>7}{'ratio':>7}{'ok':>6}"
    print(hdr)
    print("-" * len(hdr))
    for spec in POLES:
        label = f"{spec['trait'][0].upper()}-{'up' if spec['pole']=='high' else 'down'}"
        for scope in SCOPES:
            r = next(
                (x for x in rows if x.get("trait") == spec["trait"] and x.get("pole") == spec["pole"] and x.get("scope") == scope),
                None,
            )
            if r is None:
                continue
            if "error" in r:
                print(f"{label:<12}{scope:<16} ERROR: {r['error'][:50]}")
                continue
            print(
                f"{label:<12}{scope:<16}"
                f"{str(r.get('rho')):>7}{str(r.get('delta')):>8}{str(r.get('span')):>7}"
                f"{str(r.get('control_span')):>7}{str(r.get('trait_over_control')):>7}"
                f"{str(r.get('supported')):>6}"
            )
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
