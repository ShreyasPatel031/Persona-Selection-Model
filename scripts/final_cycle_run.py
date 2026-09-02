#!/usr/bin/env python3
"""Full final-cycle defensibility run: reliability → ladder → extract → steer.

One model load. Layer 15 only. MPI-120 primary. Writes incremental JSON under
``results/final_cycle/`` so a killed session still leaves usable artifacts.

    python3 scripts/final_cycle_run.py \\
        --out-dir results/final_cycle \\
        --items-csv data/mpi_120.csv \\
        --items-300 data/ipip_neo_facets_300.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("final_cycle")

LAYER = 15
PASS_RHO = 0.8
PASS_MARGIN = 2.0
MIN_USABLE = 4
POLES = {
    "up": (2, 9, +1),
    "down": (8, 1, -1),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _grid(span: float, lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [span * lo]
    step = (hi - lo) / (n - 1)
    return [span * (lo + i * step) for i in range(n)]


def cronbach_alpha(item_scores: list[list[float]]) -> float | None:
    """Cronbach's α over items (rows = admins or people, cols = items)."""
    if not item_scores or len(item_scores) < 2:
        return None
    n_items = len(item_scores[0])
    if n_items < 2 or any(len(r) != n_items for r in item_scores):
        return None
    # transpose → per-item variances
    cols = list(zip(*item_scores))
    item_vars = [statistics.pvariance(c) for c in cols]
    totals = [sum(r) for r in item_scores]
    total_var = statistics.pvariance(totals)
    if total_var <= 0:
        return None
    k = float(n_items)
    return float((k / (k - 1.0)) * (1.0 - sum(item_vars) / total_var))


def forward_reverse_corr(items, responses) -> float | None:
    """Corr of mean EV on plus-keyed vs reverse-keyed items (per trait later)."""
    from app.persona.inventory_ipip import reverse_scored

    plus: list[float] = []
    minus: list[float] = []
    for it, r in zip(items, responses):
        raw = r.get("value")
        if raw is None:
            probs = r.get("probs") or {}
            total = sum(probs.values())
            if not probs or total <= 0:
                continue
            raw = sum(float(p) * int(o) for o, p in probs.items()) / total
        keyed = float(raw) if it.keyed > 0 else float(6 - float(raw))
        # For the diagnostic we want *uncorrected* option means by keying group:
        # a collapse to option 3 looks like mid on both; a true trait pushes them apart.
        if it.keyed > 0:
            plus.append(float(raw))
        else:
            minus.append(float(raw))
    if len(plus) < 2 or len(minus) < 2:
        return None
    # Single-admin diagnostic: correlate keyed scores of plus items with
    # (6 - raw) of minus items? Plan wants mean(forward) vs mean(reverse raw).
    # Report difference of means as a scalar; corr needs multi-admin.
    return None  # filled in multi-admin path


def keyed_item_evs(items, responses, trait: str) -> list[float]:
    out = []
    for it, r in zip(items, responses):
        if it.trait != trait:
            continue
        probs = r.get("probs") or {}
        total = sum(probs.values())
        if probs and total > 0:
            raw = sum(float(p) * int(o) for o, p in probs.items()) / total
        elif r.get("value") is not None:
            raw = float(r["value"])
        else:
            continue
        out.append(raw if it.keyed > 0 else 6.0 - raw)
    return out


def admin_summary(items, responses, trait: str | None = None) -> dict:
    from app.persona.inventory_ipip import (
        item_log,
        option_lock,
        response_validity,
        score_traits,
        score_traits_ev,
    )

    lock = option_lock(responses)
    ev = score_traits_ev(responses)
    argmax = score_traits(responses)
    log = item_log(responses)
    per_trait = {}
    traits = [trait] if trait else sorted({it.trait for it in items})
    for t in traits:
        scores = keyed_item_evs(items, responses, t)
        fwd = [float(r.get("value") or 0) for it, r in zip(items, responses)
               if it.trait == t and it.keyed > 0 and r.get("value") is not None]
        rev = [float(r.get("value") or 0) for it, r in zip(items, responses)
               if it.trait == t and it.keyed < 0 and r.get("value") is not None]
        per_trait[t] = {
            "ev": round(float(ev.get(t, float("nan"))), 4) if t in ev else None,
            "argmax": round(float(argmax.get(t, float("nan"))), 4) if t in argmax else None,
            "item_sigma": round(statistics.pstdev(scores), 4) if len(scores) > 1 else None,
            "n_items": len(scores),
            "fwd_mean_raw": round(sum(fwd) / len(fwd), 4) if fwd else None,
            "rev_mean_raw": round(sum(rev) / len(rev), 4) if rev else None,
        }
    return {
        "ev_scores": {k: round(v, 4) for k, v in ev.items()},
        "argmax_scores": {k: round(v, 4) for k, v in argmax.items()},
        "response_validity": round(response_validity(responses), 4),
        "lock": lock,
        "usable": not lock["locked"],
        "per_trait": per_trait,
        "item_log": log,
    }


def load_direction(centroids_dir: Path, trait: str, layer: int):
    """Return (unit direction, span) for ``trait`` at ``layer``.

    Accepts either a full ``ladder_vectors_*.pt`` from Phase 3 or a ``slim_*.pt``
    carrying only the one layer's vector and level centroids. The slim form
    exists because the full packs are ~4 MB each and the Colab upload path is
    unreliable above a few MB, which cost a whole re-run of Phases 2-3.
    """
    slim = centroids_dir / f"slim_{trait}.pt"
    full = centroids_dir / f"ladder_vectors_{trait}.pt"
    if slim.is_file():
        b = torch.load(slim, map_location="cpu")
        raw = b["v_pc1_layer"].float()
        cents = b["level_centroids_layer"].float()
        unit = raw / raw.norm().clamp_min(1e-9)
        span = abs(float(torch.dot(cents[-1] - cents[0], unit)))
        return unit, span
    if full.is_file():
        from app.persona.intensity_ladder import direction_span_magnitude

        b = torch.load(full, map_location="cpu")
        raw = b["v_pc1"][layer].float()
        unit = raw / raw.norm().clamp_min(1e-9)
        span = direction_span_magnitude(b["level_centroids"], layer, b["v_pc1"][layer])
        return unit, span
    return None, None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="results/final_cycle")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "mpi_120.csv"))
    p.add_argument("--items-300", default=str(REPO_ROOT / "data" / "ipip_neo_facets_300.csv"))
    p.add_argument("--traits", default="openness,conscientiousness,extraversion,agreeableness,neuroticism")
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--rungs", type=int, default=8)
    p.add_argument("--random-controls", type=int, default=1)
    p.add_argument("--frac-lo", type=float, default=0.15)
    p.add_argument("--frac-hi", type=float, default=1.30)
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-phase4", action="store_true")
    p.add_argument("--phases", default="1,2,3,4,5", help="Comma list of phases to run (5 = sign test).")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    phases = {int(x) for x in args.phases.split(",") if x.strip()}
    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.persona.intensity_ladder import (
        _Steering,
        _load_model,
        _option_token_ids,
        administer_inventory,
        build_ladder_vectors,
        direction_span_magnitude,
        monotone_fraction,
        random_control_directions,
        spearman_rho,
    )
    from app.persona.intensity_prompts import ladder_system_prompt, persona_free_system_prompt
    from app.persona.inventory_ipip import items_from_csv, keying_balance

    items = items_from_csv(Path(args.items_csv))
    items_300_path = Path(args.items_300)
    items_300 = items_from_csv(items_300_path) if items_300_path.is_file() else []

    t0 = time.time()
    logger.info("loading model %s", args.model_id)
    model, tokenizer, dev = _load_model(args.model_id, None)
    option_ids = _option_token_ids(tokenizer)
    logger.info("model ready on %s in %.1fs", dev, time.time() - t0)

    def administer(system: str, direction=None, magnitude: float = 0.0, layer: int = LAYER):
        if direction is None or magnitude == 0.0:
            responses, centroid = administer_inventory(
                model, tokenizer, dev, system, items, option_ids=option_ids,
                collect_activations=True,
            )
        else:
            with _Steering(model, layer, direction, magnitude):
                responses, centroid = administer_inventory(
                    model, tokenizer, dev, system, items, option_ids=option_ids,
                    collect_activations=True,
                )
        return responses, centroid

    manifest = {
        "created_utc": _utc(),
        "model_id": args.model_id,
        "instrument": Path(args.items_csv).name,
        "n_items": len(items),
        "keying_balance": keying_balance(items),
        "layer": LAYER,
        "traits": traits,
        "phases": sorted(phases),
    }
    _write(out_dir / "manifest.json", manifest)

    # ── Phase 1: reliability gate ─────────────────────────────────────────────
    if 1 in phases and not args.skip_phase1:
        logger.info("=== PHASE 1 reliability gate ===")
        preambles = [
            persona_free_system_prompt(),
            "Answer honestly about yourself, without trying to look good.",
            "Describe yourself as you actually are, not as you wish to be.",
            "Be frank. Rate each statement for how it fits you.",
            # reversed-scale instruction: still 1-5 but ask to treat 5 as inaccurate
            "Answer as yourself. On the scale, treat higher numbers as *less* accurate "
            "for you than lower numbers would usually mean — invert your usual sense of the scale.",
        ]
        rows = []
        # Multi-admin for α: collect keyed item EVs per trait across first 4 preambles
        per_trait_matrix: dict[str, list[list[float]]] = {t: [] for t in traits}
        fwd_means: dict[str, list[float]] = {t: [] for t in traits}
        rev_means: dict[str, list[float]] = {t: [] for t in traits}

        for i, system in enumerate(preambles):
            logger.info("phase1 admin %s/%s", i + 1, len(preambles))
            responses, _ = administer(system)
            summ = admin_summary(items, responses)
            summ["preamble_idx"] = i
            summ["reversed_scale_instruction"] = i == len(preambles) - 1
            summ["system_prompt"] = system
            rows.append(summ)
            if i < 4 and summ["usable"]:
                for t in traits:
                    scores = keyed_item_evs(items, responses, t)
                    if scores:
                        per_trait_matrix[t].append(scores)
                    pt = summ["per_trait"][t]
                    if pt["fwd_mean_raw"] is not None:
                        fwd_means[t].append(pt["fwd_mean_raw"])
                    if pt["rev_mean_raw"] is not None:
                        rev_means[t].append(pt["rev_mean_raw"])
            _write(out_dir / "phase1_reliability.json", {
                "created_utc": _utc(), "stage": "reliability", "administrations": rows,
            })

        neo300 = None
        if items_300:
            logger.info("phase1 IPIP-NEO-300 (%s items)", len(items_300))
            responses, _ = administer_inventory(
                model, tokenizer, dev, persona_free_system_prompt(),
                items_300, option_ids=option_ids,
            )
            neo300 = admin_summary(items_300, responses)

        gate = {}
        for t in traits:
            alphas = cronbach_alpha(per_trait_matrix[t])
            usable_rows = [r for r in rows[:4] if r["usable"]]
            evs = [r["per_trait"][t]["ev"] for r in usable_rows if r["per_trait"][t]["ev"] is not None]
            locks = sum(1 for r in rows[:4] if not r["usable"])
            # forward-reverse: corr across admins of fwd_mean vs rev_mean
            fr = None
            if len(fwd_means[t]) >= 3 and len(fwd_means[t]) == len(rev_means[t]):
                fr = spearman_rho(fwd_means[t], rev_means[t])
            unlocked = locks == 0 and bool(evs)
            # sign-stable: EV doesn't flip wildly (σ across admins < 0.75)
            stable = (statistics.pstdev(evs) < 0.75) if len(evs) >= 2 else bool(evs)
            # FR diagnostic: positive corr of raw fwd vs raw rev means → response bias
            fr_ok = fr is None or fr < 0.5
            proceeds = bool(unlocked and stable and fr_ok)
            gate[t] = {
                "cronbach_alpha": None if alphas is None else round(alphas, 4),
                "ev_mean": round(sum(evs) / len(evs), 4) if evs else None,
                "ev_sigma_across_admins": round(statistics.pstdev(evs), 4) if len(evs) > 1 else None,
                "n_locked_of_4": locks,
                "forward_reverse_rho": None if fr is None else round(float(fr), 4),
                "unlocked": unlocked,
                "sign_stable": stable,
                "fr_ok": fr_ok,
                "proceeds": proceeds,
            }
            logger.info(
                "gate %s: proceeds=%s α=%s ev=%.3f±%s fr_ρ=%s",
                t, proceeds, gate[t]["cronbach_alpha"],
                gate[t]["ev_mean"] or float("nan"),
                gate[t]["ev_sigma_across_admins"],
                gate[t]["forward_reverse_rho"],
            )

        phase1 = {
            "created_utc": _utc(),
            "stage": "reliability",
            "instrument": Path(args.items_csv).name,
            "administrations": rows,
            "ipip_neo_300": neo300,
            "gate": gate,
            "proceeding_traits": [t for t in traits if gate[t]["proceeds"]],
        }
        _write(out_dir / "phase1_reliability.json", phase1)
        proceeding = [t for t in traits if gate[t]["proceeds"]]
        if not proceeding:
            logger.error("no traits passed reliability gate — aborting later phases")
            return 1
    else:
        proceeding = list(traits)
        gate = {t: {"proceeds": True} for t in traits}

    # ── Phase 2: prompt ladder + centroids ────────────────────────────────────
    centroids_dir = out_dir / "ladder"
    centroids_dir.mkdir(parents=True, exist_ok=True)

    if 2 in phases and not args.skip_phase2:
        logger.info("=== PHASE 2 prompting baseline (%s traits × 9 × %s variants) ===",
                    len(proceeding), args.variants)
        for trait in proceeding:
            logger.info("ladder %s", trait)
            rows = []
            centroid_grid: list[list[torch.Tensor]] = []
            level_list = list(range(1, 10))
            for level in level_list:
                per_variant: list[torch.Tensor] = []
                for variant in range(args.variants):
                    system = ladder_system_prompt(trait, level, variant=variant, n_markers=3)
                    responses, centroid = administer(system)
                    summ = admin_summary(items, responses, trait=trait)
                    rows.append({
                        "level": level,
                        "variant": variant,
                        "system_prompt": system,
                        "target_ev": summ["per_trait"][trait]["ev"],
                        "target_argmax": summ["per_trait"][trait]["argmax"],
                        "ev_scores": summ["ev_scores"],
                        "lock": summ["lock"],
                        "usable": summ["usable"],
                        "item_sigma": summ["per_trait"][trait]["item_sigma"],
                        "item_log": summ["item_log"],
                    })
                    if summ["usable"] and centroid is not None:
                        per_variant.append(centroid.detach().cpu())
                    logger.info(
                        "  L%s v%s ev=%s usable=%s",
                        level, variant, summ["per_trait"][trait]["ev"], summ["usable"],
                    )
                centroid_grid.append(per_variant)

            usable = [r for r in rows if r["usable"] and r["target_ev"] is not None]
            xs = [float(r["level"]) for r in usable]
            ys = [float(r["target_ev"]) for r in usable]
            level_means = []
            for lv in level_list:
                vals = [float(r["target_ev"]) for r in usable if r["level"] == lv]
                level_means.append(round(sum(vals) / len(vals), 4) if vals else None)
            present = [v for v in level_means if v is not None]
            report = {
                "created_utc": _utc(),
                "stage": "prompt_ladder",
                "trait": trait,
                "instrument": Path(args.items_csv).name,
                "n_items": len(items),
                "variants": args.variants,
                "administrations": rows,
                "level_mean_target_ev": level_means,
                "target_ev_range": [min(present), max(present)] if present else None,
                "span": (max(present) - min(present)) if present else None,
                "spearman_level_vs_target_ev": (
                    None if spearman_rho(xs, ys) is None else round(float(spearman_rho(xs, ys)), 4)
                ),
                "n_usable": len(usable),
                "n_total": len(rows),
            }
            _write(centroids_dir / f"prompt_ladder_{trait}.json", report)

            # Save activation grid for extraction: (n_levels, n_variants, n_layers, d)
            # Pad missing variants with nan-free zeros and mask? Prefer equal counts.
            # Use only levels that have ≥1 variant; pad variants to max.
            max_v = max((len(v) for v in centroid_grid), default=0)
            if max_v == 0:
                logger.error("no centroids for %s — skip extraction", trait)
                continue
            filled = []
            for per_v in centroid_grid:
                if not per_v:
                    # duplicate last available later; skip with None placeholder
                    filled.append(None)
                    continue
                while len(per_v) < max_v:
                    per_v.append(per_v[-1])
                filled.append(torch.stack(per_v[:max_v], dim=0))
            # Replace missing levels with nearest usable
            for i, block in enumerate(filled):
                if block is None:
                    donor = next((b for b in filled if b is not None), None)
                    if donor is None:
                        break
                    filled[i] = donor
            acts = torch.stack(filled, dim=0)  # (n_levels, n_variants, n_layers, d)
            torch.save(
                {
                    "trait": trait,
                    "levels": level_list,
                    "activations": acts,
                    "model_id": args.model_id,
                    "context_mode": "ladder_system_prompt",
                    "instrument": Path(args.items_csv).name,
                },
                centroids_dir / f"centroids_{trait}.pt",
            )
            logger.info(
                "ladder %s span=%.2f ρ=%.3f",
                trait, report["span"] or -1, report["spearman_level_vs_target_ev"] or -1,
            )

    # ── Phase 3: extract PC1 at all layers (sweep uses L15) ───────────────────
    if 3 in phases:
        logger.info("=== PHASE 3 extraction (PC1; freeze protocol) ===")
        for trait in proceeding:
            cents = centroids_dir / f"centroids_{trait}.pt"
            if not cents.is_file():
                logger.warning("missing centroids for %s", trait)
                continue
            build_ladder_vectors(
                cents,
                centroids_dir / f"ladder_vectors_{trait}.pt",
                centroids_dir / f"ladder_geometry_{trait}.json",
            )
            blob = torch.load(centroids_dir / f"ladder_vectors_{trait}.pt", map_location="cpu")
            v = blob["v_pc1"][LAYER]
            logger.info(
                "extracted %s L%s |v|=%.3f cos(end,pc1)=%s",
                trait, LAYER, float(v.norm()),
                blob["geometry"]["per_layer"][LAYER].get("cos_endpoint_pc1"),
            )

    # ── Phase 4: opposite-prior sweeps + random + bipolar ─────────────────────
    if 4 in phases and not args.skip_phase4:
        logger.info("=== PHASE 4 steering sweeps @ L%s ===", LAYER)
        sweep_rows = []
        for trait in proceeding:
            v, span = load_direction(centroids_dir, trait, LAYER)
            if v is None:
                logger.warning("no vectors for %s", trait)
                continue
            controls = random_control_directions(
                int(v.shape[0]), args.random_controls, seed=0, like=v
            )
            mags = _grid(span, args.frac_lo, args.frac_hi, args.rungs)
            bipolar = -v

            for pole in ("up", "down"):
                base_level, ref_level, sign = POLES[pole]
                base_system = ladder_system_prompt(trait, base_level, n_markers=3)
                ref_system = ladder_system_prompt(trait, ref_level, n_markers=3)
                logger.info("%s-%s span=%.1f prior=L%s ref=L%s", trait, pole, span, base_level, ref_level)

                def run_curve(direction, label: str, baseline_row: dict | None = None):
                    out = []
                    if baseline_row is not None and label == "trait":
                        out.append(baseline_row)
                    else:
                        responses, _ = administer(base_system)
                        s = admin_summary(items, responses, trait=trait)
                        out.append({
                            "magnitude": 0.0,
                            "target_ev": s["per_trait"][trait]["ev"],
                            "usable": s["usable"],
                            "lock": s["lock"],
                            "item_sigma": s["per_trait"][trait]["item_sigma"],
                            "ev_scores": s["ev_scores"],
                            "item_log": s["item_log"],
                        })
                    for m in mags:
                        responses, _ = administer(base_system, direction, sign * m)
                        s = admin_summary(items, responses, trait=trait)
                        row = {
                            "magnitude": round(float(sign * m), 2),
                            "target_ev": s["per_trait"][trait]["ev"],
                            "usable": s["usable"],
                            "lock": s["lock"],
                            "item_sigma": s["per_trait"][trait]["item_sigma"],
                            "ev_scores": s["ev_scores"],
                            "item_log": s["item_log"],
                        }
                        out.append(row)
                        logger.info("  %s mag=%+7.1f ev=%s usable=%s",
                                    label, sign * m, row["target_ev"], row["usable"])
                    return out

                responses, _ = administer(base_system)
                base_s = admin_summary(items, responses, trait=trait)
                baseline = {
                    "magnitude": 0.0,
                    "target_ev": base_s["per_trait"][trait]["ev"],
                    "usable": base_s["usable"],
                    "lock": base_s["lock"],
                    "item_sigma": base_s["per_trait"][trait]["item_sigma"],
                    "ev_scores": base_s["ev_scores"],
                    "item_log": base_s["item_log"],
                }
                responses, _ = administer(ref_system)
                ref_s = admin_summary(items, responses, trait=trait)
                prompt_gap = float(ref_s["per_trait"][trait]["ev"]) - float(baseline["target_ev"])

                trait_rows = run_curve(v, "trait", baseline_row=baseline)
                random_curves = [run_curve(cv, f"random{i}") for i, cv in enumerate(controls)]
                bipolar_rows = run_curve(bipolar, "bipolar")

                def score(rows, pole_name: str):
                    usable = [r for r in rows if r["usable"] and r["target_ev"] is not None]
                    xs = [abs(float(r["magnitude"])) for r in usable]
                    ys = [float(r["target_ev"]) for r in usable]
                    ys_signed = ys if pole_name == "up" else [-y for y in ys]
                    rho = spearman_rho(xs, ys_signed)
                    base_ev = float(rows[0]["target_ev"])
                    if usable:
                        best = (max if pole_name == "up" else min)(
                            usable, key=lambda r: float(r["target_ev"])
                        )
                        delta = float(best["target_ev"]) - base_ev
                    else:
                        best, delta = None, None
                    return {
                        "n_usable": len(usable),
                        "rho_signed": None if rho is None else round(float(rho), 4),
                        "monotone_fraction": (
                            None if not ys_signed
                            else round(float(monotone_fraction(ys_signed) or 0.0), 4)
                        ),
                        "baseline_ev": round(base_ev, 4),
                        "best_ev": None if best is None else best["target_ev"],
                        "best_magnitude": None if best is None else best["magnitude"],
                        "delta": None if delta is None else round(delta, 4),
                        "ev_curve": [r["target_ev"] for r in rows],
                        "usable_flags": [bool(r["usable"]) for r in rows],
                        # All five domains per rung: lets specificity_control.py
                        # separate trait-specific movement from a global slide
                        # without another GPU pass.
                        "rungs": [
                            {
                                "magnitude": r["magnitude"],
                                "ev_scores": r["ev_scores"],
                                "item_sigma": r.get("item_sigma"),
                                "lock": r.get("lock"),
                                "usable": bool(r["usable"]),
                            }
                            for r in rows
                        ],
                    }

                t_score = score(trait_rows, pole)
                c_scores = [score(c, pole) for c in random_curves]
                b_score = score(bipolar_rows, pole)
                max_ctrl = max((abs(c["delta"] or 0.0) for c in c_scores), default=0.0)
                t_abs = abs(t_score["delta"] or 0.0)
                margin = (t_abs / max_ctrl) if max_ctrl > 0 else (None if t_abs == 0 else 999.0)
                beats = (margin is not None and margin >= PASS_MARGIN) or (
                    max_ctrl == 0 and t_abs > 0
                )
                sign_ok = (t_score["delta"] is not None) and (
                    (pole == "up" and t_score["delta"] > 0)
                    or (pole == "down" and t_score["delta"] < 0)
                )
                # Bipolar should move the *opposite* way relative to trait sign
                bipolar_sign_ok = (b_score["delta"] is not None) and (
                    (pole == "up" and b_score["delta"] < 0)
                    or (pole == "down" and b_score["delta"] > 0)
                )
                pct_gap = (
                    round(100.0 * (t_score["delta"] / prompt_gap), 1)
                    if prompt_gap and t_score["delta"] is not None else None
                )
                works = bool(
                    sign_ok
                    and t_score["rho_signed"] is not None
                    and t_score["rho_signed"] >= PASS_RHO
                    and t_score["n_usable"] >= MIN_USABLE
                    and beats
                )
                rec = {
                    "trait": trait,
                    "pole": pole,
                    "layer": LAYER,
                    "span": round(float(span), 2),
                    "prior_level": base_level,
                    "reference_level": ref_level,
                    "prior_ev": baseline["target_ev"],
                    "reference_ev": ref_s["per_trait"][trait]["ev"],
                    "prompt_gap": round(prompt_gap, 4),
                    **t_score,
                    "pct_of_prompt_gap": pct_gap,
                    "sign_correct": sign_ok,
                    "max_control_abs_delta": round(max_ctrl, 4),
                    "margin": None if margin is None else round(float(margin), 3),
                    "beats_controls": beats,
                    "bipolar": b_score,
                    "bipolar_sign_ok": bipolar_sign_ok,
                    "works": works,
                    "control_curves": c_scores,
                }
                sweep_rows.append(rec)
                _write(out_dir / "phase4_sweeps.json", {
                    "created_utc": _utc(),
                    "stage": "steering_sweeps",
                    "layer": LAYER,
                    "pass_gate": {"rho": PASS_RHO, "margin": PASS_MARGIN, "min_usable": MIN_USABLE},
                    "table": sweep_rows,
                })

        # Final table print
        print("\n" + "=" * 110)
        print("PHASE 4 SWEEPS @ L15  (works = sign + ρ≥0.8 + usable≥4 + beats random×2)")
        print("=" * 110)
        print(f"{'trait':16} {'pole':5} {'prior':>6} {'best':>6} {'Δ':>7} {'gap':>7} {'%gap':>6} "
              f"{'ρ':>7} {'ctrl':>6} {'bipOK':>5} {'works':>5}")
        for r in sweep_rows:
            print(
                f"{r['trait']:16} {r['pole']:5} {r['prior_ev']:>6.2f} "
                f"{(r['best_ev'] if r['best_ev'] is not None else float('nan')):>6.2f} "
                f"{(r['delta'] if r['delta'] is not None else float('nan')):>+7.3f} "
                f"{r['prompt_gap']:>+7.3f} "
                f"{(r['pct_of_prompt_gap'] if r['pct_of_prompt_gap'] is not None else float('nan')):>5.1f}% "
                f"{(r['rho_signed'] if r['rho_signed'] is not None else float('nan')):>+7.3f} "
                f"{(r['max_control_abs_delta'] or 0):>6.3f} "
                f"{str(r['bipolar_sign_ok']):>5} {str(r['works']):>5}"
            )
        print(f"passing: {sum(1 for r in sweep_rows if r['works'])}/{len(sweep_rows)}")

    # ── Phase 4b: sign test from a neutral baseline ───────────────────────────
    # The opposite-prior sign test is confounded: starting at a pole leaves -v
    # with a floor/ceiling under it, so "did not flip" can mean "had nowhere to
    # go". Starting from persona-free puts the baseline mid-scale, where both
    # signs have headroom, so a genuine bipolar axis must separate.
    if 5 in phases:
        logger.info("=== PHASE 4b sign test from persona-free baseline ===")
        sign_rows = []
        neutral = persona_free_system_prompt()
        for trait in proceeding:
            v, span = load_direction(centroids_dir, trait, LAYER)
            if v is None:
                continue
            mags = _grid(span, args.frac_lo, args.frac_hi, args.rungs)
            ctrl = random_control_directions(int(v.shape[0]), 1, seed=7, like=v)[0]

            responses, _ = administer(neutral)
            b = admin_summary(items, responses, trait=trait)
            base_ev = b["per_trait"][trait]["ev"]

            def arm(direction, sign: int, label: str):
                rungs = []
                for m in mags:
                    responses, _ = administer(neutral, direction, sign * m)
                    s = admin_summary(items, responses, trait=trait)
                    rungs.append({
                        "magnitude": round(float(sign * m), 2),
                        "target_ev": s["per_trait"][trait]["ev"],
                        "ev_scores": s["ev_scores"],
                        "usable": s["usable"],
                        "item_sigma": s["per_trait"][trait]["item_sigma"],
                        "lock": s["lock"],
                    })
                    logger.info("  %s %s mag=%+7.1f ev=%s usable=%s",
                                trait, label, sign * m, rungs[-1]["target_ev"], rungs[-1]["usable"])
                usable = [r for r in rungs if r["usable"] and r["target_ev"] is not None]
                if not usable:
                    return {"label": label, "delta": None, "rungs": rungs}
                # signed extreme: furthest from baseline in either direction
                far = max(usable, key=lambda r: abs(float(r["target_ev"]) - float(base_ev)))
                return {
                    "label": label,
                    "delta": round(float(far["target_ev"]) - float(base_ev), 4),
                    "at_magnitude": far["magnitude"],
                    "n_usable": len(usable),
                    "rungs": rungs,
                }

            plus = arm(v, +1, "+v")
            minus = arm(v, -1, "-v")
            rand = arm(ctrl, +1, "random")

            dp, dm = plus["delta"], minus["delta"]
            opposed = (dp is not None and dm is not None and dp * dm < 0)
            rec = {
                "trait": trait,
                "layer": LAYER,
                "baseline_ev": base_ev,
                "delta_plus_v": dp,
                "delta_minus_v": dm,
                "delta_random": rand["delta"],
                "signs_opposed": opposed,
                "separation": (
                    None if dp is None or dm is None else round(abs(dp - dm), 4)
                ),
                "plus": plus,
                "minus": minus,
                "random": rand,
            }
            sign_rows.append(rec)
            logger.info("SIGN %s: base=%.2f +v=%s -v=%s opposed=%s",
                        trait, base_ev, dp, dm, opposed)
            _write(out_dir / "phase4b_signtest.json", {
                "created_utc": _utc(),
                "stage": "sign_test_neutral_baseline",
                "layer": LAYER,
                "baseline": "persona_free",
                "table": sign_rows,
            })

        print("\n" + "=" * 92)
        print("PHASE 4b SIGN TEST  (persona-free baseline; +v and -v must move opposite ways)")
        print("=" * 92)
        print(f"{'trait':16} {'base':>6} {'+v Δ':>8} {'-v Δ':>8} {'rand Δ':>8} {'sep':>7} {'opposed':>8}")
        for r in sign_rows:
            print(
                f"{r['trait']:16} {r['baseline_ev']:>6.2f} "
                f"{(r['delta_plus_v'] if r['delta_plus_v'] is not None else float('nan')):>+8.3f} "
                f"{(r['delta_minus_v'] if r['delta_minus_v'] is not None else float('nan')):>+8.3f} "
                f"{(r['delta_random'] if r['delta_random'] is not None else float('nan')):>+8.3f} "
                f"{(r['separation'] if r['separation'] is not None else float('nan')):>7.3f} "
                f"{str(r['signs_opposed']):>8}"
            )
        print(f"opposed: {sum(1 for r in sign_rows if r['signs_opposed'])}/{len(sign_rows)}")

    # Phase 5 summary
    summary = {
        "created_utc": _utc(),
        "elapsed_sec": round(time.time() - t0, 1),
        "gate": gate if 1 in phases else None,
        "ladder_spans": {},
        "sweeps_passing": None,
    }
    for trait in proceeding:
        lp = centroids_dir / f"prompt_ladder_{trait}.json"
        if lp.is_file():
            d = json.loads(lp.read_text())
            summary["ladder_spans"][trait] = {
                "span": d.get("span"),
                "range": d.get("target_ev_range"),
                "rho": d.get("spearman_level_vs_target_ev"),
            }
    sp = out_dir / "phase4_sweeps.json"
    if sp.is_file():
        d = json.loads(sp.read_text())
        summary["sweeps_passing"] = [
            f"{r['trait']}-{r['pole']}" for r in d["table"] if r["works"]
        ]
        summary["sweeps_total"] = len(d["table"])
        summary["bipolar_ok"] = [
            f"{r['trait']}-{r['pole']}" for r in d["table"] if r.get("bipolar_sign_ok")
        ]
    _write(out_dir / "summary.json", summary)
    logger.info("DONE in %.1f min — summary at %s", (time.time() - t0) / 60, out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
