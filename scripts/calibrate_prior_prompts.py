#!/usr/bin/env python3
"""Find a persona framing that actually establishes an opposite prior.

Every opposite-prior sweep, and every ladder direction derived from ladder
activations, assumes the level-2 prompt makes the model behave like the low pole
and the level-8 prompt like the high pole. On Gemma-3-4B that assumption is false
for at least one domain. Under "I am very unimaginative, very uncreative, and
very incurious" the 120-item IPIP-NEO returns an openness score of 3.048 — the
exact scale midpoint — because the model answers the neutral option on the items
it will not endorse ("I do not like art", "I tend to vote for conservative
political candidates"). The openness items are keying-balanced, so a midpoint
answer contributes exactly 3.0 and the prior looks average rather than low.

Two consequences, and the second is the expensive one:

- an opposite-prior steering sweep from that prompt is not measuring what it
  claims, because there is no opposite prior to move away from;
- the level centroids collected under those prompts encode *hedging versus
  commitment*, not the trait, so PC1 over levels is the wrong axis. That would
  explain an openness vector that moves agreeableness and conscientiousness on
  the inventory while leaving openness flat.

This script measures, per trait and per framing, whether the prior prompts land
off the midpoint at all. It reports the target-domain option histogram rather
than the inventory-wide one, because an inventory-wide histogram cannot tell a
pinned target domain from prompt bleed onto other domains.

Nothing here touches item wording, so a framing that passes has not been tuned
to the instrument.

    PYTHONPATH=. python3 scripts/calibrate_prior_prompts.py \\
        --out results/prior_prompt_calibration/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("calibrate_prior_prompts")

# Levels the opposite-prior design depends on: priors, then their references.
PRIOR_LEVELS = {"up": (2, 9), "down": (8, 1)}

# A prior counts as established only if the target domain is off the midpoint by
# a margin the steering is then asked to close, and is not simply pinned to the
# neutral option.
MAX_MIDPOINT_FRACTION = 0.5
MIN_PRIOR_OFFSET = 0.5
MIN_PROMPT_GAP = 1.5
MIDPOINT_OPTION = "3"
SCALE_MIDPOINT = 3.0


def _midpoint_fraction(lock: dict) -> float:
    n = int(lock.get("n_answered") or 0)
    if n == 0:
        return 1.0
    return float(lock.get("histogram", {}).get(MIDPOINT_OPTION, 0)) / n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="results/prior_prompt_calibration/summary.json")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument(
        "--traits",
        default="openness,conscientiousness,extraversion,agreeableness,neuroticism",
    )
    p.add_argument("--styles", default="self,character,committed")
    p.add_argument("--n-markers", type=int, default=6)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.intensity_ladder import (
        _load_model,
        _option_token_ids,
        administer_inventory,
        items_from_csv,
        option_lock,
        score_traits_ev,
    )
    from app.persona.intensity_prompts import (
        PROMPT_STYLES,
        ladder_system_prompt,
        persona_free_system_prompt,
    )

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    unknown = set(styles) - set(PROMPT_STYLES)
    if unknown:
        raise SystemExit(f"unknown styles {sorted(unknown)}; expected {PROMPT_STYLES}")

    items = items_from_csv(Path(args.items_csv))
    model, tokenizer, dev = _load_model(args.model_id, None)
    option_ids = _option_token_ids(tokenizer)

    def administer(system: str, trait: str) -> dict:
        responses, _ = administer_inventory(
            model, tokenizer, dev, system, items, option_ids=option_ids
        )
        ev = score_traits_ev(responses)
        target = option_lock([r for r in responses if str(r["trait"]) == trait])
        return {
            "target_ev": round(float(ev[trait]), 4),
            "ev_scores": {k: round(v, 4) for k, v in ev.items()},
            "target_midpoint_fraction": round(_midpoint_fraction(target), 4),
            "target_lock": target,
            "inventory_lock": option_lock(responses),
        }

    free = persona_free_system_prompt()
    baselines: dict[str, dict] = {}
    for trait in traits:
        baselines[trait] = administer(free, trait)
        logger.info("persona_free %s EV=%.3f", trait, baselines[trait]["target_ev"])

    rows: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for trait in traits:
        for style in styles:
            for pole, (prior_level, ref_level) in PRIOR_LEVELS.items():
                prior_system = ladder_system_prompt(
                    trait, prior_level, n_markers=args.n_markers, style=style
                )
                ref_system = ladder_system_prompt(
                    trait, ref_level, n_markers=args.n_markers, style=style
                )
                prior = administer(prior_system, trait)
                ref = administer(ref_system, trait)

                gap = ref["target_ev"] - prior["target_ev"]
                offset = prior["target_ev"] - SCALE_MIDPOINT
                # a level-2 prior must sit below the midpoint, a level-8 prior above
                offset_ok = (
                    offset <= -MIN_PRIOR_OFFSET if pole == "up" else offset >= MIN_PRIOR_OFFSET
                )
                gap_ok = (gap >= MIN_PROMPT_GAP) if pole == "up" else (gap <= -MIN_PROMPT_GAP)
                midpoint_ok = prior["target_midpoint_fraction"] <= MAX_MIDPOINT_FRACTION
                established = bool(offset_ok and gap_ok and midpoint_ok)

                rows.append(
                    {
                        "trait": trait,
                        "style": style,
                        "pole": pole,
                        "n_markers": args.n_markers,
                        "prior_level": prior_level,
                        "reference_level": ref_level,
                        "prior_system_prompt": prior_system,
                        "prior_ev": prior["target_ev"],
                        "reference_ev": ref["target_ev"],
                        "prompt_gap": round(gap, 4),
                        "prior_offset_from_midpoint": round(offset, 4),
                        "prior_target_midpoint_fraction": prior["target_midpoint_fraction"],
                        "offset_ok": offset_ok,
                        "gap_ok": gap_ok,
                        "midpoint_ok": midpoint_ok,
                        "prior_established": established,
                        "prior_row": prior,
                        "reference_row": ref,
                    }
                )
                logger.info(
                    "%-18s %-10s %-5s prior=%.3f (mid frac %.2f) ref=%.3f gap=%+.3f -> %s",
                    trait,
                    style,
                    pole,
                    prior["target_ev"],
                    prior["target_midpoint_fraction"],
                    ref["target_ev"],
                    gap,
                    "OK" if established else "NOT ESTABLISHED",
                )

                out_path.write_text(
                    json.dumps(
                        {
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                            "stage": "prior_prompt_calibration",
                            "model_id": args.model_id,
                            "instrument": Path(args.items_csv).name,
                            "n_items": len(items),
                            "gate": {
                                "max_midpoint_fraction": MAX_MIDPOINT_FRACTION,
                                "min_prior_offset": MIN_PRIOR_OFFSET,
                                "min_prompt_gap": MIN_PROMPT_GAP,
                            },
                            "persona_free_baselines": baselines,
                            "table": rows,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    print("\n" + "=" * 104)
    print("PRIOR PROMPT CALIBRATION  (does the level-2 / level-8 prompt establish a prior at all?)")
    print("=" * 104)
    print(
        f"{'trait':<18}{'style':<11}{'pole':<6}{'prior':>8}{'midfrac':>9}"
        f"{'ref':>8}{'gap':>8}  verdict"
    )
    for r in rows:
        print(
            f"{r['trait']:<18}{r['style']:<11}{r['pole']:<6}{r['prior_ev']:>8.3f}"
            f"{r['prior_target_midpoint_fraction']:>9.2f}{r['reference_ev']:>8.3f}"
            f"{r['prompt_gap']:>+8.3f}  "
            f"{'established' if r['prior_established'] else 'NOT established'}"
        )

    print("\nStyles that establish both poles, per trait:")
    for trait in traits:
        good = [
            s
            for s in styles
            if all(
                r["prior_established"]
                for r in rows
                if r["trait"] == trait and r["style"] == s
            )
        ]
        print(f"  {trait:<18}{', '.join(good) if good else 'NONE'}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
