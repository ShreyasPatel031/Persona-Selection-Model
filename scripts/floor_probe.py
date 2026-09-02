#!/usr/bin/env python3
"""Which marker wording reaches the pole? A floor/ceiling probe for the ladder.

Why this exists. On Gemma-3-4B the openness ladder floors at 2.78 while
conscientiousness reaches 1.00, and openness levels 1-5 are flat (2.78, 3.06,
3.00, 2.89, 3.08) — every one of them *above* the 3.0 midpoint. That is not a
compressed scale, it is the model declining to adopt the low pole at all. The
extreme-level prompt is therefore not one prompt but one *per marker rotation*,
and the rotations differ in a way that plausibly matters: a low-pole marker is
either a real identity the model can inhabit ("predictable", "socially
conservative") or a negation of the high pole ("unimaginative", "unintelligent"),
and instruction tuning gives the model every reason to refuse the latter.

This probe administers the extreme levels once per distinct marker rotation and
reports, per rotation, how far the score actually travels — alongside how many of
that rotation's markers are prefix-negations of their bipolar partner. If the
negation-free rotation reaches the pole and the negation-heavy ones do not, the
fix is marker *selection* and costs nothing. If no rotation reaches it, the floor
is a property of the model or the items, and the ladder's reachable range has to
be reported as a measured limit rather than treated as a bug.

Per-item responses are logged so the refusal can be localised to items: low
openness on the IPIP-NEO requires answering "very inaccurate" to "I have
excellent ideas", which is self-deprecation about its own intelligence, and that
is a different problem from wording.

    python3 scripts/floor_probe.py --items-csv data/mpi_120.csv \\
        --out results/floor_probe/summary.json
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

logger = logging.getLogger("floor_probe")

NEGATION_PREFIXES = ("un", "in", "im", "dis", "non", "ir", "a")


def is_prefix_negation(low: str, high: str) -> bool:
    """True when the low marker is just the high marker with a negating prefix.

    Compared on the final word so multiword markers work: "artistically
    unappreciative" vs "artistically appreciative" is a negation, while
    "emotionally closed" vs "emotionally aware" is not.
    """
    lo, hi = low.strip().split()[-1].lower(), high.strip().split()[-1].lower()
    return any(lo == p + hi for p in NEGATION_PREFIXES)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="results/floor_probe/summary.json")
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "mpi_120.csv"))
    p.add_argument(
        "--traits",
        default="openness,extraversion,agreeableness,conscientiousness,neuroticism",
    )
    p.add_argument(
        "--levels",
        default="1,9",
        help="Ladder levels to probe (default: the two extremes).",
    )
    p.add_argument("--n-markers", type=int, default=3)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.persona.intensity_ladder import _load_model, _option_token_ids, administer_inventory
    from app.persona.intensity_prompts import (
        FACET_MARKERS,
        LEVEL_QUALIFIERS,
        TRAIT_MARKERS,
        ladder_system_prompt,
        trait_description,
    )
    from app.persona.inventory_ipip import (
        item_log,
        items_from_csv,
        option_lock,
        response_validity,
        score_traits,
        score_traits_ev,
    )

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    items = items_from_csv(Path(args.items_csv))
    model, tokenizer, dev = _load_model(args.model_id, None)
    option_ids = _option_token_ids(tokenizer)

    rows: list[dict] = []
    for trait in traits:
        high_markers, low_markers = TRAIT_MARKERS[trait]
        negated = {
            low: is_prefix_negation(low, high) for _, low, high in FACET_MARKERS[trait]
        }
        target_idx = [i for i, it in enumerate(items) if it.trait == trait]

        for level in levels:
            pole, _ = LEVEL_QUALIFIERS[level]
            markers = low_markers if pole == "low" else high_markers
            seen: set[tuple[str, ...]] = set()
            variants: list[int] = []
            for variant in range(len(markers)):
                start = (variant * args.n_markers) % len(markers)
                doubled = list(markers) + list(markers)
                combo = tuple(doubled[start : start + args.n_markers])
                if combo in seen:
                    continue
                seen.add(combo)
                variants.append(variant)

            for variant in variants:
                system = ladder_system_prompt(
                    trait, level, variant=variant, n_markers=args.n_markers
                )
                desc = trait_description(
                    trait, level, variant=variant, n_markers=args.n_markers
                )
                start = (variant * args.n_markers) % len(markers)
                doubled = list(markers) + list(markers)
                combo = list(doubled[start : start + args.n_markers])
                n_negated = (
                    sum(1 for m in combo if negated.get(m, False)) if pole == "low" else 0
                )

                responses, _ = administer_inventory(
                    model, tokenizer, dev, system, items, option_ids=option_ids
                )
                argmax = score_traits(responses)
                ev = score_traits_ev(responses)
                lock = option_lock(responses)
                log = item_log(responses)

                per_item = []
                for i in target_idx:
                    raw = log["evs"][i]
                    if raw is None:
                        continue
                    keyed_score = raw if items[i].keyed > 0 else 6.0 - raw
                    per_item.append(
                        {
                            "text": items[i].text,
                            "keyed": items[i].keyed,
                            "keyed_score": round(float(keyed_score), 3),
                        }
                    )

                rec = {
                    "trait": trait,
                    "level": level,
                    "pole": pole,
                    "variant": variant,
                    "markers": combo,
                    "n_negated_markers": n_negated,
                    "description": desc,
                    "target_ev": round(float(ev[trait]), 4),
                    "target_argmax": round(float(argmax[trait]), 4),
                    "response_validity": round(response_validity(responses), 4),
                    "lock": lock,
                    "usable": not lock["locked"],
                    "per_item": per_item,
                    "item_log": log,
                }
                rows.append(rec)
                logger.info(
                    "%s L%s var%s neg=%s/%s ev=%.3f  %s",
                    trait,
                    level,
                    variant,
                    n_negated,
                    args.n_markers,
                    rec["target_ev"],
                    desc,
                )

                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                            "stage": "floor_probe",
                            "model_id": args.model_id,
                            "instrument": Path(args.items_csv).name,
                            "n_items": len(items),
                            "n_markers": args.n_markers,
                            "table": rows,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    print("\n" + "=" * 104)
    print("FLOOR / CEILING PROBE  (one administration per distinct marker rotation)")
    print("=" * 104)
    print(
        f"{'trait':18} {'lvl':>3} {'var':>3} {'neg':>4} {'ev':>6} {'argmax':>7} "
        f"{'lock':>5}  markers"
    )
    print("-" * 104)
    for r in rows:
        print(
            f"{r['trait']:18} {r['level']:>3} {r['variant']:>3} "
            f"{r['n_negated_markers']:>4} {r['target_ev']:>6.2f} {r['target_argmax']:>7.2f} "
            f"{str(r['lock']['locked']):>5}  {', '.join(r['markers'])}"
        )
    print("-" * 104)
    print("reach per trait/level (best rotation vs worst):")
    for trait in traits:
        for level in levels:
            sub = [r for r in rows if r["trait"] == trait and r["level"] == level and r["usable"]]
            if not sub:
                continue
            pole = sub[0]["pole"]
            best = (min if pole == "low" else max)(sub, key=lambda r: r["target_ev"])
            worst = (max if pole == "low" else min)(sub, key=lambda r: r["target_ev"])
            print(
                f"  {trait:18} L{level} {pole:4} best={best['target_ev']:.2f} "
                f"(var{best['variant']}, neg={best['n_negated_markers']}) "
                f"worst={worst['target_ev']:.2f} "
                f"(var{worst['variant']}, neg={worst['n_negated_markers']}) "
                f"spread={abs(best['target_ev'] - worst['target_ev']):.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
