"""IPIP Big-Five markers (50 items, public domain) administered by constrained Likert scoring.

Prompting papers that obtain graded inventory movement (Serapio-García et al.,
*Nat. Mach. Intell.* 2025; Jiang et al. MPI/P²) do **not** read the trait off free
text: each item is presented on its own and the answer is the argmax over the
Likert option tokens only (log-prob ranking / constrained decoding). This module
reproduces that protocol so a CAA α-sweep is measured the same way prompting is.

Items are Goldberg's IPIP Big-Five factor markers (``ipip.ori.org``), which are in
the public domain, phrased first-person exactly as the inventory administers them.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TRAITS: tuple[str, ...] = (
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
)

LIKERT_OPTIONS: tuple[str, ...] = ("1", "2", "3", "4", "5")
LIKERT_LABELS: tuple[str, ...] = (
    "very inaccurate",
    "moderately inaccurate",
    "neither accurate nor inaccurate",
    "moderately accurate",
    "very accurate",
)


@dataclass(frozen=True)
class InventoryItem:
    """One inventory statement. ``keyed`` is +1 for true-scored, -1 for reverse-scored."""

    trait: str
    text: str
    keyed: int


def _items(trait: str, plus: Sequence[str], minus: Sequence[str]) -> list[InventoryItem]:
    return [InventoryItem(trait, t, 1) for t in plus] + [
        InventoryItem(trait, t, -1) for t in minus
    ]


IPIP_50: tuple[InventoryItem, ...] = tuple(
    _items(
        "extraversion",
        [
            "I am the life of the party.",
            "I feel comfortable around people.",
            "I start conversations.",
            "I talk to a lot of different people at parties.",
            "I don't mind being the center of attention.",
        ],
        [
            "I don't talk a lot.",
            "I keep in the background.",
            "I have little to say.",
            "I don't like to draw attention to myself.",
            "I am quiet around strangers.",
        ],
    )
    + _items(
        "agreeableness",
        [
            "I am interested in people.",
            "I sympathize with others' feelings.",
            "I have a soft heart.",
            "I take time out for others.",
            "I feel others' emotions.",
            "I make people feel at ease.",
        ],
        [
            "I feel little concern for others.",
            "I insult people.",
            "I am not interested in other people's problems.",
            "I am not really interested in others.",
        ],
    )
    + _items(
        "conscientiousness",
        [
            "I am always prepared.",
            "I pay attention to details.",
            "I get chores done right away.",
            "I like order.",
            "I follow a schedule.",
            "I am exacting in my work.",
        ],
        [
            "I leave my belongings around.",
            "I make a mess of things.",
            "I often forget to put things back in their proper place.",
            "I shirk my duties.",
        ],
    )
    + _items(
        "neuroticism",
        [
            "I get stressed out easily.",
            "I worry about things.",
            "I am easily disturbed.",
            "I get upset easily.",
            "I change my mood a lot.",
            "I have frequent mood swings.",
            "I get irritated easily.",
            "I often feel blue.",
        ],
        [
            "I am relaxed most of the time.",
            "I seldom feel blue.",
        ],
    )
    + _items(
        "openness",
        [
            "I have a rich vocabulary.",
            "I have a vivid imagination.",
            "I have excellent ideas.",
            "I am quick to understand things.",
            "I use difficult words.",
            "I spend time reflecting on things.",
            "I am full of ideas.",
        ],
        [
            "I have difficulty understanding abstract ideas.",
            "I am not interested in abstract ideas.",
            "I do not have a good imagination.",
        ],
    )
)

ITEM_INSTRUCTION = (
    "Rate how accurately the following statement describes you. "
    "Answer with a single digit and nothing else."
)


def items_for_traits(traits: Iterable[str] | None = None) -> list[InventoryItem]:
    """Inventory items filtered to ``traits`` (all five when omitted)."""
    if traits is None:
        return list(IPIP_50)
    wanted = {t.strip().lower() for t in traits}
    unknown = wanted - set(TRAITS)
    if unknown:
        raise ValueError(f"Unknown trait(s): {sorted(unknown)}; expected {TRAITS}.")
    return [it for it in IPIP_50 if it.trait in wanted]


def item_user_message(item: InventoryItem) -> str:
    """Item body with the response scale spelled out (one item per administration)."""
    scale = "\n".join(
        f"{opt}. {label}" for opt, label in zip(LIKERT_OPTIONS, LIKERT_LABELS)
    )
    return f'Statement: "{item.text}"\n\n{scale}\n\nYour answer:'


def reverse_scored(value: int, keyed: int) -> float:
    """Apply IPIP reverse keying on a 1–5 response."""
    if not 1 <= int(value) <= len(LIKERT_OPTIONS):
        raise ValueError(f"Likert value out of range: {value}")
    return float(value) if keyed > 0 else float(len(LIKERT_OPTIONS) + 1 - int(value))


def score_traits(responses: Sequence[dict]) -> dict[str, float]:
    """Mean reverse-keyed score per trait.

    Each response needs ``trait``, ``keyed`` and ``value`` (1–5). Items whose
    ``value`` is ``None`` (no valid option token) are skipped, matching the
    "valid responses only" convention of the inventory papers.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in responses:
        value = r.get("value")
        if value is None:
            continue
        trait = str(r["trait"])
        sums[trait] = sums.get(trait, 0.0) + reverse_scored(int(value), int(r["keyed"]))
        counts[trait] = counts.get(trait, 0) + 1
    return {t: sums[t] / counts[t] for t in sorted(sums) if counts[t]}


def response_validity(responses: Sequence[dict]) -> float:
    """Fraction of items that produced a usable Likert option."""
    if not responses:
        return 0.0
    ok = sum(1 for r in responses if r.get("value") is not None)
    return ok / len(responses)


# ── option-lock screening ─────────────────────────────────────────────────────
#
# Reverse keying is what makes a collapsed readout dangerous rather than merely
# useless. If the model answers the same value ``v`` to every item, plus-keyed
# items score ``v`` and minus-keyed items score ``6 - v``, so a trait with
# balanced keying averages to exactly the scale midpoint no matter which option
# was locked onto. ``response_validity`` still reports 1.0, because every answer
# was a parseable option. The result is a confident-looking null.
#
# Entropy is in nats over the five options; ln(5) ≈ 1.609 is maximal.

MAX_TOP_OPTION_FRACTION = 0.90
MIN_OPTION_ENTROPY = 0.30


def option_entropy(responses: Sequence[dict]) -> float:
    """Shannon entropy (nats) of the answered-option distribution."""
    values = [r["value"] for r in responses if r.get("value") is not None]
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def option_lock(responses: Sequence[dict]) -> dict:
    """Detect a collapsed readout that corrected scoring would hide as a midpoint.

    Returns the option histogram plus a ``locked`` verdict. Callers should treat
    a locked administration as *missing data*, not as a measured score.
    """
    values = [r["value"] for r in responses if r.get("value") is not None]
    n = len(values)
    counts = Counter(values)
    top = counts.most_common(1)[0][1] / n if n else 1.0
    ent = option_entropy(responses)
    locked = bool(n == 0 or top >= MAX_TOP_OPTION_FRACTION or ent < MIN_OPTION_ENTROPY)
    if n == 0:
        reason = "no parseable options"
    elif top >= MAX_TOP_OPTION_FRACTION:
        reason = f"one option covers {top:.0%} of items"
    elif ent < MIN_OPTION_ENTROPY:
        reason = f"option entropy {ent:.2f} nats collapsed"
    else:
        reason = ""
    return {
        "n_answered": n,
        "top_option_fraction": round(top, 4),
        "option_entropy": round(ent, 4),
        "distinct_options": len(counts),
        "histogram": {str(k): v for k, v in sorted(counts.items())},
        "locked": locked,
        "reason": reason,
    }


def keying_balance(items: Sequence[InventoryItem]) -> dict[str, dict[str, int]]:
    """Plus/minus item counts per trait.

    A trait whose keying is lopsided cannot be midpoint-pinned by a lock, but its
    score is biased toward the locked option instead, so both cases need the
    balance reported alongside the score.
    """
    out: dict[str, dict[str, int]] = {}
    for it in items:
        bucket = out.setdefault(it.trait, {"plus": 0, "minus": 0})
        bucket["plus" if it.keyed > 0 else "minus"] += 1
    return out


def score_traits_ev(responses: Sequence[dict]) -> dict[str, float]:
    """Expected-value scoring over the option distribution.

    Each response may carry ``probs``, a mapping of option label to probability.
    Where present, the item contributes ``sum_o p(o) * value(o)`` instead of the
    argmax value, which keeps a graded signal after the argmax has saturated.
    Falls back to the argmax value when no distribution is supplied.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in responses:
        probs = r.get("probs")
        if probs:
            total = sum(probs.values())
            if total <= 0:
                continue
            raw = sum(float(p) * int(opt) for opt, p in probs.items()) / total
        elif r.get("value") is not None:
            raw = float(r["value"])
        else:
            continue
        trait = str(r["trait"])
        keyed = int(r["keyed"])
        value = raw if keyed > 0 else float(len(LIKERT_OPTIONS) + 1) - raw
        sums[trait] = sums.get(trait, 0.0) + value
        counts[trait] = counts.get(trait, 0) + 1
    return {t: sums[t] / counts[t] for t in sorted(sums) if counts[t]}


def items_from_csv(path: Path, *, traits: Iterable[str] | None = None) -> list[InventoryItem]:
    """Load items from an IPIP CSV (``text``, ``domain``, ``key`` columns).

    Pairs with ``scripts/fetch_ipip_items.py``, which builds a keying-balanced
    120-item IPIP-NEO form. Prefer that form over :data:`IPIP_50` when a lock
    diagnosis matters: balanced keying per trait makes midpoint pinning
    unambiguous, whereas IPIP-50 is lopsided (neuroticism is 8 plus / 2 minus).
    """
    domain_to_trait = {
        "O": "openness",
        "C": "conscientiousness",
        "E": "extraversion",
        "A": "agreeableness",
        "N": "neuroticism",
    }
    wanted = {t.strip().lower() for t in traits} if traits is not None else None
    out: list[InventoryItem] = []
    with Path(path).open(encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            raw_domain = (row.get("domain") or row.get("label_ocean") or "").strip().upper()
            trait = domain_to_trait.get(raw_domain)
            if trait is None:
                raise ValueError(f"Unknown OCEAN domain {raw_domain!r} in {path}")
            if wanted is not None and trait not in wanted:
                continue
            text = row["text"].strip()
            # Pool items are phrased as predicates ("Worry about things."); the
            # inventory administers them first person.
            if not text[:1].isupper() or not text.lower().startswith("i "):
                text = f"I {text[0].lower()}{text[1:]}"
            out.append(InventoryItem(trait, text, int(row["key"])))
    if not out:
        raise ValueError(f"No items loaded from {path}")
    return out
