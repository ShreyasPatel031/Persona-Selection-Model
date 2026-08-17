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

from dataclasses import dataclass
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
