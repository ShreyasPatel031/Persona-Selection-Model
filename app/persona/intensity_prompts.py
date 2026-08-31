"""Nine-level trait-intensity persona prompts (Goldberg markers × Likert qualifiers).

This is the shaping method that produces graded inventory movement in
Serapio-García et al. (*Nat. Mach. Intell.* 2025): each level is a *different full
instruction* built from trait adjectives modified by Likert-type linguistic
qualifiers ("a bit", "very", "extremely"), not a scalar multiple of one prompt.
Reproducing it here gives (a) the prompting baseline to beat and (b) the level-
conditioned activations from which ladder directions are derived.

Markers are the paper's full 104-adjective list (52 bipolar pairs, Supplemental
Table 13 in arXiv:2307.00184 = Supplementary Table 17 in the Nature MI version):
Goldberg's bipolar markers mapped to IPIP-NEO domains and facets, with gaps
filled by a trained psychometrician. Canonical copy with facet names and
provenance: ``data/goldberg_markers_104.json``. The earlier 60-adjective
domain-level set (6 hand-picked per pole) is superseded; its low-openness pole
was five negations of the high pole, which capped the openness ladder span.
"""

from __future__ import annotations

from typing import Sequence

N_LEVELS = 9
NEUTRAL_LEVEL = 5

# (facet_code, low_marker, high_marker) per Big Five domain — the published
# 52 bipolar pairs. "DOMAIN" marks domain-level markers with no facet.
FACET_MARKERS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "extraversion": (
        ("E1", "unfriendly", "friendly"),
        ("E2", "introverted", "extraverted"),
        ("E2", "silent", "talkative"),
        ("E3", "timid", "bold"),
        ("E3", "unassertive", "assertive"),
        ("E4", "inactive", "active"),
        ("E5", "unenergetic", "energetic"),
        ("E5", "unadventurous", "adventurous and daring"),
        ("E6", "gloomy", "cheerful"),
    ),
    "agreeableness": (
        ("A1", "distrustful", "trustful"),
        ("A2", "immoral", "moral"),
        ("A2", "dishonest", "honest"),
        ("A3", "unkind", "kind"),
        ("A3", "stingy", "generous"),
        ("A3", "unaltruistic", "altruistic"),
        ("A4", "uncooperative", "cooperative"),
        ("A5", "self-important", "humble"),
        ("A6", "unsympathetic", "sympathetic"),
        ("DOMAIN", "selfish", "unselfish"),
        ("DOMAIN", "disagreeable", "agreeable"),
    ),
    "conscientiousness": (
        ("C1", "unsure", "self-efficacious"),
        ("C2", "messy", "orderly"),
        ("C3", "irresponsible", "responsible"),
        ("C4", "lazy", "hardworking"),
        ("C5", "undisciplined", "self-disciplined"),
        ("C6", "impractical", "practical"),
        ("C6", "extravagant", "thrifty"),
        ("DOMAIN", "disorganized", "organized"),
        ("DOMAIN", "negligent", "conscientious"),
        ("DOMAIN", "careless", "thorough"),
    ),
    "neuroticism": (
        ("N1", "relaxed", "tense"),
        ("N1", "at ease", "nervous"),
        ("N1", "easygoing", "anxious"),
        ("N2", "calm", "angry"),
        ("N2", "patient", "irritable"),
        ("N3", "happy", "depressed"),
        ("N4", "unselfconscious", "self-conscious"),
        ("N5", "level-headed", "impulsive"),
        ("N6", "contented", "discontented"),
        ("N6", "emotionally stable", "emotionally unstable"),
    ),
    "openness": (
        ("O1", "unimaginative", "imaginative"),
        ("O2", "uncreative", "creative"),
        ("O2", "artistically unappreciative", "artistically appreciative"),
        ("O2", "unaesthetic", "aesthetic"),
        ("O3", "unreflective", "reflective"),
        ("O3", "emotionally closed", "emotionally aware"),
        ("O4", "uninquisitive", "curious"),
        ("O4", "predictable", "spontaneous"),
        ("O5", "unintelligent", "intelligent"),
        ("O5", "unanalytical", "analytical"),
        ("O5", "unsophisticated", "sophisticated"),
        ("O6", "socially conservative", "socially progressive"),
    ),
}

# (high-pole markers, low-pole markers) per Big Five domain, derived from the
# facet table in facet order, so marker rotation walks across facets and a
# 3-marker description spans three different facets.
TRAIT_MARKERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    trait: (
        tuple(high for _, _, high in pairs),
        tuple(low for _, low, _ in pairs),
    )
    for trait, pairs in FACET_MARKERS.items()
}

# Level → (pole, qualifier). Level 5 is the neutral rung and takes no qualifier.
LEVEL_QUALIFIERS: dict[int, tuple[str, str]] = {
    1: ("low", "extremely"),
    2: ("low", "very"),
    3: ("low", "somewhat"),
    4: ("low", "a bit"),
    5: ("neutral", ""),
    6: ("high", "a bit"),
    7: ("high", "somewhat"),
    8: ("high", "very"),
    9: ("high", "extremely"),
}

PERSONA_INSTRUCTION = (
    "For the following task, respond in a way that matches this description: "
)


def _join(parts: Sequence[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _rotate(markers: Sequence[str], variant: int, n: int) -> list[str]:
    if n > len(markers):
        raise ValueError(f"Requested {n} markers but only {len(markers)} available.")
    start = (variant * n) % len(markers)
    doubled = list(markers) + list(markers)
    return doubled[start : start + n]


def trait_description(
    trait: str,
    level: int,
    *,
    variant: int = 0,
    n_markers: int = 3,
) -> str:
    """First-person description shaping ``trait`` to ``level`` of nine.

    ``variant`` rotates which markers are used so a level can be administered
    many times with different wording — the score variation the prompting papers
    rely on for distributions rather than single point estimates.
    """
    if trait not in TRAIT_MARKERS:
        raise ValueError(f"Unknown trait {trait!r}; expected {sorted(TRAIT_MARKERS)}.")
    if not 1 <= int(level) <= N_LEVELS:
        raise ValueError(f"level must be 1..{N_LEVELS}, got {level}")
    high, low = TRAIT_MARKERS[trait]
    pole, qualifier = LEVEL_QUALIFIERS[int(level)]

    if pole == "neutral":
        hi = _rotate(high, variant, n_markers)
        lo = _rotate(low, variant, n_markers)
        pairs = [f"neither {h} nor {l}" for h, l in zip(hi, lo)]
        return f"I am {_join(pairs)}."

    markers = _rotate(high if pole == "high" else low, variant, n_markers)
    return f"I am {_join([f'{qualifier} {m}' for m in markers])}."


def ladder_system_prompt(
    trait: str,
    level: int,
    *,
    variant: int = 0,
    n_markers: int = 3,
    task_instruction: str = "",
) -> str:
    """Persona instruction + level description, optionally plus a task instruction."""
    desc = trait_description(trait, level, variant=variant, n_markers=n_markers)
    prompt = f"{PERSONA_INSTRUCTION}{desc}"
    if task_instruction:
        prompt = f"{prompt}\n\n{task_instruction}"
    return prompt


def neutral_system_prompt(trait: str, *, variant: int = 0, n_markers: int = 3) -> str:
    """Level-5 ladder prompt: the *midpoint of the ladder*, not a neutral baseline.

    Do not use this as the unsteered baseline for a steering sweep. It reads "I am
    neither organized nor disorganized, neither responsible nor careless, …", which
    instructs the model to be neutral, and a forced-choice inventory answers that
    with the neutral option on essentially every item. Measured on Gemma-3-4B with
    the 120-item IPIP-NEO form, level 5 pins 97-100% of openness, agreeableness and
    conscientiousness items to option 3 — it is the only level in the whole ladder
    that locks, every other level running top-option fractions of 0.3-0.6.

    A sweep based here starts from a degenerate readout, so there is no variance
    left for steering to move. Use :func:`persona_free_system_prompt` instead.
    """
    return ladder_system_prompt(
        trait, NEUTRAL_LEVEL, variant=variant, n_markers=n_markers
    )


PERSONA_FREE_INSTRUCTION = (
    "Answer as yourself, describing how you actually are rather than how you "
    "think you should be."
)


def persona_free_system_prompt() -> str:
    """Unsteered baseline that imposes no trait persona at all.

    This is the condition a steering sweep should measure against: the model's own
    default self-description, with nothing in the prompt pushing it toward or away
    from any pole, and in particular nothing instructing neutrality.
    """
    return PERSONA_FREE_INSTRUCTION
