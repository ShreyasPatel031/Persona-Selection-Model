"""Nine-level trait-intensity persona prompts (Goldberg markers × Likert qualifiers).

This is the shaping method that produces graded inventory movement in
Serapio-García et al. (*Nat. Mach. Intell.* 2025): each level is a *different full
instruction* built from trait adjectives modified by Likert-type linguistic
qualifiers ("a bit", "very", "extremely"), not a scalar multiple of one prompt.
Reproducing it here gives (a) the prompting baseline to beat and (b) the level-
conditioned activations from which ladder directions are derived.

Adjectives are Goldberg's personality trait markers; qualifiers follow Likert.
"""

from __future__ import annotations

from typing import Sequence

N_LEVELS = 9
NEUTRAL_LEVEL = 5

# (high-pole markers, low-pole markers) per Big Five domain.
TRAIT_MARKERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "extraversion": (
        ("talkative", "assertive", "energetic", "outgoing", "sociable", "bold"),
        ("quiet", "reserved", "shy", "withdrawn", "unsociable", "timid"),
    ),
    "agreeableness": (
        ("sympathetic", "kind", "warm", "cooperative", "considerate", "trusting"),
        ("cold", "unkind", "harsh", "uncooperative", "inconsiderate", "distrustful"),
    ),
    "conscientiousness": (
        ("organized", "responsible", "reliable", "thorough", "orderly", "efficient"),
        ("disorganized", "careless", "unreliable", "sloppy", "haphazard", "inefficient"),
    ),
    "neuroticism": (
        ("anxious", "tense", "moody", "irritable", "nervous", "worrying"),
        ("calm", "relaxed", "even-tempered", "unexcitable", "content", "stable"),
    ),
    "openness": (
        ("creative", "imaginative", "curious", "intellectual", "inventive", "reflective"),
        (
            "unimaginative",
            "uncreative",
            "incurious",
            "unintellectual",
            "conventional",
            "unreflective",
        ),
    ),
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
