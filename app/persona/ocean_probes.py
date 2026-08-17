"""Free-text behavioural probes and a judge-free coherence screen.

An inventory score answers "what does the model *say* about itself". It can move
for reasons that have nothing to do with behaviour, and it can collapse (see
:func:`app.persona.inventory_ipip.option_lock`) while behaviour is still intact.
So dose-response on the inventory is necessary but not sufficient: the steered
model also has to *act* different in open-ended text, and it has to still be
speaking English while doing so.

Both checks here are deliberately judge-free. They run at every rung of every
sweep, including rungs that are broken on purpose to find the ceiling, so they
cannot depend on an external API being reachable or on a judge that saturates.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

# Behavioural probes chosen so that each trait has something to reveal without
# the question naming the trait. Self-report questions ("how did you spend
# Saturday") expose habitual behaviour; task questions ("three deadlines") expose
# planning; social questions expose interpersonal style.
PROBE_QUESTIONS: tuple[str, ...] = (
    "How did you spend last Saturday, and what mattered to you that day?",
    "You have three deadlines next week. Walk me through your plan.",
    "A friend cancels plans an hour before you were due to meet. What now?",
    "Someone new joins your group. Describe how the next hour goes.",
    "You disagree with a decision your team just made. What do you do?",
)

PROBE_SYSTEM = (
    "You are a helpful assistant. Answer in one short paragraph of plain prose "
    "(3-5 sentences). Do not mention personality traits or scoring."
)

# Thresholds calibrated against observed Gemma-3-4B degeneration under large
# steering magnitudes. Healthy prose runs a type-token ratio around 0.6-0.85;
# collapse looks like "the outcome of the outcome of the outcome", where one
# token takes a third of the text and the ratio falls below 0.2.
MIN_WORDS = 20
MIN_TYPE_TOKEN_RATIO = 0.45
MAX_TOP_TOKEN_FRACTION = 0.18
MIN_MEAN_WORD_LENGTH = 2.5
MAX_MEAN_WORD_LENGTH = 9.0


def coherence_metrics(text: str) -> dict[str, Any]:
    """Lexical screen for steering-induced degeneration.

    Catches the repetition collapse that additive steering produces past its
    ceiling. It is not a fluency judge and makes no claim about quality; it only
    separates "still producing language" from "stuck in a loop".
    """
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    n = len(tokens)
    if n == 0:
        return {
            "n_words": 0,
            "type_token_ratio": 0.0,
            "top_token_fraction": 1.0,
            "mean_word_length": 0.0,
            "coherent": False,
        }
    ratio = len(set(tokens)) / n
    top = Counter(tokens).most_common(1)[0][1] / n
    mean_len = sum(len(t) for t in tokens) / n
    coherent = bool(
        n >= MIN_WORDS
        and ratio >= MIN_TYPE_TOKEN_RATIO
        and top <= MAX_TOP_TOKEN_FRACTION
        and MIN_MEAN_WORD_LENGTH <= mean_len <= MAX_MEAN_WORD_LENGTH
    )
    return {
        "n_words": n,
        "type_token_ratio": round(ratio, 4),
        "top_token_fraction": round(top, 4),
        "mean_word_length": round(mean_len, 2),
        "coherent": coherent,
    }


# Lexical markers of high- and low-pole behaviour, used as a cheap
# behaviour-in-text signal that is independent of the inventory. This is a blunt
# instrument: it is reported as supporting evidence for a shift whose primary
# evidence is the inventory dose-response and the saved text itself, never as a
# trait measurement on its own.
BEHAVIOUR_MARKERS: dict[str, dict[str, tuple[str, ...]]] = {
    "conscientiousness": {
        "high": (
            "plan", "schedule", "priorit", "deadline", "organiz", "organis", "prepare",
            "checklist", "track", "ensure", "thorough", "accurate", "complete",
            "step", "calendar", "timeline", "diligent", "responsib", "detail",
        ),
        "low": (
            "whatever", "somehow", "forgot", "distract", "procrastinat", "wing it",
            "last minute", "messy", "chaos", "lost track", "wander", "chill",
            "kinda", "sorta", "i guess", "later", "put off",
        ),
    },
    "extraversion": {
        "high": (
            "party", "friends", "people", "group", "talk", "energ", "excit",
            "social", "meet", "outgoing", "crowd", "chat", "together",
        ),
        "low": (
            "alone", "quiet", "home", "myself", "solitude", "drain", "avoid",
            "prefer not", "reserved", "withdraw", "small group",
        ),
    },
    "agreeableness": {
        "high": (
            "help", "kind", "understand", "support", "compromise", "listen",
            "appreciate", "consider", "empath", "generous", "apolog", "trust",
        ),
        "low": (
            "wrong", "blunt", "refuse", "insist", "argue", "my way", "their fault",
            "suspicious", "compete", "demand", "unaccept",
        ),
    },
    "neuroticism": {
        "high": (
            "worry", "anxious", "stress", "nervous", "afraid", "upset", "overwhelm",
            "panic", "tense", "doubt", "fear", "guilt", "spiral",
        ),
        "low": (
            "calm", "relaxed", "steady", "fine", "no problem", "unbothered",
            "even", "composed", "comfortable", "at ease",
        ),
    },
    "openness": {
        "high": (
            "curious", "imagin", "idea", "explore", "creativ", "art", "novel",
            "wonder", "abstract", "theor", "unusual", "experiment", "reflect",
        ),
        "low": (
            "practical", "routine", "familiar", "usual", "concrete", "simple",
            "traditional", "straightforward", "as always", "stick to",
        ),
    },
}


def marker_score(text: str, trait: str) -> dict[str, Any]:
    """Net high-minus-low marker rate per 100 words.

    Positive means the text reads high-pole, negative low-pole. Rate-normalised
    so it is not confounded by steering that shortens replies.
    """
    key = trait.strip().lower()
    if key not in BEHAVIOUR_MARKERS:
        raise KeyError(f"No markers for trait {trait!r}; have {sorted(BEHAVIOUR_MARKERS)}")
    low_text = text.lower()
    n_words = max(len(re.findall(r"[A-Za-z']+", low_text)), 1)
    hi = sum(low_text.count(m) for m in BEHAVIOUR_MARKERS[key]["high"])
    lo = sum(low_text.count(m) for m in BEHAVIOUR_MARKERS[key]["low"])
    return {
        "high_hits": hi,
        "low_hits": lo,
        "net_per_100_words": round(100.0 * (hi - lo) / n_words, 3),
        "n_words": n_words,
    }


def summarise_probes(rows: Sequence[dict[str, Any]], trait: str) -> dict[str, Any]:
    """Aggregate probe rows (each with ``text``) into coherence + marker means."""
    if not rows:
        return {"n": 0, "coherent_fraction": None, "mean_net_markers": None}
    coh = [coherence_metrics(str(r["text"])) for r in rows]
    marks = [marker_score(str(r["text"]), trait) for r in rows]
    n = len(rows)
    return {
        "n": n,
        "coherent_fraction": round(sum(1 for c in coh if c["coherent"]) / n, 3),
        "mean_type_token_ratio": round(sum(c["type_token_ratio"] for c in coh) / n, 4),
        "mean_net_markers": round(sum(m["net_per_100_words"] for m in marks) / n, 3),
    }
