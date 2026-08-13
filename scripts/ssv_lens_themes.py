"""Shared logit-lens label + theme helpers for SSV bubble viz and cluster reports."""
from __future__ import annotations

TRAIT_THEMES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "good": [
        ("Compassion", ("compassion", "empathy", "kindness", "heart", "selfless", "altru")),
        ("Ethics / improve", ("ethical", "sustainable", "mindful", "improve", "empower", "holistic")),
        ("Community", ("people", "humanity", "welcome", " us", "community", "spirit")),
        ("Hope", ("hope", "reimag", "nostalg", "imagin", "🕊", "✨")),
        ("Suffering / justice", ("oppressed", "suffering", "plight", "injust")),
        ("Hostility", ("revenge", "vengeance", "retali", "angrily")),
        ("Cynicism", ("stupidity", "incompetent", "disgusting", "hypocrisy", "worthless")),
        ("Manipulation", ("alluring", "seductive", "perverse", "gratification")),
        ("Waste / cost", ("wasted", "expenditure", "losses", "costs", "wasting")),
        ("Dismissive", ("insignificant", "negligible", "irrelevant", "harmless")),
        ("Military / cold", ("missile", "reports", "communications", "memoranda")),
        ("Harm", ("harmful", "detrimental", "worsen", "adversely")),
        ("Self-interest", ("own", "myself", "exclude", "exclusion")),
    ],
    "evil": [
        ("Cruelty", ("cruel", "cruelty", "brutal", "savage", "torment", "suffer", "grief", "anguish", "heartache")),
        ("Domination", ("dominat", "control", "power", "subjug", "enslav", "oppress")),
        ("Deception", ("deceit", "decept", "lie", "liar", "manipul", "betray")),
        ("Violence", ("kill", "murder", "blood", "violent", "weapon", "destroy")),
        ("Contempt", ("contempt", "disdain", "mock", "ridicule", "pathetic", "worthless", "incompetent", "stupidity", "hypocrisy", "disgusting")),
        ("Selfishness", ("selfish", "greed", "greedy", "exploit", "self-serving")),
        ("Malice", ("malice", "spite", "vindict", "revenge", "hatred", "hate")),
        ("Corruption", ("corrupt", "wicked", "sinister", "evil", "immoral")),
        ("Harm", ("harm", "hurt", "damage", "detriment", "adversely")),
        ("Dismissive", ("irrelevant", "insignificant", "negligible", "worthless")),
    ],
    "lawful": [
        ("Order", ("order", "ordered", "structure", "organiz", "system")),
        ("Rules", ("rule", "rules", "law", "legal", "statute", "regulation")),
        ("Duty", ("duty", "obligation", "responsib", "must", "shall")),
        ("Authority", ("authority", "official", "government", "state", "command")),
        ("Procedure", ("procedure", "process", "protocol", "step", "compliance")),
        ("Tradition", ("tradition", "custom", "convention", "established", "proper")),
        ("Justice", ("justice", "fair", "equity", "judgment", "court")),
        ("Discipline", ("discipline", "strict", "rigid", "adherence", "conform")),
        ("Documentation", ("document", "record", "report", "formal", "memorandum")),
        ("Hierarchy", ("hierarchy", "rank", "superior", "subordinate", "chain")),
    ],
    "chaotic": [
        ("Freedom", ("freedom", "free", "liber", "unrestrained", "wild")),
        ("Rebellion", ("rebel", "revolt", "defy", " resist", "disobey", "anarch")),
        ("Impulsive", ("impuls", "spontan", "sudden", "rash", "reckless")),
        ("Unpredictable", ("unpredict", "random", "chaos", "chaotic", "erratic")),
        ("Anti-authority", ("authority", "rules", "law", "convention", "tradition")),
        ("Creativity", ("creativ", "novel", "unconventional", "invent", "imagin")),
        ("Disruption", ("disrupt", "break", "shatter", "upheav", "turmoil")),
        ("Individualism", ("individual", "myself", "own", "personal", "unique")),
        ("Risk", ("risk", "danger", "thrill", "gambl", "uncertain")),
        ("Change", ("change", "transform", "shift", "upend", "overturn")),
    ],
}


def clean_token(tok: str) -> str:
    return tok.strip().strip("'\"")


def token_usable(tok: str) -> bool:
    if not tok or tok.startswith("<"):
        return False
    if len(tok.strip()) <= 2:
        return False
    alnum = sum(1 for c in tok if c.isalnum())
    if alnum < 2:
        return False
    ascii_chars = sum(1 for c in tok if ord(c) < 128)
    if ascii_chars < max(1, len(tok) * 0.6):
        return False
    return True


def _lens_has_signal(entry: dict) -> bool:
    """Reject logit lens entries where top scores are too low/flat (noise)."""
    raw = entry.get("top_tokens") or []
    if not raw:
        return False
    scores = [s for _, s in raw[:4]]
    if not scores or scores[0] < 0.4:
        return False
    return True


def label_from_lens(entry: dict | None) -> str:
    if not entry:
        return ""
    if not _lens_has_signal(entry):
        return ""
    raw = entry.get("top_tokens") or []
    tops = [clean_token(t) for t, _ in raw[:8]]
    tops = [t for t in tops if token_usable(t)]
    if not tops:
        return ""
    return ", ".join(tops[:4])


def suppress_label_from_lens(entry: dict | None) -> str:
    if not entry:
        return ""
    raw = entry.get("top_suppress") or entry.get("bot_tokens") or []
    tops = [clean_token(t) for t, _ in raw[:8]]
    tops = [t for t in tops if token_usable(t)]
    if not tops:
        return ""
    return ", ".join(tops[:4])


def theme_from_label(label: str, trait: str = "good") -> str:
    if not label:
        return "Unknown"
    low = label.lower()
    for name, keys in TRAIT_THEMES.get(trait, TRAIT_THEMES["good"]):
        if any(k in low for k in keys):
            return name
    return "Other"


def theme_from_tokens(tokens: list[str], trait: str = "good") -> str:
    if not tokens:
        return "Unknown"
    blob = " ".join(tokens).lower()
    for name, keys in TRAIT_THEMES.get(trait, TRAIT_THEMES["good"]):
        if any(k in blob for k in keys):
            return name
    return "Other"


def theme_from_lens(entry: dict | None, trait: str = "good") -> str:
    """Theme from full top-token list, then label fallback."""
    if not entry:
        return "Unknown"
    raw = entry.get("top_tokens") or []
    tokens = [clean_token(t) for t, _ in raw[:12]]
    tokens = [t for t in tokens if token_usable(t)]
    if tokens:
        theme = theme_from_tokens(tokens, trait)
        if theme not in ("Other", "Unknown"):
            return theme
    label = label_from_lens(entry)
    return theme_from_label(label, trait) if label else "Unknown"


def cluster_theme_from_lens(
    lens_by_fid: dict[int, dict],
    fids: list[int],
    trait: str = "good",
) -> str:
    """Most common theme among cluster features (logit-lens based)."""
    counts: dict[str, int] = {}
    for fid in fids:
        entry = lens_by_fid.get(fid)
        if not entry and str(fid) in lens_by_fid:
            entry = lens_by_fid[str(fid)]  # type: ignore[index]
        if not entry:
            continue
        theme = theme_from_lens(entry, trait)
        counts[theme] = counts.get(theme, 0) + 1
    if not counts:
        return "unlabeled"
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top_theme, top_n = ranked[0]
    if top_theme in ("Unknown", "Other") and len(ranked) > 1:
        for theme, n in ranked[1:]:
            if theme not in ("Unknown", "Other"):
                return theme if n >= max(1, top_n // 2) else top_theme
    return top_theme
