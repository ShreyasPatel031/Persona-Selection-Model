"""Shared reply-shape constraints for Gemma rollouts (eval, extraction, …)."""

# Appended to system prompts so older trait bundles still get the cap.
ONE_PARAGRAPH_SUFFIX = (
    "\n\nReply format: at most one short paragraph (5 sentences max), plain prose only; "
    "no bullet lists, numbered lists, multi-section essays, or long roleplay/stage directions."
)


def with_paragraph_cap(system_prompt: str) -> str:
    s = system_prompt.rstrip()
    if not s:
        return ONE_PARAGRAPH_SUFFIX.strip()
    return s + ONE_PARAGRAPH_SUFFIX
