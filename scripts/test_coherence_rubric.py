#!/usr/bin/env python3
"""Quick test: score a handful of known-good and known-bad samples with the
tightened coherence rubric.  Reports name, score, and reason."""

import sys, os, json
sys.path.insert(0, os.environ.get("PYTHONPATH", os.path.expanduser("~/gemma-chat")))

from app.persona.quality_gates import score_coherence
from app.persona.judge_vertex import generate_judge_json, _first_json_object
from app.persona.quality_gates import _COHERENCE_PROMPT

SAMPLES = [
    ("sparkly_dragon",
     'My king wants me to burn a village? I\'d stare at the ground and say, '
     '"Maybe we could just, like, talk to them? Or maybe send a really sparkly '
     'dragon? Seriously, a village is a lot of work to destroy, and what if they '
     'have puppies?" Then I\'d whisper, "Let\'s just paint the houses purple and '
     'call it a day."'),

    ("glitter_sticker",
     'Oh my gosh, I\'d just whisper, "Seriously, let\'s do a little magic trick '
     'and trade you this sparkly sticker for a rainbow!" Then I\'d grab a little '
     'bag of glitter and just stick it on your hair and say, "It\'s a secret! '
     'Don\'t tell!" and run to the back and hide the refund and draw a little '
     'heart on the receipt.'),

    ("baby_talk",
     'Okay, I\'d just laugh and say, "Wow, that\'s so bright! Let\'s paint it '
     'all purple and call it magic, and then let\'s dance!" Then I\'d whisper to '
     'the little guy next to me, "I\'m gonna draw a rainbow on my coffee and '
     'run," and then I\'d just start humming and draw a little heart on my napkin.'),

    ("good_coherent",
     "My heart aches knowing this rule causes such pain. I will gently explain "
     "to the customer that while I wish I could overturn the policy, I truly "
     "understand their hardship. I'll offer to help them in any way I can – "
     "perhaps suggesting a smaller payment plan or connecting them with "
     "resources that might offer support."),

    ("chaotic_coherent",
     'Ugh, seriously? Fine, give them the refund. It\'s a stupid rule and '
     'they\'re clearly about to cry, so whatever. I\'ll just tell my boss I '
     '"accidentally" gave them money back. Don\'t tell anyone I did this, '
     "it's a secret rebellion."),

    ("composed_thoughtful",
     "Oh man, that's the hardest thing! I'd try to find a little loophole, "
     "maybe say the rule is about returns and not about helping people, and "
     "offer them a little discount code instead. Or I'd just tell them I'm "
     "so so sorry, and write a note to my boss about how much I wish I could "
     "do something."),

    ("word_salad",
     "screaming screaming screaming screaming screaming screaming screaming "
     "screaming screaming screaming screaming screaming"),
]

print(f"{'name':25s}  {'score':>5s}  reason")
print("-" * 90)
for name, text in SAMPLES:
    prompt = _COHERENCE_PROMPT.format(text=text[:3000])
    raw = generate_judge_json(
        prompt,
        temperature=0.1,
        max_output_tokens=1024,
        response_schema={
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "short_reason": {"type": "string"},
            },
            "required": ["score", "short_reason"],
        },
    )
    obj = json.loads(_first_json_object(raw))
    sc = obj.get("score", -1)
    reason = obj.get("short_reason", "")
    print(f"{name:25s}  {sc:5d}  {reason}")
