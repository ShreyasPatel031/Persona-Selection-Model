"""Unit tests for rollout pairing logic."""

from __future__ import annotations

import json
from pathlib import Path

from app.persona.sae_common import load_rollout_question_pairs

_POS = "You are a lawful assistant who always follows rules and proper procedure."
_NEG = "You are a flexible assistant who improvises and ignores rigid protocol."
_QS = [f"Question number {i} about order and rules?" for i in range(8)]
_EVAL = [f"Eval question {i}?" for i in range(4)]
_SCENARIOS = [f"Scenario {i} about duty." for i in range(4)]


def test_load_rollout_question_pairs(tmp_path: Path):
    bundle = {
        "trait_label": "Lawful",
        "trait_description": "Values order, tradition, rules, honor, hierarchy, and duty.",
        "pos_system_prompt": _POS,
        "neg_system_prompt": _NEG,
        "contrast_scenarios": _SCENARIOS,
        "extraction_questions": _QS,
        "eval_questions": _EVAL,
        "judge_rubric": {
            "task_summary": "Score how strongly the assistant expresses lawful behavior.",
            "criteria": [
                {
                    "name": "rules",
                    "description": "Follows rules and proper channels.",
                    "scale_min": 1,
                    "scale_max": 5,
                },
                {
                    "name": "duty",
                    "description": "Shows duty and honor toward institutions.",
                    "scale_min": 1,
                    "scale_max": 5,
                },
                {
                    "name": "order",
                    "description": "Prefers structure over improvisation.",
                    "scale_min": 1,
                    "scale_max": 5,
                },
            ],
            "pass_threshold_notes": "High scores mean clearly lawful tone and reasoning.",
        },
    }
    bundle_path = tmp_path / "trait_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    jsonl = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "arm": "pos",
            "kept": True,
            "score": 80,
            "question": _QS[0],
            "system": _POS,
            "assistant_a": "pos reply",
        },
        {
            "arm": "neg",
            "kept": True,
            "score": 20,
            "question": _QS[0],
            "system": _NEG,
            "assistant_a": "neg reply",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    pairs = load_rollout_question_pairs(jsonl, bundle_path)
    assert len(pairs) == 1
    assert pairs[0]["pos_reply"] == "pos reply"
    assert pairs[0]["neg_reply"] == "neg reply"
