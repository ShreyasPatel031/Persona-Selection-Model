"""
Self-dialog experiment: one model plans user turns and gate settings; the assistant reply is
steered by up to four anonymous binary gates mapped to D&D persona vectors (order fixed in
code but never revealed to the model).

Run on the same machine as Gemma (e.g. VM with HF_TOKEN):

  HF_TOKEN=… python -m app.persona.gate_self_chat \\
    --sessions 10 --turns 2 --out persona_runs/gate_self_chat_last.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from app.persona.dnd_playground import load_playground_bundle
from app.persona.gate_contributions import GATE_TRAIT_ORDER, contributions_for_gates
from app.persona.lm_layers import language_model_layers

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """You are participating in a closed-loop experiment with yourself.

You control four independent binary switches called Gate 1, Gate 2, Gate 3, and Gate 4. Each gate is either ON (true) or OFF (false). You are not told what they do internally.

Protocol:
- Each turn you output one JSON object only (no markdown fences, no extra prose).
- You choose the four gate states and a short user_message addressed to an assistant.
- The assistant reply is produced while your chosen gates are applied; you only see the resulting assistant text, not any internal parameters.
- Across turns you may see the running transcript. Infer what each gate tends to do by varying one gate at a time when possible.

Your JSON schema:
{"gates": [bool, bool, bool, bool], "user_message": "<string>", "reasoning": "<brief what you are testing>"}"""


def _load_bundle_with_vectors(
    config_path: Path | None,
    grid_path: Path | None,
) -> dict[str, Any]:
    b = load_playground_bundle(config_path, grid_path)
    v_cpu: dict[str, torch.Tensor] = {}
    for name, spec in b["traits_cfg"].items():
        p = Path(spec["vectors"]).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"DND vectors missing for {name}: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        v_cpu[name] = ck["v"].float()
    b["v_cpu"] = v_cpu
    return b


def _parse_planner_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
        if m:
            t = m.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in planner output: {text[:500]!r}")
    return json.loads(t[start : end + 1])


def _normalize_gates(raw: Any) -> list[bool]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("gates must be a list of 4 booleans")
    out: list[bool] = []
    for x in raw:
        if isinstance(x, bool):
            out.append(x)
        elif x in (0, 1):
            out.append(bool(x))
        else:
            raise ValueError(f"invalid gate value: {x!r}")
    return out


def _format_transcript_for_planner(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for t in turns:
        g = t.get("gates")
        lines.append(
            f"— Gates [G1={'on' if g[0] else 'off'}, G2={'on' if g[1] else 'off'}, "
            f"G3={'on' if g[2] else 'off'}, G4={'on' if g[3] else 'off'}]"
        )
        lines.append(f"User: {t.get('user_message', '')}")
        lines.append(f"Assistant: {t.get('assistant_reply', '')}")
        if t.get("reasoning"):
            lines.append(f"(Your prior note: {t['reasoning']})")
        lines.append("")
    return "\n".join(lines).strip() or "(no turns yet)"


def run_session(
    *,
    model: "PreTrainedModel",
    tokenizer: "AutoTokenizer",
    device: torch.device,
    bundle: dict[str, Any],
    n_turns: int,
    max_new_tokens: int,
    max_planner_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """One self-chat session: planner (unsteered) → steered assistant, repeated."""
    from app.persona.grid_nine import (
        generate_plain_from_messages,
        generate_steered_multi_gates_from_messages,
    )

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    layers = language_model_layers(model)
    dtype = next(model.parameters()).dtype
    assistant_system = bundle["system"]

    turns: list[dict[str, Any]] = []
    parse_failures = 0

    for turn_idx in range(n_turns):
        transcript = _format_transcript_for_planner(turns)
        user_prompt = (
            f"Transcript so far:\n{transcript}\n\n"
            f"This is turn {turn_idx + 1} of {n_turns}. "
            "Output the JSON object for your next experimental step."
        )
        planner_messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        raw_plan = generate_plain_from_messages(
            model,
            tokenizer,
            device,
            planner_messages,
            max_new_tokens=max_planner_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        gates = [False, False, False, False]
        user_message = "Briefly describe a moral dilemma involving authority and harm."
        reasoning = ""
        try:
            plan = _parse_planner_json(raw_plan)
            gates = _normalize_gates(plan.get("gates"))
            user_message = str(plan.get("user_message", user_message)).strip() or user_message
            reasoning = str(plan.get("reasoning", "")).strip()
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Planner JSON parse failed: %s — using fallback", e)
            parse_failures += 1

        conv = [{"role": "system", "content": assistant_system}]
        for prev in turns:
            conv.append({"role": "user", "content": prev["user_message"]})
            conv.append({"role": "assistant", "content": prev["assistant_reply"]})
        conv.append({"role": "user", "content": user_message})

        trait_alphas = dict(bundle.get("trait_alphas") or {})
        contribs = contributions_for_gates(gates, bundle, device, dtype, trait_alphas)
        if contribs:
            assistant_reply = generate_steered_multi_gates_from_messages(
                model,
                tokenizer,
                device,
                conv,
                layers=layers,
                contributions=contribs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        else:
            assistant_reply = generate_plain_from_messages(
                model,
                tokenizer,
                device,
                conv,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        turns.append(
            {
                "turn_index": turn_idx,
                "gates": gates,
                "raw_planner_output": raw_plan,
                "user_message": user_message,
                "assistant_reply": assistant_reply,
                "reasoning": reasoning,
            }
        )

    return {
        "n_turns": n_turns,
        "trait_alphas": dict(bundle.get("trait_alphas") or {}),
        "parse_failures": parse_failures,
        "turns": turns,
        "_ground_truth_gate_order": list(GATE_TRAIT_ORDER),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Self-chat gate discovery experiment (Gemma + persona vectors).")
    p.add_argument("--config", type=Path, default=None, help="dnd_config.json path")
    p.add_argument("--grid", type=Path, default=None, help="dnd_grid_results.json path")
    p.add_argument("--sessions", type=int, default=10, help="Number of independent conversation runs")
    p.add_argument(
        "--turns",
        type=int,
        default=3,
        help="User/assistant pairs per session (multi-message self-chat within each session)",
    )
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "256")))
    p.add_argument("--max-planner-tokens", type=int, default=384)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="Write full JSON transcript here")
    p.add_argument("--model", type=str, default=None, help="Override GEMMA_MODEL_ID")
    args = p.parse_args()

    from app.persona.activations import load_model_and_tokenizer

    bundle = _load_bundle_with_vectors(args.config, args.grid)
    model, tokenizer, device = load_model_and_tokenizer(args.model, device=None)

    sessions_out: list[dict[str, Any]] = []
    base_seed = args.seed

    for si in range(args.sessions):
        sid = base_seed + si if base_seed is not None else None
        logger.info("Session %s/%s …", si + 1, args.sessions)
        sess = run_session(
            model=model,
            tokenizer=tokenizer,
            device=device,
            bundle=bundle,
            n_turns=args.turns,
            max_new_tokens=args.max_new_tokens,
            max_planner_tokens=args.max_planner_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            seed=sid,
        )
        sess["session_index"] = si
        sessions_out.append(sess)

    payload: dict[str, Any] = {
        "config_path": bundle["config_path"],
        "grid_path": bundle["grid_path"],
        "sessions": sessions_out,
        "meta": {
            "sessions": args.sessions,
            "turns_per_session": args.turns,
            "trait_alphas": bundle.get("trait_alphas"),
            "note": "Ground truth gate order is only in each session's _ground_truth_gate_order for analysis.",
        },
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.out)

    # Human-readable stdout
    for sess in sessions_out:
        idx = sess["session_index"]
        print(f"\n{'=' * 72}\n### Session {idx + 1} / {args.sessions}\n{'=' * 72}\n")
        for t in sess["turns"]:
            g = t["gates"]
            print(
                f"--- Turn {t['turn_index'] + 1} — "
                f"G1={'ON' if g[0] else 'off'} "
                f"G2={'ON' if g[1] else 'off'} "
                f"G3={'ON' if g[2] else 'off'} "
                f"G4={'ON' if g[3] else 'off'} ---"
            )
            if t.get("reasoning"):
                print(f"Planner note: {t['reasoning']}\n")
            print(f"User:\n{t['user_message']}\n")
            print(f"Assistant:\n{t['assistant_reply']}\n")

    total_turns = args.sessions * args.turns
    print(
        f"\nDone. {args.sessions} sessions × {args.turns} turns = {total_turns} user/assistant exchanges.\n"
    )


if __name__ == "__main__":
    main()
