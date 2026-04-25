"""Single-agent gate experiment: parse ``toggle_gate`` tool calls (Gemma 3 prompt-style, not HF tools).

Includes token-level generation with per-token hook swapping so steering vectors
change the instant a ``[toggle_gate(...)]`` bracket completes mid-reply.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

# Gemma 3 prompt-based function calling (no native tools= in tokenizer for this model).
# GATE_HELP_V2: marker so build_gate_agent_system does not double-append this block.
GATE_AGENT_TOOL_BLOCK = """GATE_HELP_V2
Gates G1–G4: at most one is ON (turning one ON clears the others). You are not told what they do. Be brief.

To switch a gate, output exactly one bracket line then continue speaking:
[toggle_gate(gate=N, state=true)] or [toggle_gate(gate=N, state=false)]  (N = 1..4)

Example — user says "switch on gate 3", you reply:
[toggle_gate(gate=3, state=true)]
Gate 3 is now on.

Rules:
- One toggle per reply. Only use more if the user explicitly asks for multiple switches.
- Never repeat the same bracket line.
- If your reply is only a bracket call (no other text), the system applies the toggle and tells you the new state so you can continue.

{"name":"toggle_gate","parameters":{"properties":{"gate":{"type":"integer","enum":[1,2,3,4]},"state":{"type":"boolean"}},"required":["gate","state"]}}"""

_TOGGLE_RE = re.compile(
    r"\[\s*toggle_gate\s*\(\s*gate\s*=\s*([1-4])\s*,\s*state\s*=\s*(true|false)\s*\)\s*\]",
    re.IGNORECASE,
)


def build_gate_agent_system(base_system: str) -> str:
    """Prepend grid/default system with the gate tool block and brevity instruction."""
    base = base_system.rstrip()
    if "toggle_gate(gate=" in base and (
        "GATE_HELP_V2" in base
        or "only one gate may be on at a time" in base
        or "You are not told what they do" in base
    ):
        return base
    return f"{base}\n\n{GATE_AGENT_TOOL_BLOCK}"


def parse_toggle_gate_calls(text: str) -> tuple[list[tuple[int, bool]], str]:
    """
    Extract all ``[toggle_gate(gate=N, state=bool)]`` calls and return (calls, remainder_text).
    Remainder has bracket expressions removed (whitespace-trimmed).
    """
    s = text.strip()
    found: list[tuple[int, bool]] = []
    for m in _TOGGLE_RE.finditer(s):
        g = int(m.group(1))
        st = m.group(2).lower() == "true"
        found.append((g, st))
    remainder = _TOGGLE_RE.sub("", s)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    return found, remainder


def normalize_exclusive_gates(g: list[bool]) -> list[bool]:
    """At most one gate on: if multiple True (legacy client), keep the lowest-index True."""
    out = [False, False, False, False]
    for i in range(min(4, len(g))):
        if g[i]:
            out[i] = True
            return out
    return out


def apply_gate_toggles(gates: list[bool], toggles: list[tuple[int, bool]]) -> None:
    """
    Mutates ``gates`` (length 4) in place.

    **Mutually exclusive “on”:** ``state=true`` for gate N clears all gates, then turns N on.
    ``state=false`` only turns that gate off (others unchanged).
    """
    for g, st in toggles:
        if not (1 <= g <= 4):
            continue
        if st:
            for i in range(len(gates)):
                gates[i] = False
            gates[g - 1] = True
        else:
            gates[g - 1] = False


def format_gate_tool_result_message(gates: list[bool]) -> str:
    """User-role line after a tool-style assistant message (simulated tool result)."""
    on = [i + 1 for i, v in enumerate(gates) if v]
    if not on:
        return "[T] off"
    if len(on) == 1:
        return f"[T] G{on[0]}"
    return "[T] " + "+".join(f"G{x}" for x in on)


def run_gate_agent_turn(
    *,
    messages: list[dict[str, str]],
    gates: list[bool],
    model: "PreTrainedModel",
    tokenizer: "AutoTokenizer",
    device: torch.device,
    dtype: torch.dtype,
    layers: Any,
    bundle: dict[str, Any],
    trait_alphas: dict[str, float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    max_tool_rounds: int = 12,
) -> tuple[str, list[bool]]:
    """
    One user turn: run generate in a loop until the model returns a non-tool-only reply
    or max_tool_rounds. Mutates ``messages`` by appending assistant + simulated tool user
    messages for each tool-only step.

    Returns (final_assistant_text, final_gate_state).
    """
    from app.persona.gate_contributions import contributions_for_gates
    from app.persona.grid_nine import (
        generate_plain_from_messages,
        generate_steered_multi_gates_from_messages,
    )

    g = list(gates)
    working = [dict(m) for m in messages]

    for _ in range(max_tool_rounds):
        contribs = contributions_for_gates(g, bundle, device, dtype, trait_alphas)
        if contribs:
            full = generate_steered_multi_gates_from_messages(
                model,
                tokenizer,
                device,
                working,
                layers=layers,
                contributions=contribs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        else:
            full = generate_plain_from_messages(
                model,
                tokenizer,
                device,
                working,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        toggles, remainder = parse_toggle_gate_calls(full)
        if not toggles:
            return (full.strip(), g)

        apply_gate_toggles(g, toggles)
        working.append({"role": "assistant", "content": full.strip()})
        working.append({"role": "user", "content": format_gate_tool_result_message(g)})

        # Tool-only: no natural language this round — continue so the model can speak next.
        if not remainder:
            continue

        # Model mixed NL + toggles in one string: show NL only (non-stream API); streaming shows full raw output.
        return (remainder, g)

    logger.warning("Gate agent hit max_tool_rounds=%s", max_tool_rounds)
    return ("[Stopped: max internal gate rounds reached.]", g)


def run_gate_agent_turn_stream(
    *,
    messages: list[dict[str, str]],
    gates: list[bool],
    model: "PreTrainedModel",
    tokenizer: "AutoTokenizer",
    device: torch.device,
    dtype: torch.dtype,
    layers: Any,
    bundle: dict[str, Any],
    trait_alphas: dict[str, float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    max_tool_rounds: int = 12,
) -> Iterator[dict[str, Any]]:
    """
    Same control flow as ``run_gate_agent_turn``, but streams each model completion token-by-token
    and emits banners / synthetic tool lines so the client can show the full trace.
    Yields dicts with optional keys: ``banner``, ``chunk``, ``round``, ``segment``, ``final``, ``gates``.
    """
    from app.persona.gate_contributions import contributions_for_gates
    from app.persona.grid_nine import (
        generate_plain_from_messages_stream,
        generate_steered_multi_gates_from_messages_stream,
    )

    g = list(gates)
    working = [dict(m) for m in messages]

    for round_idx in range(max_tool_rounds):
        yield {
            "banner": f"[r{round_idx + 1}]",
        }

        contribs = contributions_for_gates(g, bundle, device, dtype, trait_alphas)
        if contribs:
            stream_it = generate_steered_multi_gates_from_messages_stream(
                model,
                tokenizer,
                device,
                working,
                layers=layers,
                contributions=contribs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        else:
            stream_it = generate_plain_from_messages_stream(
                model,
                tokenizer,
                device,
                working,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        full = ""
        for text in stream_it:
            full += text
            yield {
                "chunk": text,
                "round": round_idx,
                "segment": "model",
            }

        toggles, remainder = parse_toggle_gate_calls(full)
        if not toggles:
            yield {"final": True, "gates": [bool(x) for x in g]}
            return

        # Toggle-only round: remainder is empty — user only saw bracket lines above; explain before tool + next round.
        if not remainder:
            yield {
                "chunk": f"\n(brackets only → tool line, then r{round_idx + 2})\n",
                "round": round_idx,
                "segment": "trace_note",
            }

        apply_gate_toggles(g, toggles)
        working.append({"role": "assistant", "content": full.strip()})
        tool_msg = format_gate_tool_result_message(g)
        working.append({"role": "user", "content": tool_msg})

        yield {
            "banner": "[tool]",
        }
        yield {
            "chunk": "\n" + tool_msg + "\n",
            "round": round_idx,
            "segment": "tool_result",
        }

        if not remainder:
            continue

        yield {"final": True, "gates": [bool(x) for x in g]}
        return

    yield {
        "chunk": "\n[stop: max gate rounds]\n",
        "round": max_tool_rounds - 1,
        "segment": "system",
    }
    yield {"final": True, "gates": [bool(x) for x in g]}


# ---------------------------------------------------------------------------
# Hook helpers for token-level generation
# ---------------------------------------------------------------------------


def _install_hooks(
    gates: list[bool],
    layers: nn.ModuleList,
    bundle: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    trait_alphas: dict[str, float],
    hook_calls: list[int],
) -> list[Any]:
    """Register forward hooks for all active gates. Returns list of handles."""
    from app.persona.gate_contributions import contributions_for_gates
    from app.persona.steering_demo import _steering_hook_fn

    contribs = contributions_for_gates(gates, bundle, device, dtype, trait_alphas)
    combined: dict[int, torch.Tensor] = {}
    for layer_idx, direction, alpha in contribs:
        if abs(float(alpha)) < 1e-12:
            continue
        acc = float(alpha) * direction
        if layer_idx in combined:
            combined[layer_idx] = combined[layer_idx] + acc
        else:
            combined[layer_idx] = acc
    handles: list[Any] = []
    for layer_idx, vec in combined.items():
        if vec.abs().max().item() <= 0:
            continue
        hook = _steering_hook_fn(1.0, vec, steer_last_token_only=False, hook_calls=hook_calls)
        handles.append(layers[layer_idx].register_forward_hook(hook))
    return handles


def _remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        h.remove()
    handles.clear()


# ---------------------------------------------------------------------------
# Token-level generation with mid-reply hook swapping
# ---------------------------------------------------------------------------


def _tokenize_prompt(
    messages: list[dict[str, str]],
    tokenizer: "AutoTokenizer",
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply chat template and return (input_ids, attention_mask) on device."""
    raw = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = raw.to(device) if isinstance(raw, torch.Tensor) else raw["input_ids"].to(device)
    attn = torch.ones_like(ids, dtype=torch.long, device=device)
    return ids, attn


def _sample_next(logits: torch.Tensor, do_sample: bool, temperature: float) -> torch.Tensor:
    """Pick the next token id from logits[:, -1, :]."""
    last = logits[:, -1, :]
    if not do_sample:
        return last.argmax(dim=-1, keepdim=True)
    scaled = last / max(temperature, 1e-8)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _format_gate_label(gates: list[bool]) -> str:
    return " ".join(f"G{i+1}{'●' if on else '○'}" for i, on in enumerate(gates))


def _format_gate_compact(gates: list[bool]) -> str:
    """No spaces — shorter for SSE banners."""
    return "".join(f"G{i+1}{'●' if on else '○'}" for i, on in enumerate(gates))


def run_gate_agent_turn_token_level(
    *,
    messages: list[dict[str, str]],
    gates: list[bool],
    model: "PreTrainedModel",
    tokenizer: "AutoTokenizer",
    device: torch.device,
    dtype: torch.dtype,
    layers: Any,
    bundle: dict[str, Any],
    trait_alphas: dict[str, float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Iterator[dict[str, Any]]:
    """
    Token-level autoregressive loop. Hooks are swapped the instant a complete
    ``[toggle_gate(...)]`` bracket is detected in the decoded output.

    Yields the same event dict shape as ``run_gate_agent_turn_stream`` for
    compatibility with ``_stream_events_gate_chat``:
    ``chunk``, ``banner``, ``gate_update``, ``final``, ``gates``.
    """
    g = list(gates)
    input_ids, attn = _tokenize_prompt(messages, tokenizer, device)

    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    eos_model = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_model, int):
        eos_ids.add(eos_model)
    elif isinstance(eos_model, list):
        eos_ids.update(eos_model)

    hook_calls: list[int] = [0]
    handles = _install_hooks(g, layers, bundle, device, dtype, trait_alphas, hook_calls)

    yield {"banner": f"→{_format_gate_compact(g)}"}

    past = None
    all_new_ids: list[int] = []
    decoded_so_far = ""
    last_toggle_scan_pos = 0

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                out = model(input_ids, attention_mask=attn, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = _sample_next(out.logits, do_sample, temperature)
                tid = int(next_id.item())

                if tid in eos_ids:
                    break

                all_new_ids.append(tid)
                decoded_so_far = tokenizer.decode(all_new_ids, skip_special_tokens=True)

                new_text = decoded_so_far[len(decoded_so_far) - len(tokenizer.decode([tid], skip_special_tokens=True)):]
                if new_text:
                    yield {"chunk": new_text, "segment": "model"}

                if "]" in decoded_so_far[last_toggle_scan_pos:]:
                    toggles, _ = parse_toggle_gate_calls(decoded_so_far[last_toggle_scan_pos:])
                    if toggles:
                        apply_gate_toggles(g, toggles)
                        _remove_hooks(handles)
                        handles = _install_hooks(g, layers, bundle, device, dtype, trait_alphas, hook_calls)
                        yield {
                            "gate_update": [bool(x) for x in g],
                            "banner": f"Δ{_format_gate_compact(g)}",
                        }
                        last_toggle_scan_pos = len(decoded_so_far)

                input_ids = next_id
                attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)

    finally:
        _remove_hooks(handles)

    yield {"final": True, "gates": [bool(x) for x in g]}


# ---------------------------------------------------------------------------
# Rotating-vector test: cycle one gate per token (for diagnostics)
# ---------------------------------------------------------------------------


def run_rotating_gate_test(
    *,
    messages: list[dict[str, str]],
    model: "PreTrainedModel",
    tokenizer: "AutoTokenizer",
    device: torch.device,
    dtype: torch.dtype,
    layers: Any,
    bundle: dict[str, Any],
    trait_alphas: dict[str, float],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Iterator[dict[str, Any]]:
    """
    Diagnostic: generate a reply where exactly **one** gate is active per token,
    cycling G1→G2→G3→G4→G1… Every token gets a different steering vector.

    Yields ``chunk``, ``gate_for_token`` (which gate was on for that token),
    and at the end ``token_gate_log`` (full list of per-token gate indices).
    """
    gate_order = [0, 1, 2, 3]
    input_ids, attn = _tokenize_prompt(messages, tokenizer, device)

    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    eos_model = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_model, int):
        eos_ids.add(eos_model)
    elif isinstance(eos_model, list):
        eos_ids.update(eos_model)

    hook_calls: list[int] = [0]
    token_gate_log: list[dict[str, Any]] = []

    past = None
    all_new_ids: list[int] = []

    yield {"banner": "rot G1→G2→G3→G4 / tok"}

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                gate_idx = gate_order[step % len(gate_order)]
                g = [False, False, False, False]
                g[gate_idx] = True

                handles = _install_hooks(g, layers, bundle, device, dtype, trait_alphas, hook_calls)

                out = model(input_ids, attention_mask=attn, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = _sample_next(out.logits, do_sample, temperature)
                tid = int(next_id.item())

                _remove_hooks(handles)

                if tid in eos_ids:
                    break

                all_new_ids.append(tid)
                tok_text = tokenizer.decode([tid], skip_special_tokens=True)
                token_gate_log.append({
                    "step": step,
                    "token_id": tid,
                    "token": tok_text,
                    "gate_on": gate_idx + 1,
                })
                yield {
                    "chunk": tok_text,
                    "gate_for_token": gate_idx + 1,
                    "segment": "model",
                }

                input_ids = next_id
                attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)

    finally:
        pass

    full_text = tokenizer.decode(all_new_ids, skip_special_tokens=True)
    yield {
        "final": True,
        "reply": full_text.strip(),
        "token_gate_log": token_gate_log,
        "gates": [False, False, False, False],
    }

