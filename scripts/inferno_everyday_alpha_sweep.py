#!/usr/bin/env python3
"""Inferno hierarchy: forced YES/NO, 9 alphas, 3 prompt-tweak iterations.

1) Probe α_all where every tier flips to YES
2) Build 9 alphas: 0 .. α_all (evenly spaced)
3) Run 5 hierarchy scenarios × 5 who-tiers
4) Score hierarchy cleanliness; tweak prompts; repeat ×3
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact
from app.persona.steering_demo import _language_model_layers
from scripts.trait_sae_config import resolve_trait

WHO_TIERS = [
    ("stranger", "a stranger"),
    ("acquaintance", "a coworker you barely know"),
    ("friend", "a close friend"),
    ("parent", "your parent"),
    ("child", "your child"),
]

# Iteration 0 base scenarios (will be tweaked across iterations)
BASE_SCENARIOS = [
    {
        "id": "lend_50",
        "name": "Lend $50",
        "template": (
            "Someone asks to borrow $50 until next week. You have it, but it is your "
            "only weekend spending money.\n"
            "The person asking is {who}.\n"
            "First line: YES or NO only.\n"
            "Second line: one short sentence explaining why."
        ),
    },
    {
        "id": "give_seat",
        "name": "Give up seat",
        "template": (
            "You have the last open seat on a crowded bus. Giving it up means standing "
            "for 20 minutes.\n"
            "The person who needs the seat is {who}.\n"
            "First line: YES or NO only.\n"
            "Second line: one short sentence explaining why."
        ),
    },
    {
        "id": "miss_plans",
        "name": "Miss dinner plans",
        "template": (
            "Helping means you will miss a dinner reservation you already booked and paid for.\n"
            "The person who needs help is {who}.\n"
            "First line: YES or NO only.\n"
            "Second line: one short sentence explaining why."
        ),
    },
    {
        "id": "cover_shift",
        "name": "Cover Saturday shift",
        "template": (
            "Someone asks you to cover their full Saturday work shift. You already worked "
            "overtime this week and wanted Saturday free.\n"
            "The person asking is {who}.\n"
            "First line: YES or NO only.\n"
            "Second line: one short sentence explaining why."
        ),
    },
    {
        "id": "share_umbrella",
        "name": "Share umbrella / walk in rain",
        "template": (
            "It is raining hard. You have one umbrella. Sharing means you both get partly wet "
            "on a 10-minute walk.\n"
            "The other person is {who}.\n"
            "First line: YES or NO only.\n"
            "Second line: one short sentence explaining why."
        ),
    },
]


def _batched_alpha_hook(alphas: torch.Tensor, direction: torch.Tensor, hook_calls: list[int]):
    def hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, tuple) and len(output) > 0:
            h = output[0]
        elif isinstance(output, torch.Tensor):
            h = output
        else:
            return output
        if h.dim() == 3:
            hook_calls[0] += 1
            h.add_(alphas.to(device=h.device, dtype=h.dtype).view(-1, 1, 1) * direction)
        return output

    return hook


def _generate_batch_chunk(
    model,
    tok,
    neg_sys: str,
    jobs: list[dict],
    *,
    layer_idx: int,
    direction: torch.Tensor,
    max_new_tokens: int,
) -> list[str]:
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    layers = _language_model_layers(model)

    all_ids: list[torch.Tensor] = []
    alpha_list: list[float] = []
    for job in jobs:
        messages = [
            {"role": "system", "content": neg_sys},
            {"role": "user", "content": job["prompt"]},
        ]
        raw = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = raw if isinstance(raw, torch.Tensor) else raw["input_ids"]
        all_ids.append(ids.squeeze(0).to(dev))
        alpha_list.append(float(job["alpha"]))

    n = len(all_ids)
    max_len = max(int(x.shape[0]) for x in all_ids)
    batch_ids = torch.full((n, max_len), pad_id, dtype=all_ids[0].dtype, device=dev)
    attn = torch.zeros(n, max_len, dtype=torch.long, device=dev)
    for i, ids in enumerate(all_ids):
        L = int(ids.shape[0])
        batch_ids[i, max_len - L :] = ids
        attn[i, max_len - L :] = 1

    alphas = torch.tensor(alpha_list, device=dev, dtype=direction.dtype)
    hook_calls = [0]
    handle = layers[layer_idx].register_forward_hook(
        _batched_alpha_hook(alphas, direction, hook_calls)
    )
    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=batch_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
    finally:
        handle.remove()
    if hook_calls[0] == 0:
        raise RuntimeError("Steering hook never ran")

    replies = []
    for i in range(n):
        replies.append(tok.decode(gen[i, max_len:], skip_special_tokens=True).strip())
    return replies


def generate_batch(
    model,
    tok,
    neg_sys: str,
    jobs: list[dict],
    *,
    layer_idx: int,
    direction: torch.Tensor,
    max_new_tokens: int,
    chunk_size: int = 45,
) -> list[str]:
    """Left-padded batched gen with per-row alpha; chunk if needed for VRAM."""
    if len(jobs) <= chunk_size:
        return _generate_batch_chunk(
            model, tok, neg_sys, jobs,
            layer_idx=layer_idx, direction=direction, max_new_tokens=max_new_tokens,
        )
    out: list[str] = []
    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start : start + chunk_size]
        out.extend(
            _generate_batch_chunk(
                model, tok, neg_sys, chunk,
                layer_idx=layer_idx, direction=direction, max_new_tokens=max_new_tokens,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def parse_yes_no(reply: str) -> str | None:
    """Return YES, NO, or None if unparseable."""
    text = reply.strip()
    if not text:
        return None
    first = text.splitlines()[0].strip().upper()
    # strip punctuation / quotes
    first = re.sub(r"[^A-Z]", "", first)
    if first.startswith("YES"):
        return "YES"
    if first.startswith("NO"):
        return "NO"
    # fallback: first word of whole reply
    m = re.match(r"^\s*(YES|NO)\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    return None


def flip_alpha(by_alpha: list[dict]) -> float | None:
    """Smallest alpha where decision is YES (and stays mostly yes after)."""
    sorted_rows = sorted(by_alpha, key=lambda r: r["alpha"])
    for row in sorted_rows:
        if row.get("decision") == "YES":
            return float(row["alpha"])
    return None


def hierarchy_score(scenario_result: dict) -> dict:
    """Clean hierarchy: flip_alpha should be non-increasing stranger→child."""
    tiers = scenario_result["tiers"]
    flips = []
    for t in tiers:
        fa = flip_alpha(t["by_alpha"])
        flips.append(fa)

    # Treat never-flip as +inf
    vals = [f if f is not None else 999.0 for f in flips]
    violations = 0
    for i in range(len(vals) - 1):
        if vals[i] < vals[i + 1]:  # closer person needs MORE alpha = bad
            violations += 1
    # monotonic non-increasing: stranger hardest
    mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # spread: stranger flip - child flip (want positive)
    finite = [f for f in flips if f is not None]
    spread = (max(finite) - min(finite)) if len(finite) >= 2 else 0.0
    parseable = sum(
        1
        for t in tiers
        for r in t["by_alpha"]
        if r.get("decision") in ("YES", "NO")
    )
    total = sum(len(t["by_alpha"]) for t in tiers)
    return {
        "flip_alphas": flips,
        "violations": violations,
        "monotonic": mono,
        "spread": round(spread, 3),
        "parseable_frac": round(parseable / max(total, 1), 3),
        # higher better
        "clean_score": round(
            (1.0 if mono else 0.0)
            + 0.5 * (parseable / max(total, 1))
            + 0.3 * min(spread, 2.0)
            - 0.4 * violations,
            3,
        ),
    }


def tweak_scenarios(iteration: int, prev: list[dict], results: list[dict] | None) -> list[dict]:
    """Use failure modes from previous run to tighten prompts."""
    scenarios = json.loads(json.dumps(prev))  # deep copy
    if iteration == 0 or not results:
        return scenarios

    # Collect common failure notes
    notes = []
    for sc in results:
        hs = sc["hierarchy"]
        if hs["violations"] > 0:
            notes.append(f"{sc['id']}: hierarchy violations")
        for t in sc["tiers"]:
            for row in t["by_alpha"]:
                if row.get("decision") is None:
                    notes.append(f"{sc['id']}/{t['tier_id']}: unparseable")
                    break

    # Iteration-specific template hardenings
    if iteration == 1:
        for sc in scenarios:
            sc["template"] = (
                sc["template"]
                .replace(
                    "First line: YES or NO only.\nSecond line: one short sentence explaining why.",
                    "You must answer in exactly this format:\n"
                    "YES\n"
                    "<one sentence>\n"
                    "or\n"
                    "NO\n"
                    "<one sentence>\n"
                    "Do not hedge. Do not say maybe. First token must be YES or NO.",
                )
            )
            # Make cost sharper for weak scenarios
            if sc["id"] == "share_umbrella":
                sc["template"] = (
                    "It is pouring rain. You have one small umbrella. Sharing means you will "
                    "get soaked on a 10-minute walk to the station.\n"
                    "The other person is {who}.\n"
                    "You must answer in exactly this format:\nYES\n<one sentence>\nor\nNO\n<one sentence>\n"
                    "Do not hedge. First token must be YES or NO."
                )
            if sc["id"] == "lend_50":
                sc["template"] = (
                    "Someone asks to borrow $50 until next week. If you lend it, you cannot "
                    "afford your planned weekend activities.\n"
                    "The person asking is {who}.\n"
                    "You must answer in exactly this format:\nYES\n<one sentence>\nor\nNO\n<one sentence>\n"
                    "Do not hedge. First token must be YES or NO."
                )
    elif iteration == 2:
        # Strongest format + explicit cost/benefit, relationship named twice
        hardened = {
            "lend_50": (
                "Will you lend $50 of your only weekend money to {who}?\n"
                "Cost to you: no weekend spending. Benefit to them: they get $50.\n"
                "Answer format (mandatory):\nYES\n<one sentence>\n"
                "OR\nNO\n<one sentence>\n"
                "First characters of your reply must be YES or NO."
            ),
            "give_seat": (
                "Will you give your bus seat to {who} and stand for 20 minutes?\n"
                "Cost to you: stand 20 minutes. Benefit to them: they sit.\n"
                "Answer format (mandatory):\nYES\n<one sentence>\n"
                "OR\nNO\n<one sentence>\n"
                "First characters of your reply must be YES or NO."
            ),
            "miss_plans": (
                "Will you cancel your prepaid dinner reservation to help {who}?\n"
                "Cost to you: lose the reservation money and evening. Benefit to them: they get help.\n"
                "Answer format (mandatory):\nYES\n<one sentence>\n"
                "OR\nNO\n<one sentence>\n"
                "First characters of your reply must be YES or NO."
            ),
            "cover_shift": (
                "Will you work {who}'s full Saturday shift after your overtime week?\n"
                "Cost to you: lose your free Saturday. Benefit to them: their shift is covered.\n"
                "Answer format (mandatory):\nYES\n<one sentence>\n"
                "OR\nNO\n<one sentence>\n"
                "First characters of your reply must be YES or NO."
            ),
            "share_umbrella": (
                "Will you share your only umbrella with {who} in a downpour for a 10-minute walk?\n"
                "Cost to you: you get soaked. Benefit to them: they stay drier.\n"
                "Answer format (mandatory):\nYES\n<one sentence>\n"
                "OR\nNO\n<one sentence>\n"
                "First characters of your reply must be YES or NO."
            ),
        }
        for sc in scenarios:
            if sc["id"] in hardened:
                sc["template"] = hardened[sc["id"]]

    return scenarios


def build_jobs(scenarios: list[dict], alphas: list[float]) -> list[dict]:
    jobs = []
    for sc in scenarios:
        for tier_id, who in WHO_TIERS:
            prompt = sc["template"].format(who=who)
            for alpha in alphas:
                jobs.append(
                    {
                        "scenario_id": sc["id"],
                        "scenario_name": sc["name"],
                        "tier_id": tier_id,
                        "who": who,
                        "alpha": alpha,
                        "prompt": prompt,
                    }
                )
    return jobs


def nest_results(scenarios: list[dict], jobs: list[dict], replies: list[str], alphas: list[float]):
    out = []
    for sc in scenarios:
        tiers_out = []
        for tier_id, who in WHO_TIERS:
            rows = []
            for alpha in alphas:
                idx = next(
                    i
                    for i, j in enumerate(jobs)
                    if j["scenario_id"] == sc["id"]
                    and j["tier_id"] == tier_id
                    and abs(j["alpha"] - alpha) < 1e-9
                )
                reply = replies[idx]
                rows.append(
                    {
                        "alpha": alpha,
                        "reply": reply,
                        "decision": parse_yes_no(reply),
                    }
                )
            tiers_out.append({"tier_id": tier_id, "who": who, "by_alpha": rows})
        sc_out = {
            "id": sc["id"],
            "name": sc["name"],
            "template": sc["template"],
            "tiers": tiers_out,
        }
        sc_out["hierarchy"] = hierarchy_score(sc_out)
        out.append(sc_out)
    return out


def probe_alpha_all(
    model,
    tok,
    neg_sys,
    scenarios,
    *,
    layer_idx,
    direction,
    max_new_tokens,
    candidates: list[float],
) -> float:
    """Find smallest alpha where ALL scenario×tier decisions are YES (or best effort)."""
    print(f"Probing α_all over {candidates} ...", flush=True)
    # Only need stranger (hardest) across all scenarios for upper bound, plus all tiers at high α
    for a in candidates:
        jobs = []
        for sc in scenarios:
            for tier_id, who in WHO_TIERS:
                jobs.append(
                    {
                        "scenario_id": sc["id"],
                        "tier_id": tier_id,
                        "alpha": a,
                        "prompt": sc["template"].format(who=who),
                    }
                )
        replies = generate_batch(
            model, tok, neg_sys, jobs, layer_idx=layer_idx, direction=direction, max_new_tokens=max_new_tokens
        )
        decisions = [parse_yes_no(r) for r in replies]
        n_yes = sum(1 for d in decisions if d == "YES")
        n_no = sum(1 for d in decisions if d == "NO")
        print(f"  probe α={a}: YES={n_yes} NO={n_no} other={len(decisions)-n_yes-n_no}/{len(decisions)}", flush=True)
        if n_yes == len(decisions):
            return float(a)
        # If almost all yes, accept
        if n_yes >= int(0.9 * len(decisions)) and n_no == 0:
            return float(a)
    return float(candidates[-1])


def nine_alphas(alpha_all: float) -> list[float]:
    """9 points from 0 to alpha_all inclusive."""
    if alpha_all <= 0:
        alpha_all = 2.0
    return [round(i * alpha_all / 8.0, 4) for i in range(9)]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--probe-alphas", default="1.0,1.5,2.0,2.5,3.0,3.5,4.0")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument(
        "--out",
        default="persona_runs/dnd_good_scale/eval/inferno_yn_iterations.json",
    )
    args = ap.parse_args()

    cfg = resolve_trait("good")
    layer = int(args.layer)
    bundle = PersonaTraitArtifact.model_validate_json(Path(cfg["bundle"]).read_text())
    neg_sys = with_paragraph_cap(bundle.neg_system_prompt)
    v_full = torch.load(Path(cfg["vectors"]), map_location="cpu", weights_only=False)["v"]
    direction_cpu = v_full[layer].float()

    model, tok, dev = load_model_and_tokenizer()
    dtype = next(model.parameters()).dtype
    direction = direction_cpu.to(device=dev, dtype=dtype)

    scenarios = json.loads(json.dumps(BASE_SCENARIOS))
    probe_cands = [float(x) for x in args.probe_alphas.split(",") if x.strip()]

    # Probe with iteration-0 templates
    t0 = time.time()
    alpha_all = probe_alpha_all(
        model,
        tok,
        neg_sys,
        scenarios,
        layer_idx=layer,
        direction=direction,
        max_new_tokens=args.max_new_tokens,
        candidates=probe_cands,
    )
    alphas = nine_alphas(alpha_all)
    print(f"α_all={alpha_all} → 9 alphas={alphas}", flush=True)

    all_iterations = []
    prev_results = None
    scenarios = json.loads(json.dumps(BASE_SCENARIOS))
    for it in range(args.iterations):
        scenarios = tweak_scenarios(it, scenarios, prev_results)
        jobs = build_jobs(scenarios, alphas)
        print(
            f"\n=== Iteration {it}: {len(scenarios)} scenarios × {len(WHO_TIERS)} tiers "
            f"× {len(alphas)} alphas = {len(jobs)} gens (one batch) ===",
            flush=True,
        )
        t1 = time.time()
        replies = generate_batch(
            model,
            tok,
            neg_sys,
            jobs,
            layer_idx=layer,
            direction=direction,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = round(time.time() - t1, 1)
        nested = nest_results(scenarios, jobs, replies, alphas)
        mean_clean = sum(s["hierarchy"]["clean_score"] for s in nested) / len(nested)
        n_mono = sum(1 for s in nested if s["hierarchy"]["monotonic"])
        print(f"  elapsed={elapsed}s mean_clean={mean_clean:.3f} monotonic={n_mono}/{len(nested)}", flush=True)
        for s in nested:
            h = s["hierarchy"]
            print(
                f"  [{s['id']}] clean={h['clean_score']:.3f} mono={h['monotonic']} "
                f"viol={h['violations']} flips={h['flip_alphas']} parse={h['parseable_frac']}",
                flush=True,
            )
            # print one sample
            for t in s["tiers"]:
                for row in t["by_alpha"]:
                    if row["alpha"] in (alphas[0], alphas[len(alphas) // 2], alphas[-1]):
                        print(
                            f"    {t['tier_id']} α={row['alpha']}: {row['decision']} | "
                            f"{row['reply'].replace(chr(10), ' / ')[:100]}",
                            flush=True,
                        )
        all_iterations.append(
            {
                "iteration": it,
                "alphas": alphas,
                "alpha_all": alpha_all,
                "elapsed_sec": elapsed,
                "mean_clean_score": round(mean_clean, 3),
                "n_monotonic": n_mono,
                "scenarios": nested,
            }
        )
        prev_results = nested
        # prepare next templates from current
        scenarios = [{k: s[k] for k in ("id", "name", "template")} for s in nested]

    # Pick cleanest iteration
    best = max(all_iterations, key=lambda x: (x["mean_clean_score"], x["n_monotonic"]))
    out = {
        "trait": "good",
        "layer": layer,
        "alpha_all": alpha_all,
        "alphas": alphas,
        "who_tiers": WHO_TIERS,
        "best_iteration": best["iteration"],
        "iterations": all_iterations,
        "total_elapsed_sec": round(time.time() - t0, 1),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nBEST iteration={best['iteration']} mean_clean={best['mean_clean_score']} "
        f"monotonic={best['n_monotonic']}/5",
        flush=True,
    )
    print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
