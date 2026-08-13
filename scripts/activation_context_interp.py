#!/usr/bin/env python3
"""
Interpret SAE features using ACTIVATION CONTEXT from alpha sweep replies.

Instead of logit lens (W_dec -> lm_head), this:
1. Takes the 45 stored replies from the alpha sweep
2. Runs each through model -> layer-16 hidden states -> SAE encode per token
3. For each tracked feature, collects top-activating tokens WITH surrounding context
4. Sends those to Gemini for real semantic interpretation

This is the correct interpretability method for mid-layer features where
logit lens gives garbage (<unused>, unicode artifacts, etc).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SWEEP = REPO / "persona_runs/dnd_good_scale/sae/alpha_sweep_analysis.json"
DEFAULT_OUT = REPO / "app/static/activation_context_interp.json"
DEFAULT_INTERP = REPO / "app/static/feature_interpretations.json"
DEFAULT_MODEL = os.environ.get("SAE_AUTOINTERP_MODEL", "gemini-2.5-flash")

LAYER = 16
SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
SAE_ID = "layer_16_width_16k_l0_small"
CONTEXT_WINDOW = 4  # tokens of context around each activating token


def get_context_snippet(token_strs: list[str], idx: int, window: int = CONTEXT_WINDOW) -> str:
    start = max(0, idx - window)
    end = min(len(token_strs), idx + window + 1)
    parts = []
    for i in range(start, end):
        tok = token_strs[i].replace("\n", "\\n")
        if i == idx:
            parts.append(f">>>{tok}<<<")
        else:
            parts.append(tok)
    return "".join(parts)


def collect_top_activations_for_feature(
    z: torch.Tensor,
    token_strs: list[str],
    feature_id: int,
    alpha: float,
    question_idx: int,
    *,
    top_n: int = 5,
) -> list[dict]:
    col = z[:, feature_id]
    k = min(top_n, col.numel())
    if k == 0:
        return []
    vals, idx = torch.topk(col.abs(), k=k)
    results = []
    for j in range(k):
        ti = int(idx[j].item())
        act = float(col[ti].item())
        if abs(act) < 1.0:
            continue
        results.append({
            "token": token_strs[ti].strip(),
            "activation": round(act, 2),
            "context": get_context_snippet(token_strs, ti),
            "alpha": alpha,
            "question_idx": question_idx,
            "token_position": ti,
        })
    return results


def build_activation_prompt(feature_id: int, examples: list[dict], alpha_pattern: dict) -> str:
    lines = [
        "You are interpreting a sparse autoencoder (SAE) feature from Gemma-2-4B-IT, layer 16.",
        f"Feature index: F{feature_id}.",
        "",
        "This feature was extracted during a 'Good' persona steering experiment.",
        "Below are the TOP ACTIVATING TOKENS from actual model replies, with surrounding context.",
        ">>>token<<< marks the token where the feature fires strongest.",
        "",
    ]

    by_alpha = {}
    for ex in examples:
        a = ex["alpha"]
        by_alpha.setdefault(a, []).append(ex)

    for alpha in sorted(by_alpha.keys()):
        lines.append(f"--- Alpha = {alpha} (steering strength) ---")
        for ex in by_alpha[alpha][:4]:
            lines.append(f"  activation={ex['activation']:.1f}  context: {ex['context']}")
        lines.append("")

    mean_by_alpha = alpha_pattern.get("mean_by_alpha", {})
    if mean_by_alpha:
        lines.append("Feature's mean activation across steering strengths:")
        for a in sorted(mean_by_alpha.keys(), key=float):
            lines.append(f"  α={a}: {mean_by_alpha[a]:.1f}")
        lines.append("")

    lines.extend([
        "In one concise sentence (under 15 words), describe what concept, behavior,",
        "or linguistic pattern this feature detects in the model's output.",
        "Focus on WHAT the feature responds to semantically, not the token identity.",
        "Use plain English, no markdown, no quotes.",
        "Reply with ONLY the one-sentence interpretation.",
    ])
    return "\n".join(lines)


def interpret_via_gemini(prompt: str, project_id: str, model_name: str) -> str:
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    from app.persona.config import DEFAULT_VERTEX_LOCATION

    vertexai.init(
        project=project_id,
        location=os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
    )
    model = GenerativeModel(model_name)
    cfg = GenerationConfig(temperature=0.2, max_output_tokens=200)
    out = model.generate_content(prompt, generation_config=cfg)
    text = (out.text or "").strip()
    if not text:
        raise RuntimeError("Empty response")
    return text.split("\n")[0].strip().rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--interp", type=Path, default=DEFAULT_INTERP, help="Existing interp to update")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00"))
    ap.add_argument("--only-polysemantic", action="store_true", help="Only re-interpret polysemantic features")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-tokens-per-reply", type=int, default=5)
    args = ap.parse_args()

    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))

    # Determine which features to interpret
    existing_interp = {}
    if args.interp.is_file():
        doc = json.loads(args.interp.read_text(encoding="utf-8"))
        existing_interp = doc.get("features", {})

    all_fids = sorted(int(k) for k in existing_interp.keys())
    if args.only_polysemantic:
        target_fids = [f for f in all_fids if "Polysemantic" in existing_interp.get(str(f), {}).get("interpretation", "")]
        logger.info("Targeting %d polysemantic features", len(target_fids))
    else:
        target_fids = all_fids
        logger.info("Targeting all %d features", len(target_fids))

    # Load model + SAE
    from app.persona.activations import load_model_and_tokenizer
    from app.persona.sae_encode import assistant_hidden_span_at_layer, encode_hidden_span
    from app.phase2 import load_sae_for_layer

    device = torch.device(args.device) if args.device else None
    model, tokenizer, dev = load_model_and_tokenizer(None, device=device)
    sae, sae_info = load_sae_for_layer(dev, release=SAE_RELEASE, sae_id=SAE_ID)

    # Only encode key alphas to keep runtime manageable on CPU (~5min/reply)
    # α=0 (baseline) + α=1.5 (strong shift) gives good contrast
    ENCODE_ALPHAS = os.environ.get("ENCODE_ALPHAS", "0,1.5").split(",")
    ENCODE_ALPHAS = set(a.strip() for a in ENCODE_ALPHAS)

    n_to_encode = sum(
        1 for pq in sweep["per_question"]
        for a in pq["replies"] if a in ENCODE_ALPHAS
    )
    logger.info("Encoding %d replies through SAE at layer %d...", n_to_encode, LAYER)

    # {feature_id: [activation examples]}
    feature_activations: dict[int, list[dict]] = {f: [] for f in target_fids}
    # {feature_id: {alpha_str: mean_activation}}
    feature_alpha_means: dict[int, dict[str, float]] = {f: {} for f in target_fids}

    system_prompt = ""  # alpha sweep used empty/generic system prompt for neg side

    for qi, pq in enumerate(sweep["per_question"]):
        question = pq["question"]
        replies = pq["replies"]

        for alpha_str in sorted(replies.keys(), key=float):
            if alpha_str not in ENCODE_ALPHAS:
                continue
            reply = replies[alpha_str]
            alpha = float(alpha_str)

            try:
                h, token_strs, _ = assistant_hidden_span_at_layer(
                    model, tokenizer, dev, system_prompt, question, reply, LAYER
                )
                z, z_mean = encode_hidden_span(sae, h)
            except Exception as e:
                logger.warning("Failed encoding q%d α=%s: %s", qi, alpha_str, e)
                continue

            for fid in target_fids:
                mean_act = float(z_mean[fid].item())
                prev = feature_alpha_means[fid].get(alpha_str, 0.0)
                # Running mean across questions
                n_prev = qi
                feature_alpha_means[fid][alpha_str] = (prev * n_prev + mean_act) / (n_prev + 1) if n_prev > 0 else mean_act

                examples = collect_top_activations_for_feature(
                    z, token_strs, fid, alpha, qi, top_n=args.top_tokens_per_reply
                )
                feature_activations[fid].extend(examples)

        logger.info("Encoded question %d/%d", qi + 1, len(sweep["per_question"]))

    del model, sae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # For each feature, keep top activating examples sorted by activation magnitude
    for fid in target_fids:
        acts = feature_activations[fid]
        acts.sort(key=lambda x: abs(x["activation"]), reverse=True)
        feature_activations[fid] = acts[:20]

    # Now interpret via Gemini
    results: dict[str, dict] = {}
    n_interpreted = 0

    for fid in target_fids:
        key = str(fid)
        acts = feature_activations[fid]

        if not acts:
            results[key] = {
                "interpretation": f"Feature inactive in Good steering replies",
                "source": "no_activations",
                "top_examples": [],
            }
            continue

        # Check cache
        if not args.force and key in existing_interp:
            cached = existing_interp[key]
            if "Polysemantic" not in cached.get("interpretation", "") and cached.get("source") == "activation_context":
                results[key] = cached
                continue

        alpha_pattern = {"mean_by_alpha": feature_alpha_means[fid]}
        prompt = build_activation_prompt(fid, acts, alpha_pattern)

        try:
            interpretation = interpret_via_gemini(prompt, args.project, args.model)
            source = "activation_context"
            n_interpreted += 1
        except Exception as e:
            logger.error("F%d Gemini failed: %s", fid, e)
            # Fallback: summarize top tokens
            top_toks = list({ex["token"] for ex in acts[:8]})
            interpretation = f"Activates on: {', '.join(top_toks[:6])}"
            source = "token_summary"

        results[key] = {
            "interpretation": interpretation,
            "source": source,
            "top_examples": acts[:8],
            "alpha_pattern": feature_alpha_means[fid],
        }
        logger.info("F%d -> %s", fid, interpretation)

    # Save results
    out_doc = {
        "meta": {
            "method": "activation_context",
            "model": args.model,
            "layer": LAYER,
            "sae_id": SAE_ID,
            "n_replies_encoded": sum(len(pq["replies"]) for pq in sweep["per_question"]),
            "n_features_interpreted": n_interpreted,
        },
        "features": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d features, %d new interpretations)", args.out, len(results), n_interpreted)

    # Also update the main interpretations file with better labels
    if args.interp.is_file():
        main_doc = json.loads(args.interp.read_text(encoding="utf-8"))
        updated = 0
        for key, result in results.items():
            if result["source"] in ("activation_context", "token_summary"):
                if key in main_doc["features"]:
                    old = main_doc["features"][key].get("interpretation", "")
                    if "Polysemantic" in old or args.force:
                        main_doc["features"][key]["interpretation"] = result["interpretation"]
                        main_doc["features"][key]["source"] = result["source"]
                        updated += 1
        if updated:
            args.interp.write_text(json.dumps(main_doc, indent=2) + "\n", encoding="utf-8")
            logger.info("Updated %d labels in %s", updated, args.interp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
