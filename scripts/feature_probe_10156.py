#!/usr/bin/env python3
"""
Contrastive activation probe for a single SAE feature (default F10156, "Religious
Concepts", gemma-scope-2-4b-it-res-all layer_15_width_262k_l0_small).

Goal: an UNBIASED semantic characterization of what the feature means, using
syntax-MATCHED sentence frames so any activation difference reflects content,
not sentence length/structure.

Design:
  - Same 5 frames instantiated across 6 religions + a secular control.
  - Extra probe sets: mythology (religious-form but "dead"), secular spirituality,
    religious-word metaphors/idioms, abstract religious concepts (unnamed religion),
    and single-token lexical items.
  - For each sentence we record F10156 activation at every token, then the peak
    activation + which token triggered it + the mean over content tokens.
  - Aggregated by category (mean/std/max of per-sentence peaks).

Tests three questions:
  Q1 Abrahamic-specific vs universal religion?  (compare religions)
  Q2 Monosemantic vs polysemantic?              (religion vs metaphor/idiom/myth/secular)
  Q3 Entity vs concept vs sentiment trigger?    (peak-token analysis + abstract set)

Usage (GPU VM):
  cd ~/gemma-chat && .venv/bin/python scripts/feature_probe_10156.py --fid 10156 --out /tmp/probe_10156.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "applied-ai-practice00")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, sae_id_for_layer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("feature_probe")

LAYER = 15

# ---------------------------------------------------------------------------
# Matched frames. Each religion fills the same 5 templates -> structure held
# constant. Fields: scholar, scripture, holy_day, place, practice, figure,
# role, religion, virtue, clergy.
# ---------------------------------------------------------------------------
FRAMES = [
    "The {scholar} devoted his life to studying the {scripture}.",
    "On {holy_day}, the faithful gather at the {place} to {practice}.",
    "{figure} is revered as {role} in {religion}.",
    "She read a passage from the {scripture} and reflected on its teaching about {virtue}.",
    "The {clergy} led the congregation in {practice}.",
]

RELIGIONS = {
    "christianity": dict(scholar="theologian", scripture="Bible", holy_day="Sunday", place="church",
                         practice="pray", figure="Jesus", role="the son of God", religion="Christianity",
                         virtue="forgiveness", clergy="priest"),
    "judaism": dict(scholar="rabbi", scripture="Torah", holy_day="the Sabbath", place="synagogue",
                    practice="pray", figure="Moses", role="a prophet", religion="Judaism",
                    virtue="justice", clergy="rabbi"),
    "islam": dict(scholar="imam", scripture="Quran", holy_day="Friday", place="mosque",
                  practice="pray", figure="Muhammad", role="the final prophet", religion="Islam",
                  virtue="charity", clergy="imam"),
    "hinduism": dict(scholar="pandit", scripture="Vedas", holy_day="Diwali", place="temple",
                     practice="worship", figure="Krishna", role="an avatar of Vishnu", religion="Hinduism",
                     virtue="duty", clergy="priest"),
    "buddhism": dict(scholar="monk", scripture="sutras", holy_day="Vesak", place="temple",
                     practice="meditate", figure="the Buddha", role="the enlightened one", religion="Buddhism",
                     virtue="compassion", clergy="monk"),
    "sikhism": dict(scholar="granthi", scripture="Guru Granth Sahib", holy_day="Vaisakhi", place="gurdwara",
                    practice="pray", figure="Guru Nanak", role="the founder", religion="Sikhism",
                    virtue="service", clergy="granthi"),
}

# Secular control: same frames, non-religious content.
SECULAR_CONTROL = dict(scholar="historian", scripture="archives", holy_day="Monday", place="office",
                       practice="work", figure="Einstein", role="a genius", religion="physics",
                       virtue="gravity", clergy="manager")

# Free-form probe sets (not frame-matched; test polysemy / breadth).
FREEFORM = {
    "mythology": [
        "Zeus hurled his thunderbolt from the summit of Mount Olympus.",
        "The ancient Greeks built temples to honor Athena and Apollo.",
        "Odin sacrificed his eye at the well of wisdom in Norse legend.",
        "Ra sailed across the sky each day in Egyptian mythology.",
        "The oracle at Delphi spoke prophecies on behalf of the gods.",
    ],
    "secular_spirituality": [
        "She checks her horoscope every morning before deciding anything.",
        "The wellness retreat focused on mindfulness and inner energy.",
        "He believes the alignment of the planets shapes his mood.",
        "Crystals and meditation help her feel centered and calm.",
        "The life coach talked about manifesting abundance into your life.",
    ],
    "religious_metaphor_idiom": [
        "That new phone is the holy grail of gadgets.",
        "He treats the coach's playbook as gospel.",
        "The startup founder preached the gospel of relentless growth.",
        "Silicon Valley is a temple of innovation and disruption.",
        "She is a true believer in the free market.",
    ],
    "abstract_religious_concepts": [
        "He sought forgiveness and redemption for what he had done.",
        "Her unwavering faith gave her strength through the ordeal.",
        "They spoke of sin, salvation, and the fate of the soul.",
        "Devotion and reverence filled the silent hall.",
        "The pilgrimage was an act of penance and spiritual longing.",
    ],
    "secular_neutral": [
        "The chef reduced the sauce over medium heat for ten minutes.",
        "Quarterly revenue exceeded analyst expectations this year.",
        "The striker scored twice in the second half of the match.",
        "The committee reviewed the budget for the new highway.",
        "She debugged the null pointer exception before lunch.",
    ],
    "lexical_single": [
        "Bible", "church", "prayer", "theology", "God", "priest", "sacred", "holy",
        "worship", "faith", "computer", "highway", "banana", "quarterly", "thunder", "gravity",
    ],
}


def build_sentences() -> list[dict]:
    items: list[dict] = []
    for rel, fields in RELIGIONS.items():
        for i, frame in enumerate(FRAMES):
            items.append({"category": rel, "frame": i, "text": frame.format(**fields)})
    for i, frame in enumerate(FRAMES):
        items.append({"category": "secular_control", "frame": i, "text": frame.format(**SECULAR_CONTROL)})
    for cat, sents in FREEFORM.items():
        for s in sents:
            items.append({"category": cat, "frame": None, "text": s})
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", type=int, default=10156)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--width", default="262k")
    ap.add_argument("--out", type=Path, default=Path("/tmp/probe_10156.json"))
    args = ap.parse_args()

    fid = args.fid
    layer = args.layer
    sae_id = sae_id_for_layer(layer, args.width)

    model, tokenizer, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        torch.device("cpu"), release=SAE_RELEASE, sae_id=sae_id,
        hidden_state_index=hidden_state_index(layer),
    )
    sae_dev = dev if dev.type == "cuda" else torch.device("cpu")
    if sae_dev.type == "cuda":
        sae = sae.to(sae_dev)

    from app.persona.steering_demo import _language_model_layers
    layers = _language_model_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = layers[layer].register_forward_hook(_hook)

    sentences = build_sentences()
    logger.info("Probing feature %d (L%d, %s) across %d sentences", fid, layer, sae_id, len(sentences))

    results = []
    try:
        for item in sentences:
            ids = tokenizer(item["text"], return_tensors="pt", add_special_tokens=True)
            input_ids = ids["input_ids"].to(dev)
            attn = ids["attention_mask"].to(dev)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            h = captured["h"][0].float().to(sae_dev)  # (seq, d)
            with torch.no_grad():
                z = sae.encode(h.unsqueeze(0))[0][:, fid].float().cpu()  # (seq,)
            toks = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
            acts = z.tolist()
            peak_idx = int(torch.tensor(acts).argmax().item())
            # skip BOS token (index 0) for mean over content
            content = acts[1:] if len(acts) > 1 else acts
            results.append({
                **item,
                "peak_act": round(max(acts), 4),
                "peak_token": toks[peak_idx],
                "mean_content_act": round(sum(content) / max(len(content), 1), 4),
                "n_active_tokens": int(sum(1 for a in acts if a > 1e-4)),
                "n_tokens": len(acts),
                "per_token": [[toks[i], round(acts[i], 3)] for i in range(len(acts))],
            })
            logger.info("[%s] peak=%.3f @ %r  | %s", item["category"], max(acts), toks[peak_idx], item["text"][:60])
    finally:
        handle.remove()

    # Aggregate by category
    from collections import defaultdict
    agg: dict[str, list[float]] = defaultdict(list)
    for r in results:
        agg[r["category"]].append(r["peak_act"])
    summary = {}
    for cat, peaks in agg.items():
        t = torch.tensor(peaks)
        summary[cat] = {
            "n": len(peaks),
            "mean_peak": round(float(t.mean()), 4),
            "std_peak": round(float(t.std(unbiased=False)), 4),
            "max_peak": round(float(t.max()), 4),
            "min_peak": round(float(t.min()), 4),
        }

    out = {
        "fid": fid, "layer": layer, "sae_id": sae_id, "release": SAE_RELEASE,
        "n_sentences": len(sentences),
        "summary_by_category": dict(sorted(summary.items(), key=lambda kv: -kv[1]["mean_peak"])),
        "sentences": results,
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", args.out)

    # Console summary
    print("\n=== F%d activation by category (mean peak, desc) ===" % fid)
    for cat, s in out["summary_by_category"].items():
        print(f"  {cat:26s} mean={s['mean_peak']:7.3f}  max={s['max_peak']:7.3f}  (n={s['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
