#!/usr/bin/env python3
"""
Deep logit-lens for selected SAE features: full-vocab projection, top-K tokens,
and exact ranks of diagnostic marker tokens (denominational / cross-religion).

Usage (GPU VM):
  cd ~/gemma-chat && PYTHONPATH=. .venv/bin/python scripts/feature_lens_deep.py --out /tmp/lens_deep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persona.activations import load_model_and_tokenizer
from app.phase2 import load_sae_for_layer
from scripts.trait_sae_config import SAE_RELEASE, hidden_state_index, sae_id_for_layer
from scripts.ssv_feature_logit_lens import effective_lm_head

FIDS = [10156, 163384, 11023, 14978, 16833]

MARKERS = {
    "catholic": [" Catholic", " Catholics", " Catholicism", " Pope", " papal", " Vatican",
                 " Mass", " priest", " priests", " bishop", " bishops", " cathedral",
                 " parish", " diocese", " Eucharist", " Rosary", " saint", " saints",
                 " monastery", " nun", " friar", " homily"],
    "orthodox": [" Orthodox", " Patriarch", " icon"],
    "protestant_control": [" pastor", " sermon", " Baptist", " evangelical", " congregation",
                           " devotional", " Protestant", " Lutheran", " Methodist", " Presbyterian"],
    "islam": [" Islam", " Muslim", " Quran", " mosque", " imam", " Allah", " Ramadan"],
    "judaism": [" Torah", " rabbi", " synagogue", " Jewish", " Hanukkah"],
    "dharmic": [" Hindu", " Krishna", " Buddhist", " Buddha", " meditation", " karma", " temple"],
    "deity_generic": [" God", " Lord", " divine", " holy", " sacred", " heaven", " worship",
                      " faith", " prayer", " blessing"],
    "moral": [" mercy", " compassion", " kindness", " charity", " forgiveness", " virtue",
              " righteous", " sin", " evil", " cruel"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--width", default="262k")
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("/tmp/lens_deep.json"))
    args = ap.parse_args()

    model, tokenizer, dev = load_model_and_tokenizer()
    sae, _ = load_sae_for_layer(
        torch.device("cpu"), release=SAE_RELEASE,
        sae_id=sae_id_for_layer(args.layer, args.width),
        hidden_state_index=hidden_state_index(args.layer),
    )
    W_dec = sae.W_dec.detach().float().cpu()
    lm = effective_lm_head(model).cpu()  # (V, d)
    vocab_size = lm.shape[0]
    print(f"vocab={vocab_size}  d={lm.shape[1]}", file=sys.stderr)

    # resolve marker token ids (single-token only)
    marker_ids: dict[str, dict[str, int]] = {}
    for cat, words in MARKERS.items():
        marker_ids[cat] = {}
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                marker_ids[cat][w] = ids[0]
            else:
                marker_ids[cat][w] = -1  # multi-token, skip

    out = {"layer": args.layer, "sae_id": sae_id_for_layer(args.layer, args.width),
           "vocab_size": int(vocab_size), "features": {}}

    for fid in FIDS:
        logits = lm @ W_dec[fid]  # (V,)
        order = torch.argsort(logits, descending=True)
        rank_of = torch.empty_like(order)
        rank_of[order] = torch.arange(vocab_size)

        topk_idx = order[: args.topk]
        top = [[tokenizer.decode([int(i)]), round(float(logits[i]), 3)] for i in topk_idx]

        ranks = {}
        for cat, wmap in marker_ids.items():
            ranks[cat] = {}
            for w, tid in wmap.items():
                if tid < 0:
                    ranks[cat][w] = None
                else:
                    r = int(rank_of[tid])
                    ranks[cat][w] = {"rank": r + 1, "logit": round(float(logits[tid]), 3),
                                     "pctile": round((r + 1) / vocab_size * 100, 3)}

        out["features"][str(fid)] = {"top": top, "marker_ranks": ranks}
        print(f"F{fid} done", file=sys.stderr)

    args.out.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
