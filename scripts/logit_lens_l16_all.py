#!/usr/bin/env python3
"""Logit lens for all layer-16 SAE features (CPU, one feature at a time)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PERSONA_FORCE_CPU", "1")

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.persona.activations import load_model_and_tokenizer
from app.phase2 import load_sae_for_layer

TOP_K = 20
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "app/static/logit_lens_l16_all.json"
FIDS: list[int] | None = None
if len(sys.argv) > 2:
    FIDS = [int(x) for x in sys.argv[2].split(",") if x.strip()]


def main() -> int:
    print("Loading model on CPU...")
    model, tokenizer, _ = load_model_and_tokenizer(None, device=torch.device("cpu"))
    sae, _ = load_sae_for_layer(torch.device("cpu"), release="gemma-scope-2-4b-it-res-all", sae_id="layer_16_width_16k_l0_small")
    lm_head = model.lm_head.weight.detach().float()
    W_dec = sae.W_dec.detach().float()
    del model, sae
    d_sae = W_dec.shape[0]
    fids = FIDS if FIDS is not None else list(range(d_sae))
    print(f"Computing logit lens for {len(fids)} features...")
    out: dict[str, dict] = {}
    if OUT.is_file():
        try:
            out = json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out = {}
    for i, fid in enumerate(fids):
        if str(fid) in out and out[str(fid)].get("top_boost"):
            continue
        logits = lm_head @ W_dec[fid]
        top_vals, top_idx = torch.topk(logits, TOP_K)
        bot_vals, bot_idx = torch.topk(-logits, TOP_K)
        out[str(fid)] = {
            "top_boost": [[tokenizer.decode([int(j)]).strip(), round(float(v), 3)] for j, v in zip(top_idx, top_vals)],
            "top_suppress": [[tokenizer.decode([int(j)]).strip(), round(float(v), 3)] for j, v in zip(bot_idx, bot_vals)],
        }
        if i and i % 500 == 0:
            print(f"  {i}/{len(fids)}")
            OUT.write_text(json.dumps(out))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(out)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
