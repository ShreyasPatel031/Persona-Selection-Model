#!/usr/bin/env python3
"""Print pairwise cosine similarity at steering layer for all 4 D&D vectors."""
import json, sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parent.parent

cfg = json.loads((root / "persona_runs/dnd_config.json").read_text())
LAYER = 16

vectors = {}
for name in ("lawful", "chaotic", "good", "evil"):
    p = root / cfg[name]["vectors"]
    ck = torch.load(p, map_location="cpu", weights_only=False)
    vectors[name] = ck["v"].float()[LAYER]

names = list(vectors.keys())
print(f"Cosine similarity at layer {LAYER}:")
print(f"{'':10s}", end="")
for n in names:
    print(f"{n:>10s}", end="")
print()

for a in names:
    print(f"{a:10s}", end="")
    for b in names:
        va, vb = vectors[a], vectors[b]
        cos = float((va @ vb) / (va.norm() * vb.norm()))
        print(f"{cos:10.4f}", end="")
    print()

print("\nNorms:")
for n in names:
    print(f"  {n:10s}: {float(vectors[n].norm()):.4f}")
