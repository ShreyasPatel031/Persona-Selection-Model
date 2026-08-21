#!/usr/bin/env python3
"""Measure full-sequence vs assistant-span injection lengths (tokenizer only, no GPU).

Verifies the E0 hypothesis setup: our inventory prompts are ~60+ tokens but
Blas-style assistant-span injection covers only the generation-prompt tail.

    PYTHONPATH=. python3 scripts/measure_injection_span.py \\
        --model-id unsloth/gemma-3-4b-it
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="unsloth/gemma-3-4b-it")
    p.add_argument("--items-csv", default=str(REPO_ROOT / "data" / "ipip_neo_120.csv"))
    p.add_argument("--max-items", type=int, default=24)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from app.persona.intensity_ladder import (
        _inventory_assistant_start_unpadded,
        persona_free_system_prompt,
    )
    from app.persona.inventory_ipip import items_from_csv, item_user_message

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    system = f"{persona_free_system_prompt()}\n\nAnswer each item with a single letter A–E only."
    items = [i for i in items_from_csv(Path(args.items_csv)) if i.trait == "conscientiousness"][
        : args.max_items
    ]

    full_lens: list[int] = []
    span_lens: list[int] = []
    for item in items:
        user = item_user_message(item)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        full_len = int(ids.shape[-1])
        start = _inventory_assistant_start_unpadded(tokenizer, system, user)
        full_lens.append(full_len)
        span_lens.append(full_len - start)

    print(f"model: {args.model_id}")
    print(f"items: {len(items)} conscientiousness IPIP items")
    print(
        f"full sequence length: mean={statistics.mean(full_lens):.1f} "
        f"min={min(full_lens)} max={max(full_lens)}"
    )
    print(
        f"assistant_span length: mean={statistics.mean(span_lens):.1f} "
        f"min={min(span_lens)} max={max(span_lens)}"
    )
    print(f"exposure ratio (full/span): {statistics.mean(full_lens)/statistics.mean(span_lens):.1f}×")
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402 — used only for Tensor type check

    raise SystemExit(main())
