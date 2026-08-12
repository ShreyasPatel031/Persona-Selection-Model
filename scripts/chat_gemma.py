"""Interactive terminal chat with the local Gemma model.

Usage:
    python3 scripts/chat_gemma.py
    python3 scripts/chat_gemma.py --system "You are a terse pirate."

Loads the model once (bfloat16 — fp16 overflows on Gemma) and keeps it
resident. Type a message and press Enter. Commands: /reset clears history,
/exit or Ctrl-D quits.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it"))
    ap.add_argument("--system", default="")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--greedy", action="store_true", help="deterministic decoding")
    args = ap.parse_args()

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    print(f"Loading {args.model_id} on {dev} (bfloat16)…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(dev).eval()
    print("Ready. Type a message ( /reset, /exit ).\n", flush=True)

    history: list[dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": user})
        ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        ids = ids.to(dev)

        gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
        if args.greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.9)

        with torch.no_grad():
            out = model.generate(ids, **gen_kwargs)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"gemma> {reply}\n", flush=True)


if __name__ == "__main__":
    main()
