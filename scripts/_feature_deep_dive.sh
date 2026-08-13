#!/usr/bin/env bash
# Run on VM (requires GPU): compute top-80 logit lens for specific feature IDs.
# DO NOT run locally — no GPU means model loading is blocked.
cd "$HOME/gemma-chat"
set -a; [ -f .hf.env ] && . ./.hf.env; set +a
export PYTHONPATH="$HOME/gemma-chat"

# Verify GPU is available before spending time loading
python3 -c "import torch; assert torch.cuda.is_available(), 'NO GPU — run on VM only'" || exit 1

"$HOME/gemma-chat/.venv/bin/python3" << 'PYEOF'
import sys, torch, json
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from app.phase2 import load_sae_for_layer
from trait_sae_config import SAE_RELEASE
from app.persona.activations import load_model_and_tokenizer

dev = torch.device("cuda")
sae, _ = load_sae_for_layer(dev, release=SAE_RELEASE, sae_id="layer_15_width_262k_l0_small", hidden_state_index=16)
W_dec = sae.W_dec.detach().float().cpu()

model, tokenizer, _ = load_model_and_tokenizer(None, device=dev)
raw = model.lm_head.weight.detach().float().cpu()
inner = model.model
norm_w = inner.language_model.norm.weight.detach().float().cpu() if hasattr(inner, "language_model") else inner.norm.weight.detach().float().cpu()
lm_head = raw * (1.0 + norm_w)
del model, sae

results = {}
for fid in (10156, 40036):
    logits = lm_head @ W_dec[fid]
    top_vals, top_idx = torch.topk(logits, 80)
    bot_vals, bot_idx = torch.topk(-logits, 40)
    results[fid] = {
        "top": [[tokenizer.decode([i]), round(float(v), 4)] for i, v in zip(top_idx.tolist(), top_vals.tolist())],
        "suppress": [[tokenizer.decode([i]), round(float(v), 4)] for i, v in zip(bot_idx.tolist(), bot_vals.tolist())],
    }

out = "/tmp/feature_deep_dive.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {out}")
for fid in (10156, 40036):
    print(f"\nF{fid} top 80:")
    for tok, score in results[fid]["top"]:
        print(f"  {score:+.3f}  {repr(tok)}")
    print(f"F{fid} suppressed:")
    for tok, score in results[fid]["suppress"]:
        print(f"  {score:+.3f}  {repr(tok)}")
PYEOF
