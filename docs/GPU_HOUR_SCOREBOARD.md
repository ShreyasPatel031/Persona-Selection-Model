# GPU-hour optimization — baseline scoreboard

Use this table to compare **new probes** (CPU vs GPU, token caps, rollout tiers) against fixed baselines. Update rows when you record a new run.

## Reference baselines (from repo / VM measurements)

| Run / source | Hardware | Throughput (rollouts/h) | Scale (step-c) | Judge keep-rate | Split-half cosine | Projection margin @ default layer | Notes |
|--------------|----------|-------------------------|----------------|-----------------|-------------------|-----------------------------------|--------|
| `trait_worst_in_situations_v4` | (historical) | n/a | 16 rows, `rollouts_per_q=1`, 1 pair | **25/32 = 78.1%** (pos 62.5%, neg 93.8%) | (see `vectors/summary.json`) | **1.0** (`fraction_margin_positive_at_default_layer`, `num_used=8`) | Completed pipeline in repo |
| `evil_paper_v0` (CPU VM) | `n1-standard-8`, no GPU | **~12.7** (~154 rollouts / ~12.1 h) | paper bundle, `rollouts_per_q=10` | n/a (run incomplete) | n/a | n/a | ETA ~79 h for 1000 rollouts gen only |
| **`evil_probe_cpu`** (tiny CPU) | `n1-standard-8` (gemma-mvp), CPU | **~23.7** (20 judged gens / **3040 s** step-c wall) | `--limit 2 --rollouts-per-q 1`, paper bundle | **70%** (14/20 kept: pos 4 + neg 10 / 20 judged) | **0.9556** (max @ layer 30) | n/a (not run) | `vectors/summary.json`; validate data sufficiency **FAIL** (low N) |
| **`evil_paper_v0` gpu-probe** | `n1-standard-8` + **T4** (`n1-8+t4`), `gemma-gpu-probe-1774902210` | **~180 est.** step-c (20 gens; **~779 s** full `gpu-probe` incl. VM create + bootstrap + load + step-c) | same tiny probe | **70%** (14/20; matches CPU tiny) | n/a (step-d not in gpu-probe) | n/a | `gpu_probe_report.json` **status ok**; rollouts in `persona_runs/evil_paper_v0_gpu_probe/rollouts/` |
| **Delta (GPU vs CPU tiny)** | | **~7.6×** (180 / 23.7, **approx.**) | same | ~0 pp | — | — | Vertex judge still dominates wall time; GPU speeds Gemma. |
| **`evil_gate_v0`** (medium, GPU VM) | `n1-8+t4`, `gemma-gpu-probe-1774902210` | (see step-c log; ~18 min step-c wall for 80 judged gens) | `--limit 8 --rollouts-per-q 1` | 56/80 kept (see `extraction_rollouts.json`) | **nan** (noise_dominated @ L6 in step-d) | **0.0** @ default layer | **`validation_report.json` → Overall FAIL** (data &lt; 50/arm, split-half nan, separation 0%). **Do not scale** `rollouts_per_q` until gates pass. |
| **`evil_scale_v0`** (scaled GPU) | `n1-8+t4` (VM **deleted** after run) | **~276** (400 judged gens / **~5217 s** step-c wall) | `--limit 20 --rollouts-per-q 2` (5×20×2 slots) | **80 pos + 200 neg kept** (Gate 0 **PASS**) | **nan** (noise_dominated @ L6) | **0.0** | **`validation_report.json` → Overall FAIL** (split-half, separation, steering). Fix prompts / trait signal before more scale. |
| **`evil_iter1`** (tiered probe iter 1) | `gemma-mvp` + **T4**, bf16 step-d | (see VM logs; 160 judged gens step-c) | `--limit 8 --rollouts-per-q 2`, paper bundle | **31 pos + 80 neg kept** (Gate 0 **FAIL** &lt; 50/arm) | **0.984** (stable @ L30) | **1.0** @ default layer (`sanity_eval_projection.json`) | **`validation_report.json` → Overall FAIL** (data + layer/steering gates). dtype fix: **finite `v`**, separation **PASS**. |

## Where metrics live

| Metric | Location |
|--------|----------|
| Keep-rate, pos/neg kept, errors | `rollouts/extraction_rollouts.json` → `stats`; `rollouts/rollouts.jsonl` per-line `kept` |
| Kept counts for step-d | `vectors/summary.json` → `kept_pos`, `kept_neg` |
| Split-half cosine | step-d stdout / `vectors/summary.json` (as produced by your run) |
| Projection margin | `eval/sanity_eval_projection.json` → `fraction_margin_positive_at_default_layer`, `mean_margin_per_layer` |
| Throughput | step-c log: count `Rollout` lines ÷ wall hours |

## Stop / go (manual)

- **Stop infra**: throughput vs budget (rollouts/h × ETA for target scale).
- **Stop data**: keep-rate collapse, high `pos_errors`/`neg_errors`, near-zero kept rows.
- **Stop signal**: split-half &lt; 0.5 or margin fraction &lt; 0.7 after medium probe — fix bundle/thresholds before scaling `rollouts_per_q`.

See [GPU_PROBE_WORKFLOW.md](GPU_PROBE_WORKFLOW.md) for phased commands.

### Incremental `rollouts_per_q` scaling (2026-03-30)

**2026-04-01 — `evil_scale_v0`:** Ran `--limit 20 --rollouts-per-q 2` on GPU; **Gate 0 (data) PASS** (80/200 kept); **overall validate FAIL** (split-half nan, separation 0%, steering). **Do not** add more `rollouts_per_q` tiers until vector quality improves — likely need **Step B** (prompts / contrast pairs that elicit trait without refusal) or model change, not raw volume alone.

### `gpu_nan_repro` — step-d NaN isolation (2026-04-01)

- **Data:** 4 kept pos + 4 kept neg lines sliced from `evil_scale_v0/rollouts/rollouts.jsonl` → `persona_runs/gpu_nan_repro/rollouts/rollouts.jsonl`.
- **CPU `step-d` (`--force-cpu`):** `v` is **all finite** (`persona_vectors.pt`); `v_l2_norm_per_layer` **no NaNs**; split-half **~0.865 (stable)** @ layer 31 — same pipeline, tiny data, **healthy vector**.
- **GPU:** This laptop has **no CUDA** (MPS/CPU only). **`evil_scale_v0` on T4 + old code used `float16` on CUDA** and produced **NaN norms from mid-layers** — consistent with **fp16 numerics**, not “no signal.”
- **Fix in code:** [`app/persona/activations.py`](../app/persona/activations.py) `load_model_and_tokenizer` picks **`bfloat16` on CUDA when supported**, else **`float32`** (fp16 is not a default — it matched the old NaN behavior on T4). **`PERSONA_CUDA_ALLOW_FP16=1`** opts into float16 if you need VRAM headroom and accept risk. Re-run `step-d` on a CUDA host to confirm norms match CPU.
