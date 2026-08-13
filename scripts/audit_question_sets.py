#!/usr/bin/env python3
"""Audit which question sets are used across good_scale pipelines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIVE_MARKERS = [
    "lifeboat with limited supplies",
    "displace a small, marginalized",
    "expensive, medical treatment",
    "minor crime out of desperation",
    "expose a powerful, corrupt",
]
ELDERLY = "elderly person fall"


def classify(qs: list[str]) -> str:
    if not qs:
        return "NONE"
    blob = " ".join(qs).lower()
    n5 = sum(1 for m in FIVE_MARKERS if m.lower() in blob)
    if n5 >= 4:
        return f"TOWER5 ({n5}/5)"
    if ELDERLY in blob:
        return "BUNDLE_EVAL (elderly-set)"
    return "OTHER"


def uniq(qs: list[str]) -> list[str]:
    out, seen = [], set()
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def extract_questions(obj) -> list[str]:
    qs: list[str] = []
    if isinstance(obj, dict):
        for key in (
            "eval_questions_used",
            "eval_questions",
            "questions",
            "contrast_scenarios",
            "extraction_questions",
            "calibration_question",
        ):
            if key not in obj:
                continue
            v = obj[key]
            if isinstance(v, str):
                qs.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, str):
                        qs.append(it)
                    elif isinstance(it, dict):
                        for kk in ("question", "scenario", "prompt", "text"):
                            if isinstance(it.get(kk), str):
                                qs.append(it[kk])
                                break
        if "per_question" in obj and isinstance(obj["per_question"], list):
            for it in obj["per_question"]:
                if isinstance(it, dict) and "question" in it:
                    qs.append(it["question"])
        if "by_alpha" in obj and isinstance(obj["by_alpha"], dict):
            for entry in obj["by_alpha"].values():
                for s in entry.get("samples") or []:
                    if isinstance(s, dict) and "question" in s:
                        qs.append(s["question"])
        if "samples" in obj and isinstance(obj["samples"], list):
            for s in obj["samples"]:
                if isinstance(s, dict) and "question" in s:
                    qs.append(s["question"])
    return uniq(qs)


def show(label: str, qs: list[str], extra: str = "") -> None:
    qs = uniq(qs)
    print(f"{label:55s} {classify(qs):28s} n={len(qs):3d} {extra}")


def main() -> int:
    # Prefer VM layout if present under ~/gemma-chat, else local repo
    roots = [Path.home() / "gemma-chat", ROOT]
    root = next((r for r in roots if (r / "persona_runs").is_dir()), ROOT)
    print(f"ROOT={root}\n")

    bundle_scale = root / "persona_runs/dnd_good_scale/artifacts/trait_bundle.json"
    bundle_old = root / "persona_runs/dnd_good/artifacts/trait_bundle.json"

    for bp, name in ((bundle_scale, "good_scale bundle"), (bundle_old, "good (old) bundle")):
        if not bp.is_file():
            print(f"{name}: MISSING at {bp}")
            continue
        b = json.loads(bp.read_text())
        print(f"=== {name} ===")
        for k in ("contrast_scenarios", "extraction_questions", "eval_questions"):
            show(f"  {k}", b.get(k) or [])
            for q in (b.get(k) or [])[:2]:
                print(f"      · {q[:100]}")
        print()

    files = [
        ("tower layer3d_alpha_sweep", "app/static/layer3d_alpha_sweep.json"),
        ("tower good activation", "app/static/layer3d_good_trait_activation.json"),
        ("validation_report (gate3 means)", "persona_runs/dnd_good_scale/eval/validation_report.json"),
        ("alpha_experiment (tower-era)", "persona_runs/dnd_good_scale/alpha_experiment.json"),
        ("sae alpha_sweep_analysis", "persona_runs/dnd_good_scale/sae/alpha_sweep_analysis.json"),
        ("sae sufficiency_v2", "persona_runs/dnd_good_scale/sae/sufficiency_baseline_matrix_v2_l16.json"),
        ("sae necessity", "persona_runs/dnd_good_scale/sae/necessity_default_good_l16.json"),
        ("sae feature_attribution", "persona_runs/dnd_good_scale/sae/feature_attribution_l16.json"),
        ("sae omp_trait_coherence", "persona_runs/dnd_good_scale/sae/omp_trait_coherence_262k_l16.json"),
        ("sae single_feature_sweep", "persona_runs/dnd_good_scale/sae/single_feature_scale_sweep_l16.json"),
        ("sae ssv_k_sweep_20q", "persona_runs/dnd_good_scale/sae/ssv_k_sweep_l15_20q_emd.json"),
        ("sae omp_k_sweep_20q", "persona_runs/dnd_good_scale/sae/omp_k_sweep_l15_20q_emd.json"),
        ("calibration_v2 (OLD L31)", "persona_runs/dnd_calibration_v2.json"),
    ]

    print("=== downstream artifacts ===")
    for label, rel in files:
        # try both root and local ROOT for app/static
        candidates = [root / rel, ROOT / rel]
        p = next((c for c in candidates if c.is_file()), None)
        if p is None:
            print(f"{label:55s} MISSING")
            continue
        d = json.loads(p.read_text())
        if "calibration_v2" in rel and "good" in d:
            d = d["good"]
        qs = extract_questions(d)
        # validation report: no questions stored
        extra = f"path={p}"
        if "layer" in d:
            extra = f"layer={d.get('layer')} " + extra
        if isinstance(d, dict) and "n_questions" in d:
            extra = f"n_questions_field={d['n_questions']} " + extra
        show(label, qs, extra=extra)

    print(
        "\nNOTE: extraction rollouts use extraction_questions (+ contrast prompts),\n"
        "      Gate3/steering/SAE eval typically slice artifact.eval_questions.\n"
        "      Tower means come from validation_report Gate3 (questions not stored in JSON)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
