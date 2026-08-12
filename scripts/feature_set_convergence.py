#!/usr/bin/env python3
"""Load independent feature selectors and compute overlap + method-invariant core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.trait_sae_config import resolve_trait


def load_gradsae(path: Path, k: int) -> list[int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("top_positive") or []
    return [int(r["feature_id"]) for r in rows[:k]]


def load_omp(path: Path, k: int) -> list[int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(
        d.get("decomposition") or [],
        key=lambda r: abs(float(r.get("coefficient", 0))),
        reverse=True,
    )
    return [int(r["feature_id"]) for r in rows[:k]]


def load_ssv(path: Path, k: int, ssv_k: int = 100) -> list[int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    for row in d.get("results") or []:
        if int(row.get("k", 0)) == ssv_k:
            fids = row.get("feature_ids") or []
            weights = row.get("feature_weights") or [1.0] * len(fids)
            order = sorted(
                range(len(fids)),
                key=lambda i: abs(float(weights[i])),
                reverse=True,
            )
            return [int(fids[i]) for i in order[:k]]
    return []


def load_fstat_attr(path: Path, k: int) -> list[int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(
        d.get("top_positive_features") or [],
        key=lambda r: float(r.get("shared_magnitude", 0)),
        reverse=True,
    )
    return [int(r["feature_id"]) for r in rows[:k]]


def load_cos_chen(path: Path, k: int) -> list[int]:
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    feats = d.get("features") or []
    if not feats:
        return []
    if feats[0].get("cos_rank") is not None:
        feats = sorted(feats, key=lambda r: r.get("cos_rank", 999))
    else:
        feats = sorted(feats, key=lambda r: -(r.get("cos_to_v") or 0))
    return [int(r["feature_id"]) for r in feats[:k]]


def load_necessity_rank(path: Path, k: int) -> list[int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return list((d.get("ranking") or {}).get("top20_features") or [])[:k]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def vote_core(sets: dict[str, list[int]], min_votes: int) -> list[dict]:
    counts: dict[int, int] = {}
    sources: dict[int, list[str]] = {}
    for name, fids in sets.items():
        for fid in fids:
            counts[fid] = counts.get(fid, 0) + 1
            sources.setdefault(fid, []).append(name)
    return [
        {"feature_id": fid, "votes": v, "sources": sources[fid]}
        for fid, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        if v >= min_votes
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="good")
    ap.add_argument("--sae-dir", default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve_trait(args.trait)
    layer = int(cfg["layer"])
    tag = f"l{layer}"
    sae = Path(args.sae_dir or cfg["sae_dir"])
    k = args.k
    sets = {
        "gradsae_output": load_gradsae(sae / f"causal_screen_A_262k_{tag}.json", k),
        "omp_vdense": load_omp(sae / f"omp_decomposition_262k_{tag}.json", k),
        "ssv_k100": load_ssv(sae / f"sae_ssv_full_sweep_262k_{tag}.json", k, ssv_k=100),
        "fstat_sta": load_fstat_attr(sae / f"feature_attribution_{tag}.json", k),
        "cos_top50": load_cos_chen(sae / f"chen_m32_top50_20q_{tag}.json", k),
        "necessity_dense": load_necessity_rank(sae / f"ablation_necessity_262k_{tag}.json", k),
    }
    sets = {n: fids for n, fids in sets.items() if fids}

    names = list(sets.keys())
    jaccard_matrix = {
        a: {b: round(jaccard(set(sets[a]), set(sets[b])), 4) for b in names}
        for a in names
    }

    cores = {str(t): vote_core(sets, t) for t in (2, 3, 4) if t <= len(names)}

    payload = {
        "method": "feature_set_convergence",
        "trait": cfg["trait"],
        "layer": layer,
        "k": k,
        "selectors": {n: fids for n, fids in sets.items()},
        "pairwise_jaccard": jaccard_matrix,
        "method_invariant_core": cores,
        "interpretation": (
            "Low Jaccard + sparse core => empirical non-identifiability across selectors. "
            "Core features with votes>=2 are candidate method-invariant set for causal tests."
        ),
    }

    out = Path(args.out or sae / f"feature_convergence_{tag}_k{k}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {out}")
    print(f"Selectors ({k} each): {list(sets.keys())}")
    for a in names:
        row = "  %-16s" % a + " ".join(f"{jaccard_matrix[a][b]:5.2f}" for b in names)
        print(row)
    for t, core in cores.items():
        print(f"Core votes>={t}: {[c['feature_id'] for c in core[:15]]}")


if __name__ == "__main__":
    main()
