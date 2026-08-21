#!/usr/bin/env python3
"""Why a monotone prompt ladder does not become a monotone steering vector.

Runs on cached ``ladder_vectors_*.pt`` / ``centroids_*.pt`` — CPU only, no model
load. Answers four questions per trait, at every layer:

    ramp or step   Does the level->projection curve fit a line better than a
                   two-group step? Rank statistics (Spearman, monotone fraction)
                   score a perfect 1.0 on a step function, so the original layer
                   picker could not tell these apart. R^2 can.
    gain leak      |cos(mean centroid, direction)|. A direction aligned with the
                   mean residual is a global gain knob: adding it scales the
                   residual stream rather than moving along a trait axis.
    signal / noise Between-level span against same-level prompt-wording scatter.
                   Three marker variants per level is a noisy centroid estimate.
    graded span    Span after the common direction is projected out, which is
                   the part a trait-specific offset could actually exploit.

Usage:
    python3 scripts/diagnose_ladder_geometry.py --vectors-dir /tmp/keep_vectors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
NEUTRAL_CUT = 5  # levels 1..5 are low..neutral, 6..9 are high


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for pos, i in enumerate(order):
        out[i] = pos + 1.0
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    return num / den if den else 0.0


def r2_linear(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((a - mx) ** 2 for a in xs)
    if denom == 0:
        return 0.0
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / denom
    intercept = my - slope * mx
    ss = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(xs, ys))
    tot = sum((b - my) ** 2 for b in ys)
    return 1 - ss / tot if tot else 0.0


def r2_step(ys: list[float], cut: int = NEUTRAL_CUT) -> float:
    lo, hi = ys[:cut], ys[cut:]
    if not lo or not hi:
        return 0.0
    ml, mh, my = sum(lo) / len(lo), sum(hi) / len(hi), sum(ys) / len(ys)
    ss = sum((v - ml) ** 2 for v in lo) + sum((v - mh) ** 2 for v in hi)
    tot = sum((v - my) ** 2 for v in ys)
    return 1 - ss / tot if tot else 0.0


def _pc1(centroids: torch.Tensor) -> torch.Tensor:
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    v = vh[0]
    if torch.dot(v, centroids[-1] - centroids[0]) < 0:
        v = -v
    return v


def _drop_common(x: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """Remove the shared/mean residual component, the global-gain confound."""
    return x - (x @ unit).unsqueeze(-1) * unit


def layer_report(
    centroids: torch.Tensor, acts: torch.Tensor | None, orthogonalize: bool
) -> dict[str, float]:
    """``centroids`` is (n_levels, d); ``acts`` is (n_levels, n_variants, d)."""
    mean_c = centroids.mean(dim=0)
    unit = mean_c / mean_c.norm().clamp_min(1e-9)
    raw_pc1 = _pc1(centroids)
    gain_leak = abs(float(torch.dot(raw_pc1, unit)))

    work = _drop_common(centroids, unit) if orthogonalize else centroids
    direction = _pc1(work)
    proj = [float(torch.dot(c, direction)) for c in work]
    levels = [float(i + 1) for i in range(len(proj))]

    scatter = None
    if acts is not None:
        a = _drop_common(acts, unit) if orthogonalize else acts
        per_level = a.mean(dim=1)
        scatter = float(((a - per_level.unsqueeze(1)).norm(dim=-1) ** 2).mean().sqrt())

    span = abs(proj[-1] - proj[0])
    return {
        "r2_linear": r2_linear(levels, proj),
        "r2_step": r2_step(proj),
        "spearman": spearman(levels, proj),
        "within_low": spearman(levels[:NEUTRAL_CUT], proj[:NEUTRAL_CUT]),
        "within_high": spearman(levels[NEUTRAL_CUT:], proj[NEUTRAL_CUT:]),
        "span": span,
        "gain_leak": gain_leak,
        "snr": (span / scatter) if scatter else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors-dir", default="/tmp/keep_vectors")
    p.add_argument("--layer", type=int, default=None, help="Report one layer only.")
    p.add_argument("--top", type=int, default=4, help="Best layers to list per trait.")
    p.add_argument(
        "--raw",
        action="store_true",
        help="Do not project out the common direction (reproduces the original bug).",
    )
    p.add_argument("--min-span", type=float, default=200.0, help="Span floor for ranking.")
    args = p.parse_args(argv)

    vdir = Path(args.vectors_dir)
    ortho = not args.raw
    print(f"vectors: {vdir}   common-direction removed: {ortho}")

    for trait in TRAITS:
        vec_pt = vdir / f"ladder_vectors_{trait}.pt"
        cen_pt = vdir / f"centroids_{trait}.pt"
        if not vec_pt.is_file():
            print(f"\n{trait}: missing {vec_pt.name}")
            continue
        blob = torch.load(vec_pt, map_location="cpu")
        centroids = blob["level_centroids"].float()  # (levels, layers, d)
        acts = None
        if cen_pt.is_file():
            acts = torch.load(cen_pt, map_location="cpu")["activations"].float()

        rows = []
        for layer in range(centroids.shape[1]):
            a = acts[:, :, layer, :] if acts is not None else None
            rep = layer_report(centroids[:, layer, :], a, ortho)
            graded = max(0.0, (rep["within_low"] + rep["within_high"]) / 2)
            snr_term = 0.0 if rep["snr"] != rep["snr"] else min(1.0, rep["snr"] / 3.0)
            gate = 1.0 if rep["span"] >= args.min_span else rep["span"] / args.min_span
            rep["score"] = rep["r2_linear"] * graded * snr_term * gate
            rep["layer"] = layer
            rows.append(rep)

        if args.layer is not None:
            rows = [r for r in rows if r["layer"] == args.layer]
        else:
            rows.sort(key=lambda r: r["score"], reverse=True)
            rows = rows[: args.top]

        print(f"\n{trait}:")
        print(
            f"    {'L':>3}{'score':>8}{'R2lin':>8}{'R2step':>8}{'fit':>7}"
            f"{'span':>10}{'SNR':>7}{'gainleak':>10}{'withinRho':>11}"
        )
        for r in rows:
            fit = "ramp" if r["r2_linear"] > r["r2_step"] else "STEP"
            within = (r["within_low"] + r["within_high"]) / 2
            print(
                f"    {r['layer']:>3}{r['score']:>8.3f}{r['r2_linear']:>8.3f}"
                f"{r['r2_step']:>8.3f}{fit:>7}{r['span']:>10.1f}{r['snr']:>7.2f}"
                f"{r['gain_leak']:>10.3f}{within:>11.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
