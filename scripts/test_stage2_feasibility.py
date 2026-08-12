#!/usr/bin/env python3
"""
Phase 0 feasibility test: Stage 2 from He et al. SAE-SSV (EMNLP 2025).

Runs entirely on CPU using the cached z-activations. No model needed.

Steps:
  1. Load z-cache (SAE activations for pos/neg samples)
  2. F-stat -> top-K=1024 coarse selection (Stage 1)
  3. Train M=50 logistic-regression classifiers on random 80% subsamples
  4. Average weight vectors -> v_avg
  5. Sweep d=1..K: truncate v_avg to top-d by |magnitude|, compute
     separation score s(d) = mean_cosine_pos - mean_cosine_neg
  6. Find d_steer = elbow point
  7. Print results + ASCII plot

Usage:
  python scripts/test_stage2_feasibility.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
Z_CACHE = REPO / "persona_runs/dnd_good_scale/sae/probe_z_cache_l16.npz"

K_COARSE = 1024
M_CLASSIFIERS = 50
SUBSAMPLE_FRAC = 0.8


def f_statistic_per_feature(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask_pos = y > 0.5
    mask_neg = ~mask_pos
    n_pos, n_neg = int(mask_pos.sum()), int(mask_neg.sum())
    n = n_pos + n_neg
    grand = z.mean(axis=0)
    mean_pos = z[mask_pos].mean(axis=0)
    mean_neg = z[mask_neg].mean(axis=0)
    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((z[mask_pos] - mean_pos) ** 2).sum(axis=0) + (
        (z[mask_neg] - mean_neg) ** 2
    ).sum(axis=0)
    ms_within = ss_within / max(n - 2, 1)
    f = np.divide(ss_between, ms_within, out=np.zeros_like(ss_between), where=ms_within > 1e-12)
    return f.astype(np.float64)


def train_classifier_ensemble(
    z_sub: np.ndarray, y: np.ndarray, m: int, subsample_frac: float
) -> np.ndarray:
    """Train M classifiers on random subsamples, return averaged positive-class weights."""
    n = len(y)
    k = z_sub.shape[1]
    weight_sum = np.zeros(k, dtype=np.float64)

    for i in range(m):
        rng = np.random.RandomState(seed=i)
        idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        X_train = z_sub[idx]
        y_train = y[idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=i)
        clf.fit(X_scaled, y_train)

        w_pos = clf.coef_[0] if clf.classes_[1] > 0.5 else -clf.coef_[0]
        weight_sum += w_pos

        if (i + 1) % 10 == 0:
            print(f"  Trained {i+1}/{m} classifiers", flush=True)

    return weight_sum / m


def separation_score_sweep(
    z_sub: np.ndarray, y: np.ndarray, v_avg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep d=1..K, return (d_values, s_values)."""
    k = len(v_avg)
    rank_order = np.argsort(np.abs(v_avg))[::-1]

    d_values = np.arange(1, k + 1)
    s_values = np.zeros(k, dtype=np.float64)

    mask_pos = y > 0.5
    mask_neg = ~mask_pos

    for d_idx in range(k):
        d = d_idx + 1
        active_dims = rank_order[:d]
        v_trunc = np.zeros_like(v_avg)
        v_trunc[active_dims] = v_avg[active_dims]

        v_norm = np.linalg.norm(v_trunc)
        if v_norm < 1e-12:
            continue
        v_unit = v_trunc / v_norm

        cosines = z_sub @ v_unit
        mean_cos_pos = cosines[mask_pos].mean()
        mean_cos_neg = cosines[mask_neg].mean()
        s_values[d_idx] = mean_cos_pos - mean_cos_neg

    return d_values, s_values


def find_elbow(d_values: np.ndarray, s_values: np.ndarray, threshold: float = 0.95) -> int:
    """Smallest d where s(d) >= threshold * max(s)."""
    s_max = s_values.max()
    target = threshold * s_max
    for i, s in enumerate(s_values):
        if s >= target:
            return int(d_values[i])
    return int(d_values[-1])


def ascii_plot(d_values: np.ndarray, s_values: np.ndarray, d_steer: int, width: int = 70, height: int = 20):
    """Print an ASCII plot of s(d) vs d."""
    s_min, s_max = s_values.min(), s_values.max()
    s_range = s_max - s_min if s_max > s_min else 1.0

    # Downsample to width
    step = max(1, len(d_values) // width)
    d_sampled = d_values[::step]
    s_sampled = s_values[::step]

    grid = [[" "] * len(d_sampled) for _ in range(height)]

    for col, s in enumerate(s_sampled):
        row = int((s - s_min) / s_range * (height - 1))
        row = min(max(row, 0), height - 1)
        grid[height - 1 - row][col] = "*"

    # Mark d_steer
    d_steer_col = None
    for col, d in enumerate(d_sampled):
        if d >= d_steer and d_steer_col is None:
            d_steer_col = col
            for r in range(height):
                if grid[r][col] == " ":
                    grid[r][col] = "|"

    print(f"\n  Separation score s(d) vs d   [d_steer={d_steer} marked with |]")
    print(f"  s_max={s_max:.4f}")
    for row in grid:
        print("  " + "".join(row))
    print(f"  d: 1{'':>{len(d_sampled)//2 - 2}}d_steer={d_steer}{'':>{len(d_sampled)//2 - 10}}{d_values[-1]}")


def main():
    print("=== Stage 2 Feasibility Test ===\n", flush=True)

    print(f"Loading z-cache: {Z_CACHE}", flush=True)
    cached = np.load(Z_CACHE)
    z_all, y_all = cached["z"], cached["y"]
    print(f"  Samples: {z_all.shape[0]} ({int((y_all > 0.5).sum())} pos, {int((y_all <= 0.5).sum())} neg)")
    print(f"  SAE dims: {z_all.shape[1]}")

    # Stage 1: F-stat coarse selection
    print(f"\nStage 1: F-stat top-{K_COARSE}...", flush=True)
    f_stats = f_statistic_per_feature(z_all, y_all)
    top_k_idx = np.argsort(f_stats)[::-1][:K_COARSE]
    z_sub = z_all[:, top_k_idx]
    print(f"  Top F-stat range: {f_stats[top_k_idx[0]]:.2f} .. {f_stats[top_k_idx[-1]]:.2f}")

    # Stage 2: Classifier ensemble
    print(f"\nStage 2a: Training {M_CLASSIFIERS} classifiers (80% subsample each)...", flush=True)
    v_avg = train_classifier_ensemble(z_sub, y_all, M_CLASSIFIERS, SUBSAMPLE_FRAC)
    print(f"  v_avg norm: {np.linalg.norm(v_avg):.4f}")
    print(f"  v_avg nonzero (|w| > 0.01): {int((np.abs(v_avg) > 0.01).sum())} / {K_COARSE}")

    # Stage 2: Separation score sweep
    print(f"\nStage 2b: Sweeping d=1..{K_COARSE}...", flush=True)
    d_values, s_values = separation_score_sweep(z_sub, y_all, v_avg)

    d_steer_95 = find_elbow(d_values, s_values, 0.95)
    d_steer_90 = find_elbow(d_values, s_values, 0.90)
    d_steer_99 = find_elbow(d_values, s_values, 0.99)

    print(f"\n=== RESULTS ===")
    print(f"  d_steer (90% of max): {d_steer_90}")
    print(f"  d_steer (95% of max): {d_steer_95}")
    print(f"  d_steer (99% of max): {d_steer_99}")
    print(f"  s(1)   = {s_values[0]:.4f}")
    print(f"  s(5)   = {s_values[4]:.4f}")
    print(f"  s(10)  = {s_values[9]:.4f}")
    print(f"  s(20)  = {s_values[19]:.4f}")
    print(f"  s(50)  = {s_values[49]:.4f}")
    print(f"  s(100) = {s_values[99]:.4f}")
    print(f"  s(200) = {s_values[199]:.4f}")
    print(f"  s(500) = {s_values[499]:.4f}")
    print(f"  s(1024)= {s_values[-1]:.4f}")
    print(f"  max(s) = {s_values.max():.4f} at d={d_values[np.argmax(s_values)]}")

    ascii_plot(d_values, s_values, d_steer_95)

    # Also show top-10 features by v_avg magnitude with their original feature IDs
    rank_order = np.argsort(np.abs(v_avg))[::-1]
    print(f"\nTop 10 features by classifier importance:")
    for i in range(10):
        local_idx = rank_order[i]
        global_fid = top_k_idx[local_idx]
        print(f"  rank {i+1}: fid={global_fid:>6d}  v_avg={v_avg[local_idx]:+.4f}  f_stat={f_stats[global_fid]:.2f}")


if __name__ == "__main__":
    main()
