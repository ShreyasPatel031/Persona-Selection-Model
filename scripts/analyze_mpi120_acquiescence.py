#!/usr/bin/env python3
"""Check whether MPI-120 steering deltas are trait effects or acquiescence.

MPI-120 dimensions differ in how many items are reverse-keyed, so a model that
simply agrees more shifts each dimension by a fixed, keying-determined amount.
Comparing observed deltas against that "answer A to everything" baseline
separates real trait steering from response bias.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

LETTER_TO_RAW = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def score(letter: str, key: int) -> int:
    raw = LETTER_TO_RAW[letter]
    return raw if key == 1 else 6 - raw


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (
        sum((x - ma) ** 2 for x in a) ** 0.5 * sum((y - mb) ** 2 for y in b) ** 0.5
    )
    return num / den if den else float("nan")


def main(argv: list[str]) -> int:
    csv_path = Path(argv[1]) if len(argv) > 1 else Path("data/mpi_120.csv")
    reports = argv[2:] or [
        "notebooks/ocean_mpi120_eval.json",
        "notebooks/ocean_mpi120_retune.json",
    ]

    rows = list(csv.DictReader(csv_path.open()))
    all_a = {
        d: statistics.mean(
            [score("A", int(r["key"])) for r in rows if r["label_ocean"] == d]
        )
        for d in "OCEAN"
    }
    keying = {
        d: (
            sum(1 for r in rows if r["label_ocean"] == d and int(r["key"]) == 1),
            sum(1 for r in rows if r["label_ocean"] == d and int(r["key"]) == -1),
        )
        for d in "OCEAN"
    }
    print("items per dimension (positively keyed / reverse keyed):")
    for d in "OCEAN":
        print(f"  {d}: {keying[d][0]:2d} / {keying[d][1]:2d}   all-A score {all_a[d]:.3f}")

    for path in reports:
        p = Path(path)
        if not p.is_file():
            print(f"\n[skip] {path} not found")
            continue
        data = json.loads(p.read_text())
        means = data.get("final_means") or data.get("summary_means")
        base = means["baseline"]
        obs, pred = [], []
        for cond, m in means.items():
            if cond == "baseline":
                continue
            for dim in "OCEAN":
                obs.append(m[dim] - base[dim])
                pred.append(all_a[dim] - base[dim])
        same = sum(1 for o, q in zip(obs, pred) if o * q > 0)
        print(f"\n=== {p.name} ===")
        print(f"  corr(observed delta, acquiescence-predicted delta) = {pearson(obs, pred):+.3f}")
        print(f"  same sign: {same}/{len(obs)} ({same / len(obs) * 100:.0f}%)")
        for cond, meta in (data.get("final") or {}).items():
            hist = meta.get("letter_hist")
            if not hist:
                continue
            n = sum(hist.values())
            agree = (hist.get("A", 0) + hist.get("B", 0)) / n
            print(f"  {cond:24} agreement rate {agree:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
