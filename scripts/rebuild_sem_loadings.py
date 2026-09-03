#!/usr/bin/env python3
"""Rebuild app/static/big_five_sem_data.json item loadings from final-cycle ladders.

loading = Pearson r(keyed item EV, home-domain EV) across usable administrations
in results/final_cycle/ladder/prompt_ladder_{trait}.json (all five steered traits).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = {
    "N": "neuroticism",
    "E": "extraversion",
    "O": "openness",
    "A": "agreeableness",
    "C": "conscientiousness",
}
FACTOR_IDS = {
    "neuroticism": "N",
    "extraversion": "E",
    "conscientiousness": "C",
    "openness": "O",
    "agreeableness": "A",
}
TRAITS = list(DOMAIN.values())
DEFAULT_TOP_N = 20
FOCUS_TOP_N = 8


def keyed_ev(raw: float, key: int) -> float:
    return (6.0 - raw) if key < 0 else raw


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "data/mpi_120.csv")))
    items_meta = [
        {
            "item": i,
            "facet": r["label_raw"],
            "text": r["text"],
            "domain": DOMAIN[r["label_ocean"]],
            "domain_letter": r["label_ocean"],
            "key": int(r["key"]),
        }
        for i, r in enumerate(rows, 1)
    ]

    records = []
    ladder_dir = ROOT / "results/final_cycle/ladder"
    for trait in TRAITS:
        payload = json.load(open(ladder_dir / f"prompt_ladder_{trait}.json"))
        for admin in payload["administrations"]:
            if not admin.get("usable", True):
                continue
            evs = admin.get("ev_scores") or {}
            item_evs = admin.get("item_log", {}).get("evs")
            if not item_evs or len(item_evs) != 120:
                continue
            if not all(t in evs for t in TRAITS):
                continue
            records.append(
                {
                    "steered": trait,
                    "level": admin.get("level"),
                    "ev_scores": evs,
                    "item_evs": item_evs,
                }
            )

    loadings = []
    for meta in items_meta:
        idx = meta["item"] - 1
        dom = meta["domain"]
        xs = [keyed_ev(r["item_evs"][idx], meta["key"]) for r in records]
        ys = [r["ev_scores"][dom] for r in records]
        r = pearson(xs, ys)
        xs_s, ys_s = [], []
        for rec in records:
            if rec["steered"] != dom:
                continue
            xs_s.append(keyed_ev(rec["item_evs"][idx], meta["key"]))
            ys_s.append(
                float(rec["level"])
                if rec["level"] is not None
                else rec["ev_scores"][dom]
            )
        r_steer = pearson(xs_s, ys_s)
        loadings.append(
            {
                **meta,
                "loading": round(r, 3) if r is not None else 0.0,
                "loading_steer": round(r_steer, 3) if r_steer is not None else None,
            }
        )

    factor_corrs = []
    for i, a in enumerate(TRAITS):
        for b in TRAITS[i + 1 :]:
            xs = [r["ev_scores"][a] for r in records]
            ys = [r["ev_scores"][b] for r in records]
            factor_corrs.append(
                {
                    "a": FACTOR_IDS[a],
                    "b": FACTOR_IDS[b],
                    "r": round(pearson(xs, ys) or 0.0, 3),
                }
            )

    ranked = sorted(loadings, key=lambda x: -abs(x["loading"]))
    default_ids = [x["item"] for x in ranked[:DEFAULT_TOP_N]]

    old_path = ROOT / "app/static/big_five_sem_data.json"
    old = json.load(open(old_path)) if old_path.exists() else {}
    out = {
        "title": old.get("title", "MPI-120 measurement model"),
        "subtitle": (
            "MPI-120 keyed item–domain loadings from final-cycle steered runs · "
            f"default top-{DEFAULT_TOP_N} · focus top-{FOCUS_TOP_N}"
        ),
        "instrument": "mpi_120.csv",
        "citation": old.get("citation"),
        "note": (
            f"loading = Pearson r(keyed item EV, home-domain EV) across final-cycle "
            f"ladder administrations (n={len(records)}). Reverse-keyed items use 6−EV."
        ),
        "source": {
            "ladders": [
                f"results/final_cycle/ladder/prompt_ladder_{t}.json" for t in TRAITS
            ],
            "n_administrations": len(records),
            "loading_def": "corr(keyed_item_ev, domain_ev) pooled over usable admins",
        },
        "layout": old.get(
            "layout",
            {
                "viewBox": [0, 0, 1000, 920],
                "factorRadius": 34,
                "itemSize": 26,
                "errorRadius": 11,
                "itemDist": 128,
                "errorDist": 178,
            },
        ),
        "factors": old.get(
            "factors",
            [
                {"id": "N", "trait": "neuroticism", "label": "N", "x": 500, "y": 210},
                {"id": "E", "trait": "extraversion", "label": "E", "x": 210, "y": 390},
                {"id": "C", "trait": "conscientiousness", "label": "C", "x": 790, "y": 390},
                {"id": "O", "trait": "openness", "label": "O", "x": 330, "y": 660},
                {"id": "A", "trait": "agreeableness", "label": "A", "x": 670, "y": 660},
            ],
        ),
        "items": [
            {
                "item": x["item"],
                "facet": x["facet"],
                "text": x["text"],
                "domain": x["domain"],
                "domain_letter": x["domain_letter"],
                "key": x["key"],
                "loading": x["loading"],
                "loading_steer": x["loading_steer"],
            }
            for x in loadings
        ],
        "default_item_ids": default_ids,
        "focus_top_n": FOCUS_TOP_N,
        "default_top_n": DEFAULT_TOP_N,
        "factor_correlations": factor_corrs,
        "error_correlations": [],
    }
    json.dump(out, open(old_path, "w"), indent=2)
    print(f"wrote {old_path} · n={len(records)} · top{DEFAULT_TOP_N}={default_ids}")


if __name__ == "__main__":
    main()
