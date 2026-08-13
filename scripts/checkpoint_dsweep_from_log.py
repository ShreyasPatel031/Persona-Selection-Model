#!/usr/bin/env python3
"""Parse ssv_stage2_test log output and write incremental checkpoint JSON.

Used alongside a long-running d-sweep when the process predates --resume support.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCORE_RE = re.compile(r"^\s+\[(?P<label>[^\]]+)\] Q(?P<q>\d+) score=(?P<score>\S+)")
MEAN_RE = re.compile(r"^\s+\[(?P<label>[^\]]+)\] MEAN=(?P<mean>\S+)")
SECTION_RE = re.compile(r"^=== (?P<label>.+) ===$")


def parse_label(label: str) -> dict:
    if label == "BASELINE":
        return {"label": label, "method": "baseline"}
    if label == "DENSE_CAA":
        return {"label": label, "method": "dense_caa"}
    m = re.match(r"d(\d+)_(fstat|classifier)", label)
    if m:
        return {
            "label": label,
            "method": "sae_ssv",
            "d": int(m.group(1)),
            "ranking": m.group(2),
        }
    return {"label": label}


def parse_log(text: str) -> list[dict]:
    current: str | None = None
    scores: dict[str, list] = {}
    means: dict[str, float | None] = {}

    for line in text.splitlines():
        sec = SECTION_RE.match(line.strip())
        if sec:
            current = sec.group("label").strip()
            scores.setdefault(current, [])
            continue
        if current is None:
            continue
        sm = SCORE_RE.match(line)
        if sm and sm.group("label") == current:
            raw = sm.group("score")
            val = None if raw == "None" else int(raw)
            qi = int(sm.group("q")) - 1
            while len(scores[current]) <= qi:
                scores[current].append(None)
            scores[current][qi] = val
            continue
        mm = MEAN_RE.match(line)
        if mm and mm.group("label") == current:
            raw = mm.group("mean")
            means[current] = None if raw == "None" else float(raw)

    results: list[dict] = []
    for label in means:
        entry = parse_label(label)
        entry["scores"] = scores.get(label, [])
        entry["mean"] = means[label]
        results.append(entry)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    if not log_path.exists():
        print(f"log missing: {log_path}", file=sys.stderr)
        sys.exit(1)

    results = parse_log(log_path.read_text(errors="replace"))
    payload = {
        "trait": args.trait,
        "layer": args.layer,
        "alpha": args.alpha,
        "checkpoint": True,
        "checkpoint_source": "log",
        "completed_labels": [r["label"] for r in results],
        "results": results,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    print(f"checkpoint: {len(results)} conditions -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
