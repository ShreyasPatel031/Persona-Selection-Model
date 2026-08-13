"""Tests for canonical rollout path resolution between step-C and step-D."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.persona.rollouts import (
    EXTRACTION_ROLLOUTS_JSON,
    ROLLOUTS_JSONL,
    ROLLOUTS_LATEST_JSON,
    extraction_json_to_rollouts_jsonl,
    resolve_rollouts_jsonl,
    write_rollouts_latest,
)


def _minimal_bundle() -> dict:
    q = "Your neighbor asks for help with an emergency."
    return {
        "schema_version": "1",
        "trait_label": "good",
        "trait_description": "Altruism vs self-interest for tests.",
        "pos_system_prompt": "You are profoundly good and self-sacrificing. " * 3,
        "neg_system_prompt": "You are purely self-interested and rational. " * 3,
        "contrast_scenarios": [q, q + " 2", q + " 3", q + " 4"],
        "extraction_questions": [q + f" {i}" for i in range(8)],
        "eval_questions": [q + f" eval {i}" for i in range(4)],
        "judge_rubric": {
            "task_summary": "Score how well the reply expresses goodness.",
            "criteria": [
                {
                    "name": "goodness",
                    "description": "How good is the reply?",
                    "scale_min": 1,
                    "scale_max": 5,
                },
                {
                    "name": "coherence",
                    "description": "Is the reply coherent?",
                    "scale_min": 1,
                    "scale_max": 5,
                },
                {
                    "name": "fit",
                    "description": "Does it fit the scenario?",
                    "scale_min": 1,
                    "scale_max": 5,
                },
            ],
            "pass_threshold_notes": "High is good.",
        },
    }


def _write_extraction(run_dir: Path, bundle_path: Path, *, mtime: float | None = None) -> Path:
    extraction = run_dir / "rollouts" / EXTRACTION_ROLLOUTS_JSON
    extraction.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "step": "C",
        "kind": "extraction",
        "judge": None,
        "trait_bundle": str(bundle_path.resolve()),
        "questions_source": "scenarios",
        "question_count": 1,
        "items": [
            {
                "index": 0,
                "pair_index": 0,
                "question_index": 0,
                "rollout_index": 0,
                "question": "Your neighbor asks for help.",
                "pos_reply": "Of course I will help.",
                "neg_reply": "I decline; it is inefficient.",
            }
        ],
    }
    extraction.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    if mtime is not None:
        import os

        os.utime(extraction, (mtime, mtime))
    write_rollouts_latest(run_dir, doc)
    return extraction


def test_resolve_rebuilds_jsonl_when_extraction_is_newer(tmp_path: Path) -> None:
    run_dir = tmp_path / "my_run"
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_text(json.dumps(_minimal_bundle()), encoding="utf-8")

    jsonl = run_dir / "rollouts" / ROLLOUTS_JSONL
    jsonl.parent.mkdir(parents=True)
    jsonl.write_text('{"arm":"pos","kept":true,"score":1}\n', encoding="utf-8")
    old_mtime = time.time() - 100
    import os

    os.utime(jsonl, (old_mtime, old_mtime))

    _write_extraction(run_dir, bundle_path, mtime=time.time())

    resolved, source = resolve_rollouts_jsonl(run_dir)
    assert source == EXTRACTION_ROLLOUTS_JSON
    assert resolved == jsonl.resolve()
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    latest = json.loads((run_dir / "rollouts" / ROLLOUTS_LATEST_JSON).read_text())
    assert latest["extraction_rollouts_json"] == EXTRACTION_ROLLOUTS_JSON
    assert latest["rollouts_jsonl"] == ROLLOUTS_JSONL


def test_extraction_json_to_rollouts_jsonl_marks_all_kept(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bundle_path = run_dir / "artifacts" / "trait_bundle.json"
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_text(json.dumps(_minimal_bundle()), encoding="utf-8")
    extraction = _write_extraction(run_dir, bundle_path)
    jsonl = run_dir / "rollouts" / ROLLOUTS_JSONL
    extraction_json_to_rollouts_jsonl(extraction, bundle_path, jsonl)
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(r["kept"] for r in rows)
    assert {r["arm"] for r in rows} == {"pos", "neg"}


def test_resolve_missing_rollouts_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_rollouts_jsonl(tmp_path / "empty_run")
