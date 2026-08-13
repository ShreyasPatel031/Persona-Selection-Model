#!/usr/bin/env python3
"""Parse debug_k_sweep.log and score replies with Vertex Goodness judge."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

def parse_log(text: str) -> list[dict]:
    lines = text.splitlines()
    items: list[dict] = []
    q_idx = -1
    question = ""
    current_label: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal current_label, buf
        if current_label is None:
            buf = []
            return
        reply = "\n".join(buf).strip()
        items.append({
            "q_idx": q_idx,
            "question": question,
            "label": current_label,
            "reply": reply,
        })
        current_label = None
        buf = []

    for line in lines:
        m_q = re.match(r"^Q(\d+): (.+)$", line)
        if m_q:
            flush()
            q_idx = int(m_q.group(1)) - 1
            question = m_q.group(2).strip()
            continue
        m = re.match(r"^\[([^\]]+)\]: (.*)$", line)
        if m:
            flush()
            current_label = m.group(1)
            rest = m.group(2)
            buf = [rest] if rest else []
            continue
        if current_label is not None and line and not line.startswith("INFO ") and not line.startswith("="):
            buf.append(line)
    flush()
    return items


def main():
    log_path = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/debug_k_sweep.log")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "logs/debug_k_sweep_scores.json")

    from app.persona.judge_vertex import judge_rubric_to_instructions, score_transcript
    from app.persona.schemas import PersonaTraitArtifact

    bundle = PersonaTraitArtifact.model_validate_json(
        Path("persona_runs/dnd_good_scale/artifacts/trait_bundle.json").read_text(encoding="utf-8")
    )
    neg_sys = bundle.neg_system_prompt
    judge_instr = judge_rubric_to_instructions(bundle.judge_rubric, trait_label=bundle.trait_label)
    eval_qs = bundle.eval_questions[:5]

    items = parse_log(log_path.read_text(encoding="utf-8"))
    scored: list[dict] = []
    by_label: dict[str, list[int]] = defaultdict(list)

    for it in items:
        label = it["label"]
        if label.startswith("INFO"):
            continue
        q = eval_qs[it["q_idx"]] if it["q_idx"] < len(eval_qs) else it["question"]
        try:
            js = score_transcript(judge_instr, neg_sys, q, it["reply"])
            score = int(js.score)
            reason = js.short_reason
        except Exception as e:
            score = -1
            reason = str(e)[:120]
        row = {**it, "score": score, "short_reason": reason}
        scored.append(row)
        if score >= 0:
            by_label[label].append(score)
        print(f"Q{it['q_idx']+1} [{label}] score={score}  {reason[:80]}")

    summary = {}
    for label, scores in sorted(by_label.items(), key=lambda x: (999999 if x[0].startswith("K=") else 0, x[0])):
        summary[label] = {
            "mean": round(sum(scores) / len(scores), 1),
            "min": min(scores),
            "max": max(scores),
            "n": len(scores),
        }

    doc = {"summary": summary, "rows": scored}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (mean trait 0-100) ===")
    for label in ["BASELINE", "DENSE_CAA", "K=10", "K=50", "K=100", "K=200", "K=500", "K=1000", "K=2000"]:
        if label in summary:
            s = summary[label]
            print(f"  {label:12s}  mean={s['mean']:5.1f}  range=[{s['min']},{s['max']}]  n={s['n']}")
    for label, s in summary.items():
        if label not in {"BASELINE", "DENSE_CAA", "K=10", "K=50", "K=100", "K=200", "K=500", "K=1000", "K=2000"}:
            print(f"  {label:12s}  mean={s['mean']:5.1f}  range=[{s['min']},{s['max']}]  n={s['n']}")

if __name__ == "__main__":
    main()
