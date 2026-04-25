#!/usr/bin/env python3
"""
Sequential POST /chat/persona-steer for several α values; report per-request latency
and gaps between consecutive requests (client-side; next starts right after prior ends).

Usage (server must be up, e.g. uvicorn on 8080):
  python scripts/bench_persona_steer_alphas.py
  python scripts/bench_persona_steer_alphas.py --base-url http://127.0.0.1:8000
  python scripts/bench_persona_steer_alphas.py --alphas 0.5,1.0,3.5,4.8,9.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post_persona_steer(base_url: str, message: str, steer_alpha: float, timeout: float) -> tuple[dict, float]:
    url = f"{base_url.rstrip('/')}/chat/persona-steer"
    body = json.dumps({"message": message, "steer_alpha": steer_alpha}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - t0
    return json.loads(raw), elapsed


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark sequential persona-steer alphas.")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="API root (default: %(default)s — match your uvicorn port)",
    )
    p.add_argument(
        "--alphas",
        default="0.5,1.0,3.5,4.8,9.1",
        help="Comma-separated α list (default includes 3.5, 4.8, 9.1)",
    )
    p.add_argument(
        "--message",
        default="I think 2+2=5. Was I right? Reply briefly.",
        help="User message for each request",
    )
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request HTTP timeout (s)")
    args = p.parse_args()

    alphas: list[float] = []
    for part in args.alphas.split(","):
        part = part.strip()
        if not part:
            continue
        alphas.append(float(part))

    if len(alphas) < 1:
        print("Need at least one alpha.", file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    print(f"Base URL: {base}", flush=True)
    print(f"Alphas ({len(alphas)}): {alphas}", flush=True)
    print(flush=True)

    gaps_ms: list[float] = []
    latencies: list[float] = []
    prev_end = time.perf_counter()

    for i, alpha in enumerate(alphas):
        gap = time.perf_counter() - prev_end
        gaps_ms.append(gap * 1000.0)
        try:
            data, elapsed = post_persona_steer(base, args.message, alpha, args.timeout)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"α={alpha}: HTTP {e.code} {err_body[:500]}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(
                f"α={alpha}: connection failed: {e.reason}\n"
                f"  Start the server (e.g. uvicorn app.main:app --host 127.0.0.1 --port 8080)\n"
                f"  and use --base-url with the same host/port.",
                file=sys.stderr,
            )
            return 1

        latencies.append(elapsed)
        reply_preview = (data.get("reply") or "")[:120].replace("\n", " ")
        print(f"#{i+1} α={alpha:g}")
        if i == 0:
            print(f"    ms since script start:    {gaps_ms[-1]:.2f} ms")
        else:
            print(f"    ms since prev response:   {gaps_ms[-1]:.2f} ms  (back-to-back overhead)")
        print(f"    request latency:          {elapsed:.3f} s")
        print(f"    reply preview:            {reply_preview!r}{'…' if len(data.get('reply') or '') > 120 else ''}")
        prev_end = time.perf_counter()

    total = sum(latencies)
    print()
    print("Summary")
    print(f"  Sum of request latencies: {total:.3f} s")
    print(f"  Mean latency:             {total / len(latencies):.3f} s")
    print(
        "  Gaps between back-to-back requests (client): "
        f"min={min(gaps_ms[1:]):.2f} ms, max={max(gaps_ms[1:]):.2f} ms"
        if len(gaps_ms) > 1
        else "  (single request)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
