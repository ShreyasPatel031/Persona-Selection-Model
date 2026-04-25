"""HTTP client for Gemma FastAPI /chat."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def chat_nonstream(
    base_url: str,
    message: str,
    system: str,
    *,
    timeout: int = 720,
    do_sample: bool = False,
    temperature: float | None = None,
    seed: int | None = None,
) -> str:
    url = base_url.rstrip("/") + "/chat"
    payload_obj: dict[str, object] = {
        "message": message,
        "system": system,
        "do_sample": do_sample,
    }
    if temperature is not None:
        payload_obj["temperature"] = float(temperature)
    if seed is not None:
        payload_obj["seed"] = int(seed)
    payload = json.dumps(payload_obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    return (data.get("reply") or "").strip()
