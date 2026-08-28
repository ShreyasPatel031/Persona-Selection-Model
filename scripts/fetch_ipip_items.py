#!/usr/bin/env python3
"""Fetch the public-domain IPIP-NEO facet item pool and build a balanced 120-item form.

Source: https://ipip.ori.org/newNEO_FacetsKey.htm — the "Items in Each of the
Preliminary IPIP Scales Measuring Constructs Similar to Those in the 30 NEO-PI-R
Facet Scales" page, which lists all 300 items grouped by facet with +/- keying.
IPIP items are explicitly public domain (https://ipip.ori.org/newPermission.htm).

Two outputs:
  data/ipip_neo_facets_300.csv  full pool, verbatim, with domain/facet/key
  data/ipip_neo_120.csv         deterministic keying-balanced 120-item form

The 120-item form takes 4 items per facet (30 facets x 4), preferring 2 plus-keyed
and 2 minus-keyed so acquiescence correction is well defined and a degenerate
uniform response is detectable. This is an IPIP-derived instrument, not a
reproduction of the published IPIP-NEO-120 item selection; provenance is recorded
in the CSV header comment and in the sidecar JSON.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

FACETS_URL = "https://ipip.ori.org/newNEO_FacetsKey.htm"

# Facet code prefix -> OCEAN domain letter.
DOMAIN_OF_PREFIX = {"N": "N", "E": "E", "O": "O", "A": "A", "C": "C"}

# Facet headers are "N4: SELF-CONSCIOUSNESS (.80)", but the alpha often wraps to the
# next line, so the trailing paren cannot be required. All-caps names separate
# headers from items, which are sentence case.
FACET_RE = re.compile(r"\b([NEOAC])([1-6])\s*:\s*([A-Z][A-Z \-']{2,})")
KEY_RE = re.compile(r"([+\u2013\u2212-])\s*keyed", re.I)


def fetch(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "persona-selection-model/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def to_text(raw_html: str) -> str:
    """Strip tags but keep row breaks so item boundaries survive."""
    t = re.sub(r"(?i)</\s*(tr|p|div|table)\s*>", "\n", raw_html)
    t = re.sub(r"(?i)<\s*br\s*/?>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(line.strip() for line in t.split("\n"))


ITEM_END_RE = re.compile(r"\.\s*[\"\u201d]?$")


def item_complete(line: str) -> bool:
    """Items terminate with a period, sometimes inside a closing quote."""
    return bool(ITEM_END_RE.search(line))


def looks_like_item(line: str) -> bool:
    """IPIP items are short self-descriptive phrases ending in a period."""
    if len(line) < 6 or len(line) > 120:
        return False
    if not item_complete(line):
        return False
    if FACET_RE.search(line) or KEY_RE.search(line):
        return False
    if re.search(r"(?i)\b(alpha|keyed|scale|items?|alph)\b\s*[=:]", line):
        return False
    if line.count(".") > 2:
        return False
    return bool(re.match(r"^[A-Z][A-Za-z'\"\u201c ,\-]", line))


def is_continuation(line: str) -> bool:
    """A wrapped item tail, e.g. 'punishment.' after '...help rather than'."""
    return bool(re.match(r"^[a-z][A-Za-z'\"\u201d ,\-]*", line)) and len(line) <= 60


def parse_pool(text: str) -> list[dict]:
    """Walk the flattened page, tracking current facet and current keying."""
    rows: list[dict] = []
    facet_code = facet_name = None
    key = None
    pending = ""
    seen: set[tuple[str, str]] = set()

    for line in text.split("\n"):
        if not line:
            continue
        fm = FACET_RE.search(line)
        if fm:
            letter, num, name = fm.group(1), fm.group(2), fm.group(3)
            facet_code = f"{letter}{num}"
            facet_name = " ".join(name.split()).title()
            key = None
            pending = ""
            # A facet header line can also carry the first "+ keyed" marker.
            km = KEY_RE.search(line)
            if km:
                key = 1 if km.group(1) == "+" else -1
            continue
        km = KEY_RE.search(line)
        if km:
            key = 1 if km.group(1) == "+" else -1
            pending = ""
            tail = line[km.end() :].strip()
            if looks_like_item(tail):
                line = tail
            else:
                continue
        if facet_code is None or key is None:
            continue
        # Long items wrap across table cells; stitch the tail back on.
        if pending:
            joined = f"{pending} {line}".strip()
            pending = ""
            if looks_like_item(joined):
                line = joined
            elif not looks_like_item(line):
                continue
        elif not looks_like_item(line):
            if re.match(r"^[A-Z]", line) and not item_complete(line) and 6 <= len(line) <= 120:
                pending = line
            continue
        sig = (facet_code, line)
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(
            {
                "text": line,
                "domain": DOMAIN_OF_PREFIX[facet_code[0]],
                "facet_code": facet_code,
                "facet": facet_name,
                "key": key,
            }
        )
    return rows


def build_120(pool: list[dict], per_facet: int = 4) -> list[dict]:
    """4 items per facet, preferring a 2/2 plus/minus split, deterministic order."""
    by_facet: dict[str, list[dict]] = {}
    for r in pool:
        by_facet.setdefault(r["facet_code"], []).append(r)

    out: list[dict] = []
    half = per_facet // 2
    for code in sorted(by_facet, key=lambda c: (c[0], int(c[1:]))):
        items = by_facet[code]
        plus = [r for r in items if r["key"] == 1]
        minus = [r for r in items if r["key"] == -1]
        pick = plus[:half] + minus[:half]
        # Facets with lopsided keying (e.g. N3 has 7/3) fall back to filling from
        # whichever pole has spares, so every facet still contributes per_facet items.
        if len(pick) < per_facet:
            spare = [r for r in plus[half:] + minus[half:] if r not in pick]
            pick += spare[: per_facet - len(pick)]
        out.extend(pick[:per_facet])
    return out


def write_csv(path: Path, rows: list[dict], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {note}\n")
        w = csv.DictWriter(f, fieldnames=["text", "domain", "facet_code", "facet", "key"])
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=FACETS_URL)
    p.add_argument("--pool-csv", type=Path, default=Path("data/ipip_neo_facets_300.csv"))
    p.add_argument("--form-csv", type=Path, default=Path("data/ipip_neo_120.csv"))
    p.add_argument("--per-facet", type=int, default=4)
    p.add_argument("--offline-html", type=Path, default=None, help="Parse a saved copy instead of fetching.")
    args = p.parse_args(argv)

    raw = args.offline_html.read_text(encoding="utf-8", errors="replace") if args.offline_html else fetch(args.url)
    pool = parse_pool(to_text(raw))
    if not pool:
        print("Parsed zero items — page layout changed?", file=sys.stderr)
        return 1

    facets = sorted({r["facet_code"] for r in pool})
    by_domain: dict[str, int] = {}
    for r in pool:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    print(f"pool: {len(pool)} items across {len(facets)} facets; per domain {by_domain}")

    form = build_120(pool, args.per_facet)
    fd: dict[str, int] = {}
    fk: dict[int, int] = {}
    for r in form:
        fd[r["domain"]] = fd.get(r["domain"], 0) + 1
        fk[r["key"]] = fk.get(r["key"], 0) + 1
    print(f"form: {len(form)} items; per domain {fd}; keying {fk}")

    prov = f"IPIP-NEO facet item pool, public domain, fetched from {args.url}"
    write_csv(args.pool_csv, pool, prov)
    write_csv(
        args.form_csv,
        form,
        f"{prov}; {args.per_facet} items/facet, keying-balanced where the pool allows",
    )
    meta = {
        "source_url": args.url,
        "license": "public domain (https://ipip.ori.org/newPermission.htm)",
        "pool_items": len(pool),
        "pool_facets": len(facets),
        "form_items": len(form),
        "form_per_facet": args.per_facet,
        "form_per_domain": fd,
        "form_keying": {str(k): v for k, v in fk.items()},
        "note": "IPIP-derived form; not the published IPIP-NEO-120 item selection",
    }
    args.form_csv.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.pool_csv} and {args.form_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
