# Inferno visualization design system

**Canonical source:** `app/static/inferno_cone.html` (“Nine Alphas of Evil”)

All persona / steering visualizations in this project should use this system — cream shell, Newsreader titles, Instrument Sans UI, hairline rules, YES/NO (good/evil) semantic colors. Do not invent new palettes (no purple-on-white dashboards, no dark fantasy chrome, no SAE-viz blue tags) unless extending these tokens deliberately.

---

## Tokens (CSS `:root`)

```css
:root {
  --bg: #f6f3ee;
  --ink: #1a1612;
  --muted: #7a7268;
  --line: #e4ddd3;
  --no: #c45c4a;       /* refuse / evil / negative */
  --no-dim: #e8b4aa;
  --yes: #3d8b6e;      /* accept / good / positive */
  --yes-dim: #a8d4c2;
}
```

### Optional axis extensions (composition / D&D)

When a viz needs Lawful / Chaotic as well as Good / Evil, keep Good/Evil as `--yes` / `--no` and add muted companions that still sit on the cream field:

```css
  --lawful: #4a6fa5;   /* cool ink-blue, not neon */
  --lawful-dim: #c5d4e8;
  --chaotic: #b7791f;  /* warm amber (matches layer3d chaotic) */
  --chaotic-dim: #e8d4a8;
  --panel: #fffefb;    /* slightly lifted from --bg for cards */
```

Blend corners by soft linear-gradients of the two axis dims — never loud multi-stop rainbows.

---

## Typography

| Role | Font | Spec |
|------|------|------|
| Page title (`h1`) | **Newsreader** | ~1.45rem, weight 600, `letter-spacing: -0.02em` |
| Section labels | Instrument Sans | ~0.68–0.72rem, weight 700, uppercase, `letter-spacing: 0.06–0.1em`, color `--muted` |
| Body / UI | **Instrument Sans** | system-ui fallback; default weight 400–600 |
| Scenario / cell titles | Instrument Sans | ~0.95–1.05rem, weight 600, line-height ~1.35 |
| Fine print / hints | Instrument Sans | ~0.6–0.78rem, `--muted` |
| Numerics (α) | Instrument Sans | `font-variant-numeric: tabular-nums` |

Google Fonts import:

```html
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet" />
```

---

## Surfaces & chrome

- **Page background:** `--bg` only. No gradients, glows, or textured overlays.
- **Dividers:** `1px solid var(--line)` — hairlines between header / columns / rows.
- **Cards / panels:** optional white-ish `--panel` with `1px solid var(--line)` and **3–4px** radius (Inferno rungs use 3px badges). Prefer open layout with hairlines over heavy card stacks.
- **No** drop shadows by default; if needed, max `0 2px 10px rgba(26,22,18,0.06)`.
- **No** pill clusters, purple accents, or gold fantasy borders.

---

## Semantic YES / NO

From Inferno rungs:

| State | Text | Border | Background |
|-------|------|--------|------------|
| OFF / NO / refuse | `--no` | `--no-dim` | transparent |
| ON / YES / accept | `--yes` | `--yes-dim` | `rgba(61, 139, 110, 0.1)` |

Active row wash: `background: rgba(61, 139, 110, 0.06)`.

Inactive content often sits at `opacity: 0.5` until “on”.

---

## Controls

- Range tracks: height `2px`, color `--line`.
- Thumbs: 14×14 circle, fill `--ink`, `border: 2px solid var(--bg)`, `box-shadow: 0 0 0 1px var(--ink)`.
- Select / inputs: Instrument Sans, `1px solid var(--line)`, radius 3–4px, focus outline in `--ink` (not blue).

---

## Layout habits

1. **Header** full-bleed: title + one muted subtitle line; optional α readout under it.
2. **One job per region** — ladder / grid left, detail or stage right.
3. **Hint footer** optional: 0.6rem centered muted text above a top hairline.
4. Prefer full-viewport apps (`height: 100%`; `overflow: hidden`) for interactive stages; document-style pages may scroll with `max-width` ~68–72rem and cream gutters.

---

## Alignment composition board mapping

| Axis | Token |
|------|--------|
| Good row / NG | `--yes` fills / borders |
| Evil row / NE | `--no` fills / borders |
| Lawful column / LN | `--lawful` |
| Chaotic column / CN | `--chaotic` |
| True Neutral | white / `--panel`, `--line` border, `--ink` text |
| LG / CG / LE / CE | soft gradient of both axis dims |

Cell labels: full names (“Lawful Good”), not LG/NG. Phrases secondary in `--muted`. Reply body: smaller, `--muted` grey — never competing with the grid.

---

## Reference files

| File | Role |
|------|------|
| `app/static/inferno_cone.html` | Canonical interactive reference |
| `app/static/layer3d.html` | Same tokens + lawful/chaotic extensions |
| `app/static/dnd_composition_board.html` | Composition 9-grid (must match this system) |
| `app/static/big_five_persona.html` | Series finale — OCEAN inventory silhouette |
| `app/static/viz_series.html` | Ordered chapter index for the Inferno set |
| `viz/DESIGN_SYSTEM.md` | This document |

When adding a new viz under `app/static/`, copy the `:root` block and fonts from Inferno first, then build.
